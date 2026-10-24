"""Keyboard-driven LiteGrip gripper for direct kinesthetic collection.

The arm stays hand-guided; the gripper is the only device this recorder
commands. :mod:`litearm_vla.record` turns key presses into open and grasp
requests, and records the measured opening like the arm joints: each frame's
action is the opening measured at the next sample, so the gripper column stays
continuous while the jaws move.

The controller keeps the CAN link busy for the whole session. A DM4310 latches
a communication-loss fault about 100 ms after the frames stop, so between
moves the worker streams zero-displacement hold frames instead of leaving the
motor idle.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import litegrip

log = logging.getLogger(__name__)

#: Feature name added to both the observation and the action when enabled.
GRIPPER_FEATURE = "gripper.pos"

#: Openings this far outside the calibrated travel are treated as a fault.
#: A power grasp legitimately drives past the closed calibration limit: the
#: SDK's own hold guard allows ``MotionConfig.hold_runaway_rad`` (0.05 rad,
#: about 3.1 mm at the shipped 61.01 mm/rad scale) before it releases. A
#: tighter host-side margin turns that normal grasp excursion into a false
#: safety trip, so the default leaves a little room above the SDK's bound.
POSITION_TOLERANCE_MM = 4.0

#: Speed limit for the SDK force setpoint (rated force is 40 N).
MAX_GRASP_FORCE_N = 40.0

#: Consecutive slices a grasp may end in free travel before the host gives up
#: and holds the measured position. The SDK reports that outcome as
#: ``ok=False, reached=False, stalled=False`` after it has already released
#: the force, so nothing is left energized: retrying simply starts another
#: bounded close ramp from where the jaws stopped.
GRASP_RETRIES = 3


class GripperError(RuntimeError):
    """The gripper session no longer satisfies its safety contract."""


@dataclass(kw_only=True)
class LiteGripOptions:
    """LiteGrip connection and keyboard-control options."""

    #: Opt in to the gripper and its motor. When false, no CAN device is
    #: opened and the dataset has no gripper feature.
    enabled: bool = False
    #: SocketCAN interface, brought up at 1 Mbit/s before the session.
    channel: str = "can0"
    can_id: int = 0x08
    #: ``normal``, ``reverse``, or ``None`` to take the direction from the
    #: calibration.
    mount: str | None = None
    #: Calibration file. ``None`` loads the SDK's shipped factory calibration
    #: (a path in ``$LITEGRIP_CALIB`` still wins inside the SDK).
    calibration_file: str | None = None
    #: Open-side target in millimetres. ``None`` uses the SDK work stroke when
    #: the calibration sets one, otherwise the calibrated open limit.
    open_position_mm: float | None = None
    #: Keyboard keys, one printable character each.
    open_key: str = "o"
    close_key: str = "c"
    speed_mm_s: float = 50.0
    #: Closing goes through ``grasp()`` so an object is held with force rather
    #: than released by the position ramp's stall protection.
    grasp_force_n: float = 20.0
    #: A grasp is split into slices this long, so a new key press is honoured
    #: within roughly this delay instead of waiting for an unbounded hold.
    grasp_hold_s: float = 0.2
    max_temperature_c: float = 70.0
    feedback_timeout_s: float = 0.25
    #: How far outside the calibrated 0-to-travel range a measured opening may
    #: sit. Must not be tighter than the SDK force-hold guard, which allows a
    #: closed-side excursion of ``hold_runaway_rad`` (~3.1 mm) during a grasp.
    position_tolerance_mm: float = POSITION_TOLERANCE_MM
    #: Hold-frame rate between moves. Must exceed the motor's ~100 ms
    #: communication-loss timeout.
    hold_hz: float = 200.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if not str(self.channel).strip():
            raise ValueError("gripper.channel must not be empty")
        if not isinstance(self.can_id, int) or not 0 <= self.can_id <= 0x7FF:
            raise ValueError("gripper.can_id must be a classic CAN ID (0-0x7FF)")
        if self.calibration_file is not None and not str(self.calibration_file).strip():
            raise ValueError("gripper.calibration_file must not be empty when set")
        if self.mount not in (None, "normal", "reverse"):
            raise ValueError("gripper.mount must be 'normal', 'reverse', or None")
        for name in ("speed_mm_s", "grasp_force_n", "grasp_hold_s", "hold_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"gripper.{name} must be finite and greater than zero")
        if self.grasp_force_n > MAX_GRASP_FORCE_N:
            raise ValueError(
                f"gripper.grasp_force_n must not exceed {MAX_GRASP_FORCE_N:.0f} N"
            )
        if self.hold_hz < 20:
            raise ValueError(
                "gripper.hold_hz must be at least 20; the motor faults about "
                "100 ms after the control frames stop"
            )
        if not math.isfinite(self.max_temperature_c) or self.max_temperature_c <= 0:
            raise ValueError("gripper.max_temperature_c must be finite and positive")
        if (
            not math.isfinite(self.feedback_timeout_s)
            or self.feedback_timeout_s <= 0
        ):
            raise ValueError("gripper.feedback_timeout_s must be finite and positive")
        if (
            not math.isfinite(self.position_tolerance_mm)
            or self.position_tolerance_mm <= 0
        ):
            raise ValueError("gripper.position_tolerance_mm must be finite and positive")
        if self.open_position_mm is not None:
            if (
                not math.isfinite(self.open_position_mm)
                or self.open_position_mm <= 0
            ):
                raise ValueError(
                    "gripper.open_position_mm must be finite and greater than zero"
                )
        for name in ("open_key", "close_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 1 or not value.isprintable():
                raise ValueError(f"gripper.{name} must be one printable character")
        if self.open_key.lower() == self.close_key.lower():
            raise ValueError("gripper.open_key and gripper.close_key must differ")


@dataclass(frozen=True)
class GripperSample:
    """One measured opening.

    The recorder pairs it with the previous observation exactly like the arm
    joints: ``observation`` carries this opening and the next sample's opening
    becomes ``action[t] = q[t+1]``.
    """

    position_mm: float


class LiteGripController:
    """Own one LiteGrip CAN session and its background hold stream."""

    def __init__(self, options: LiteGripOptions) -> None:
        self.options = options
        self._gripper: Any | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._command: str | None = None
        self._applied_command: str | None = None
        self._open_mm = 0.0
        self._travel_mm = 0.0

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._gripper is not None:
            return
        options = self.options
        options.validate()
        gripper = litegrip.LiteGrip(
            channel=options.channel,
            can_id=options.can_id,
            mount=options.mount,
        )
        try:
            gripper.connect()
            loaded = gripper.load_calibration(path=options.calibration_file)
            if not loaded or not gripper.config.calibrated:
                raise GripperError(
                    "LiteGrip calibration is unavailable; the motion actions "
                    "refuse to run on placeholder limits"
                )
            result = gripper.enable()
            if not result.ok:
                status = getattr(getattr(result, "state", None), "error_code", None)
                raise GripperError(
                    f"LiteGrip did not enable (tries={result.tries}, "
                    f"status={status}); check that {options.channel} is UP at "
                    f"1 Mbit/s: ip -details link show {options.channel}"
                )
            # An idle motor answers control frames only. Ask for one status
            # frame before trusting the first snapshot.
            refresh = getattr(gripper, "refresh_status", None)
            if refresh is not None:
                refresh()
            state = gripper.get_state(wait=True)
            self._travel_mm = self._travel(gripper)
            self._open_mm = self._resolve_open_mm(gripper)
            self._validate_state(state)
        except BaseException:
            self._close_partial(gripper)
            raise

        self._gripper = gripper
        self._stop.clear()
        self._error = None
        self._command = None
        self._applied_command = None
        self._thread = threading.Thread(
            target=self._run, name="litegrip-control", daemon=True
        )
        self._thread.start()
        log.info(
            "LiteGrip ready: %s travel=%.1f mm open=%.1f mm",
            options.channel,
            self._travel_mm,
            self._open_mm,
        )

    def disconnect(self) -> None:
        gripper, self._gripper = self._gripper, None
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=max(5.0, 2.0 / self.options.hold_hz))
        error = self._error
        if gripper is not None:
            try:
                gripper.stop()
            except BaseException as exc:  # noqa: BLE001 - still close the link
                error = error or exc
                log.exception("Could not send the LiteGrip zero-torque frame")
            try:
                gripper.disconnect()
            except BaseException as exc:  # noqa: BLE001 - report the first cause
                error = error or exc
                log.exception("Could not close the LiteGrip CAN session")
        if error is not None:
            raise GripperError(f"unsafe LiteGrip shutdown: {error}") from error

    def fail_safe(self, cause: BaseException) -> None:
        """Stop commanding and take the force off without disabling the motor.

        Called from the safety monitor, so it never joins the worker: the
        worker notices the stop flag at its next frame or grasp slice.
        """
        if self._error is None:
            self._error = cause
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        gripper = self._gripper
        if gripper is not None:
            try:
                gripper.stop()
            except BaseException:  # noqa: BLE001 - the original cause is reported
                log.exception("Could not send the LiteGrip zero-torque frame")

    # -- keyboard commands -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._gripper is not None

    def request_open(self) -> None:
        self._request("open")

    def request_close(self) -> None:
        self._request("close")

    def _request(self, command: str) -> None:
        if self._gripper is None:
            raise GripperError("LiteGrip is not connected")
        self._raise_if_failed()
        with self._condition:
            # Auto-repeat while the same command is already applied is a no-op;
            # otherwise open()/grasp() would restart on every key repeat.
            if self._command == command or (
                self._command is None and self._applied_command == command
            ):
                return
            self._command = command
            self._condition.notify()

    # -- feedback ----------------------------------------------------------

    def capture_state(self) -> GripperSample:
        self._raise_if_failed()
        gripper = self._gripper
        if gripper is None:
            raise GripperError("LiteGrip is not connected")
        state = gripper.get_state(wait=False)
        self._validate_state(state)
        return GripperSample(position_mm=float(state.position_mm))

    def assert_safe(self) -> None:
        self._raise_if_failed()
        gripper = self._gripper
        if gripper is None:
            raise GripperError("LiteGrip is not connected")
        self._validate_state(gripper.get_state(wait=False))

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._condition:
                    if self._stop.is_set():
                        return
                    command, self._command = self._command, None
                if command is None:
                    # Hold from the moment the session connects, not only after
                    # the first key press. The monitor checks feedback every
                    # monitor period, and a DM4310 answers control frames rather
                    # than reporting on its own: an idle link goes stale faster
                    # than feedback_timeout_s and trips the monitor.
                    self._hold_until_command()
                else:
                    self._apply(command)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            self._error = exc
            log.exception("LiteGrip worker stopped")
            self._stop.set()

    def _apply(self, command: str) -> None:
        gripper = self._gripper
        if gripper is None:
            return
        self._applied_command = command
        if command == "open":
            if self.options.open_position_mm is None:
                result = gripper.open(speed_mm_s=self.options.speed_mm_s)
            else:
                result = gripper.move_at_speed(
                    self._open_mm, speed_mm_s=self.options.speed_mm_s
                )
            if not result.ok:
                log.warning(
                    "LiteGrip open did not reach its target: reached=%s stalled=%s",
                    result.reached,
                    result.stalled,
                )
        else:
            self._grasp_until_command()
        self._hold_until_command()

    def _grasp_until_command(self) -> None:
        """Close with force in bounded slices so a key press can interrupt."""
        gripper = self._gripper
        if gripper is None:
            return
        failures = 0
        while not self._stop.is_set():
            with self._condition:
                if self._command is not None:
                    return
            result = gripper.grasp(
                force_n=self.options.grasp_force_n,
                hold_s=self.options.grasp_hold_s,
            )
            if result.ok or result.reached or result.stalled:
                failures = 0
                continue
            # The closing segment ended with the jaws still in free travel, so
            # the SDK released the force. That is a grasp that did not establish
            # contact, not a broken controller: the position it stopped at is
            # still safe. Continue the close with another bounded ramp, and if
            # it keeps failing, keep the link alive with a zero-displacement
            # hold instead of stopping the worker (which would let feedback go
            # stale and trip the safety monitor).
            failures += 1
            if failures < GRASP_RETRIES:
                log.warning(
                    "LiteGrip grasp did not reach the object or the closed "
                    "stop; retrying (%d/%d)",
                    failures,
                    GRASP_RETRIES,
                )
                continue
            position = float(result.state.position_mm)
            if not math.isfinite(position):
                raise GripperError(
                    "LiteGrip reported a non-finite opening after a failed grasp"
                )
            # Let a fresh close press restart the grasp instead of being
            # swallowed as an auto-repeat of the command that just failed.
            self._applied_command = None
            log.warning(
                "LiteGrip grasp did not establish contact after %d slices; "
                "holding at %.1f mm (press %r to retry)",
                failures,
                position,
                self.options.close_key,
            )
            return

    def _hold_until_command(self) -> None:
        period_s = 1.0 / self.options.hold_hz
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._condition:
                if self._command is not None:
                    return
            self._hold_once()
            next_tick += period_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def _hold_once(self) -> None:
        gripper = self._gripper
        if gripper is None:
            return
        state = gripper.get_state(wait=False)
        if state.error_code != 1:
            raise GripperError(
                f"LiteGrip stopped responding while holding (err={state.error_code})"
            )
        sent = gripper.send_mit_frame(
            state.position_rad,
            kp=gripper.config.kp,
            kd=gripper.config.kd,
        )
        if not sent:
            raise GripperError("LiteGrip did not accept a hold frame")

    # -- validation --------------------------------------------------------

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise GripperError(f"LiteGrip controller stopped: {self._error}") from self._error

    def _validate_state(self, state: Any) -> None:
        age = getattr(state, "data_age_s", None)
        if age is None or age > self.options.feedback_timeout_s:
            raise GripperError(
                "LiteGrip feedback is stale "
                f"(age={age!r}, limit={self.options.feedback_timeout_s * 1000.0:.0f} ms)"
            )
        if state.error_code != 1:
            raise GripperError(
                f"LiteGrip is not enabled (status {state.error_code}); "
                "0 means disabled and other values are faults"
            )
        temperatures = (state.temperature_mos, state.temperature_coil)
        if not all(math.isfinite(float(value)) for value in temperatures):
            raise GripperError("LiteGrip temperature feedback contains NaN or Inf")
        hottest = max(float(value) for value in temperatures)
        if hottest > self.options.max_temperature_c:
            raise GripperError(
                f"LiteGrip temperature {hottest:.1f} C exceeds "
                f"{self.options.max_temperature_c:.1f} C"
            )
        position = float(state.position_mm)
        if not math.isfinite(position):
            raise GripperError("LiteGrip position feedback is not finite")
        tolerance = self.options.position_tolerance_mm
        if position < -tolerance or position > self._travel_mm + tolerance:
            raise GripperError(
                f"LiteGrip opening {position:.1f} mm is outside the calibrated "
                f"travel 0-{self._travel_mm:.1f} mm by more than "
                f"{tolerance:.1f} mm"
            )

    @staticmethod
    def _travel(gripper: Any) -> float:
        config = gripper.config
        travel = abs(float(config.pos_open_rad) - float(config.pos_closed_rad))
        travel *= float(config.rad_to_mm)
        if not math.isfinite(travel) or travel <= 0:
            raise GripperError("LiteGrip calibration reports no usable travel")
        return travel

    def _resolve_open_mm(self, gripper: Any) -> float:
        if self.options.open_position_mm is not None:
            target = float(self.options.open_position_mm)
            if target > self._travel_mm:
                raise GripperError(
                    f"gripper.open_position_mm={target:.1f} exceeds the "
                    f"calibrated travel {self._travel_mm:.1f} mm"
                )
            return target
        work_stroke = float(gripper.config.work_stroke_mm)
        if 0.0 < work_stroke < self._travel_mm:
            return work_stroke
        return self._travel_mm

    @staticmethod
    def _close_partial(gripper: Any) -> None:
        try:
            gripper.stop()
        except BaseException:  # noqa: BLE001 - still close the link
            log.exception("Could not stop LiteGrip during connect cleanup")
        try:
            gripper.disconnect()
        except BaseException:  # noqa: BLE001 - preserve the connect failure
            log.exception("Could not close LiteGrip during connect cleanup")
