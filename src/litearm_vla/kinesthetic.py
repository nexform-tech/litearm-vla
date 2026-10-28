"""Direct LiteArm kinesthetic Robot for LeRobot data collection.

The adapter reuses ``LiteArmRobot`` for CDC setup and firmware-derived limits,
then adds the ``litearm-python`` zero-gravity lifecycle and LeRobot cameras.
It never sends policy actions; hand-guided actions are derived one sample later
by :mod:`litearm_vla.record`.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from litearm_lerobot import (
    GRIPPER_FEATURE,
    LiteArmRobot,
    LiteArmRobotConfig,
    LiteGripController,
    LiteGripOptions,
)

RECORD_FPS = 30
DEFAULT_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

#: Firmware mode number for a latched emergency stop.
MODE_EMERGENCY = 6
#: Settle time after ``clear_faults()`` before reading state again.
CLEAR_SETTLE_S = 0.2
#: Settle time after ``reset()`` before reading state again.
RESET_SETTLE_S = 1.5

#: Firmware fault-bit names (mirrors ``litearm._protocol.FLAG_NAMES``). Kept
#: local so the public config can validate without importing a private module.
KNOWN_FLAG_NAMES = (
    "FAULT",
    "WD_TRIPPED",
    "FB_STALE",
    "TEMP_WARN",
    "POS_VIOL",
    "OVERSPEED",
)

log = logging.getLogger(__name__)


def _normalize_flag_names(value: object) -> tuple[str, ...]:
    """Parse a flag list from a sequence or a CLI string.

    The canonical CLI form is JSON, e.g.
    ``--robot.ignored_flags='["WD_TRIPPED"]'``. draccus decodes a bare string
    for a sequence field into one character per element, so a plain
    ``--robot.ignored_flags=WD_TRIPPED`` is rejoined here as a convenience.
    """
    if isinstance(value, str):
        parts = value.strip().strip("[]").split(",")
    else:
        parts = [str(part) for part in value]
    if len(parts) > 1 and all(len(part) == 1 for part in parts):
        parts = ["".join(parts)]
    return tuple(
        part.strip().strip("\"'").upper() for part in parts if part.strip()
    )


class SafetyError(RuntimeError):
    """The direct arm session no longer satisfies its safety contract."""


@RobotConfig.register_subclass("litearm_kinesthetic")
@dataclass(kw_only=True)
class KinestheticRobotConfig(LiteArmRobotConfig):
    """Configuration for direct, hand-guided LiteArm collection."""

    # The base driver must remain observation-only. Starting its servo loop and
    # zero-gravity at the same time would create two control modes.
    enable_on_connect: bool = field(default=False, init=False, repr=False)
    disable_on_disconnect: bool = field(default=False, init=False, repr=False)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    #: Optional keyboard-driven LiteGrip. Disabled by default so the arm-only
    #: workflow never opens a CAN interface.
    gripper: LiteGripOptions = field(default_factory=LiteGripOptions)
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
    #: Tool payload mass in kg for the firmware's gravity feed-forward. ``None``
    #: leaves the firmware value alone. The write is RAM-only: run
    #: ``Arm.save_params()`` yourself if the value should survive a power cycle.
    payload_kg: float | None = None
    #: Payload centre of mass in metres, one per axis. ``set_payload()`` always
    #: writes this, so the default clears any previously stored offset.
    payload_com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    auto_enable: bool = False
    zero_g_period_s: float = 0.04
    feedback_timeout_s: float = 0.25
    monitor_period_s: float = 0.02
    max_temperature_c: float = 70.0
    exit_wait_s: float = 0.10
    hold_speed: float = 0.30

    # Firmware flags to treat as recoverable warnings instead of faults.
    # ``WD_TRIPPED`` is latched when the firmware's 0.10 s command watchdog
    # expires between keepalives; the zero-gravity keepalive re-sends ``ZERO_G``
    # every ``zero_g_period_s``, so the trip self-clears. The host-side
    # keepalive health (``zero_g_error`` / ``zero_g_active``) is still enforced,
    # and every other flag stays fatal.
    ignored_flags: tuple[str, ...] = ()

    # Clear a fault latched by a previous fail-closed session (``EMERGENCY`` /
    # ``FAULT``) before giving up. A latched arm cannot enable or enter
    # zero-gravity, and the firmware refuses to clear while energized, so this
    # only ever runs on a disabled arm and the safety check is re-run afterwards.
    # Off by default because a latched fault is a safety signal.
    auto_clear_faults: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.joint_names = tuple(self.joint_names)
        self.ignored_flags = _normalize_flag_names(self.ignored_flags)
        self.payload_com = tuple(float(value) for value in self.payload_com)

    def validate(self) -> None:
        super().validate()
        self.gripper.validate()
        if self.num_joints != 7 or len(self.joint_names) != self.num_joints:
            raise ValueError("kinesthetic collection requires seven named joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if not 0.005 <= self.zero_g_period_s < 0.10:
            raise ValueError("zero_g_period_s must be in [0.005, 0.10)")
        for name in (
            "feedback_timeout_s",
            "monitor_period_s",
            "max_temperature_c",
        ):
            if not math.isfinite(float(getattr(self, name))) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if not math.isfinite(self.exit_wait_s) or self.exit_wait_s < 0:
            raise ValueError("exit_wait_s must be finite and non-negative")
        if not 0 < self.hold_speed <= 1:
            raise ValueError("hold_speed must be in (0, 1]")
        unknown = sorted(set(self.ignored_flags) - set(KNOWN_FLAG_NAMES))
        if unknown:
            raise ValueError(
                f"unknown ignored_flags {unknown}; expected names from {KNOWN_FLAG_NAMES}"
            )
        if self.payload_kg is not None:
            if (
                not math.isfinite(self.payload_kg)
                or not 0.0 <= self.payload_kg <= 20.0
            ):
                raise ValueError("payload_kg must be finite and in [0, 20] kg")
        if len(self.payload_com) != 3 or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.payload_com
        ):
            raise ValueError(
                "payload_com must contain three finite values in [-1, 1] m"
            )
        for name, camera in self.cameras.items():
            if int(camera.fps) < RECORD_FPS:
                raise ValueError(
                    f"camera {name!r} is {camera.fps} Hz; recording requires {RECORD_FPS} Hz"
                )


@dataclass(frozen=True)
class KinestheticSample:
    """One measured observation and its candidate action for the prior frame."""

    observation: RobotObservation
    future_action_candidate: RobotAction


class _SafetyMonitor:
    """Continuously checks the arm while the main thread saves or encodes data."""

    def __init__(self, robot: "KinestheticRobot") -> None:
        self._robot = robot
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="litearm-vla-safety", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._robot.config.monitor_period_s):
            try:
                self._robot._read_safe_state(refresh=False, require_zero_g=True)
                self._robot._assert_gripper_safe()
            except BaseException as exc:  # noqa: BLE001 - preserve the safety cause
                self.error = exc
                self._robot._fail_into_hold(exc)
                return

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise SafetyError(f"safety monitor stopped the session: {self.error}") from self.error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self._robot.config.monitor_period_s))
            self._thread = None


class KinestheticRobot(LiteArmRobot):
    """A single-owner, zero-gravity LiteArm with optional LeRobot cameras."""

    name = "litearm_kinesthetic"
    config_class = KinestheticRobotConfig

    def __init__(self, config: KinestheticRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._gripper = (
            LiteGripController(config.gripper) if config.gripper.enabled else None
        )
        self._monitor: _SafetyMonitor | None = None
        self._trip_lock = threading.Lock()
        self._fail_closed = False
        self._logged_ignored_flags: set[str] = set()

    @property
    def _joint_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def gripper(self) -> LiteGripController | None:
        """The optional keyboard-driven gripper controller."""
        return self._gripper

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        camera_features = {
            name: (
                int(self.config.cameras[name].height),
                int(self.config.cameras[name].width),
                3,
            )
            for name in self.cameras
        }
        gripper_features = (
            {GRIPPER_FEATURE: float} if self._gripper is not None else {}
        )
        return {**self._joint_features, **gripper_features, **camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        gripper_features = (
            {GRIPPER_FEATURE: float} if self._gripper is not None else {}
        )
        return {**self._joint_features, **gripper_features}

    @property
    def is_connected(self) -> bool:
        return (
            self._arm is not None
            and all(camera.is_connected for camera in self.cameras.values())
            and (self._gripper is None or self._gripper.is_connected)
        )

    def _read_safe_state(
        self,
        *,
        refresh: bool,
        require_zero_g: bool,
        require_enabled: bool = True,
    ) -> Any:
        arm = self._arm
        if arm is None:
            raise SafetyError("LiteArm CDC session is not connected")
        if require_zero_g:
            if arm.zero_g_error is not None:
                raise SafetyError(f"zero-gravity keepalive failed: {arm.zero_g_error}")
            if not arm.zero_g_active:
                raise SafetyError("zero-gravity keepalive is not active")

        msg = arm.get_state(refresh=refresh, timeout=self.config.feedback_timeout_s)
        state = msg.value
        if state is None:
            raise SafetyError("no LiteArm feedback has been received")
        age = time.monotonic() - float(msg.timestamp)
        if msg.timestamp <= 0 or age > self.config.feedback_timeout_s:
            raise SafetyError(
                f"LiteArm feedback is stale ({age * 1000.0:.1f} ms; "
                f"limit {self.config.feedback_timeout_s * 1000.0:.1f} ms)"
            )
        if state.n != self.config.num_joints:
            raise SafetyError(
                f"expected {self.config.num_joints} joints, received {state.n}"
            )
        if require_enabled and not state.enabled:
            raise SafetyError("LiteArm is not enabled")
        ignored = set(self.config.ignored_flags)
        active_flags = [name for name in state.flag_names if name not in ignored]
        for name in sorted(ignored.intersection(state.flag_names)):
            if name not in self._logged_ignored_flags:
                self._logged_ignored_flags.add(name)
                log.warning(
                    "Ignoring LiteArm flag %s (listed in --robot.ignored_flags)", name
                )
        if state.faulted or active_flags:
            raise SafetyError(
                f"LiteArm fault: flags={active_flags}, joint_fault={state.joint_fault}"
            )

        q = tuple(float(value) for value in state.q)
        if len(q) != self.config.num_joints or not all(math.isfinite(value) for value in q):
            raise SafetyError("joint feedback must contain seven finite positions")
        if self._limits is None:
            raise SafetyError("firmware-derived joint limits are unavailable")
        outside = [
            index + 1
            for index, (value, lower, upper) in enumerate(
                zip(q, self._limits.lo, self._limits.hi, strict=True)
            )
            if value < lower or value > upper
        ]
        if outside:
            raise SafetyError(f"joints outside the configured limit margin: {outside}")

        temperatures = [
            float(value)
            for joint in state.joints
            for value in (joint.t_mos, joint.t_coil)
        ]
        if not all(math.isfinite(value) for value in temperatures):
            raise SafetyError("joint temperature feedback contains NaN or Inf")
        hottest = max(temperatures)
        if hottest > self.config.max_temperature_c:
            raise SafetyError(
                f"joint temperature {hottest:.1f} C exceeds "
                f"{self.config.max_temperature_c:.1f} C"
            )
        return state

    def _apply_payload(self, mass: float) -> None:
        """Write the tool payload and verify the firmware kept the value.

        The firmware silently clamps ``payload_mass`` to [0, 20] and still
        ACKs, so the readback is the only evidence that the write took effect.
        """
        arm = self._arm
        if arm is None:
            raise SafetyError("cannot set the payload without a connected arm")
        arm.set_payload(float(mass), tuple(self.config.payload_com))
        try:
            readback = arm.get_ff_scalar(4, 0).value
        except BaseException as exc:  # noqa: BLE001 - fail closed on no readback
            raise SafetyError(f"cannot verify the LiteArm payload: {exc}") from exc
        if (
            readback is None
            or not math.isfinite(float(readback))
            or abs(float(readback) - mass) > 1e-3
        ):
            raise SafetyError(
                f"LiteArm payload readback {readback!r} kg does not match "
                f"the requested {mass:.3f} kg"
            )
        log.info(
            "LiteArm payload set to %.3f kg at com=%s m (readback %.3f kg)",
            mass,
            self.config.payload_com,
            float(readback),
        )

    def _assert_gripper_safe(self) -> None:
        if self._gripper is None:
            return
        try:
            self._gripper.assert_safe()
        except BaseException as exc:  # noqa: BLE001 - preserve the gripper cause
            raise SafetyError(f"LiteGrip: {exc}") from exc

    def connect(self, calibrate: bool = True) -> None:
        if self._arm is not None:
            return
        self.config.validate()
        try:
            # The base adapter opens CDC, checks the joint count, and reads the
            # firmware limits. Its servo loop remains disabled by configuration.
            super().connect(calibrate=calibrate)
            # Validate the disabled pose before applying energy. After enable(),
            # enter zero-gravity immediately: its first command and keepalive
            # own the firmware's 0.10 s command-watchdog deadline. Waiting for
            # another status frame here can trip WD_TRIPPED before zero-gravity
            # has a chance to start.
            try:
                state = self._read_safe_state(
                    refresh=True,
                    require_zero_g=False,
                    require_enabled=False,
                )
            except SafetyError:
                if not self.config.auto_clear_faults:
                    raise
                self._clear_latched_fault()
                state = self._read_safe_state(
                    refresh=True,
                    require_zero_g=False,
                    require_enabled=False,
                )
            if not state.enabled and not self.config.auto_enable:
                raise SafetyError(
                    "LiteArm is disabled; pass --robot.auto_enable=true only after "
                    "the low-risk hardware checks"
                )
            # Write the payload while the arm is still disabled when possible:
            # item 4/5 are feed-forward parameters, not motion commands.
            if self.config.payload_kg is not None:
                self._apply_payload(self.config.payload_kg)
            if not state.enabled:
                self._arm.enable()
            self._arm.zero_g_start(period=self.config.zero_g_period_s)
            self._read_safe_state(refresh=True, require_zero_g=True)

            # Start the gripper before the monitor so its first check always has
            # a session to check, and before any camera so a CAN failure does
            # not leave video devices open.
            if self._gripper is not None:
                self._gripper.connect()
            self._monitor = _SafetyMonitor(self)
            self._monitor.start()
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            if self._monitor is not None:
                self._monitor.stop()
                self._monitor = None
            if self._gripper is not None:
                try:
                    self._gripper.disconnect()
                except BaseException:  # noqa: BLE001 - the connect cause wins
                    log.exception("Could not close LiteGrip after connect failure")
            for camera in reversed(tuple(self.cameras.values())):
                if camera.is_connected:
                    try:
                        camera.disconnect()
                    except BaseException:  # noqa: BLE001
                        log.exception("Could not close camera after connect failure")
            self._cleanup_arm_after_failure()
            raise

    def _clear_latched_fault(self) -> None:
        """Clear a fault latched by a previous session (``auto_clear_faults``).

        A fail-closed session leaves the firmware latched in ``EMERGENCY`` /
        ``FAULT``. In that state ``enable()`` is rejected and zero-gravity can
        never start, so the recorder cannot simply ignore the flag. The firmware
        refuses to clear while energized: this only runs on a disabled arm, logs
        what it cleared, and lets the caller re-run the full safety check.
        """
        arm = self._arm
        if arm is None:
            return
        state = arm.get_state(
            refresh=True, timeout=self.config.feedback_timeout_s
        ).value
        if state is None:
            raise SafetyError("cannot clear a latched fault without fresh feedback")
        if not (state.flag_names or state.faulted):
            # The connect failure was something else (stale feedback, pose,
            # temperature); clearing faults would not help and hides the cause.
            return
        if state.enabled:
            raise SafetyError(
                "refusing to clear a latched fault while the arm is enabled; "
                "disable the arm first"
            )
        log.warning(
            "auto_clear_faults: clearing latched fault flags=%s joint_fault=%s mode=%s",
            state.flag_names,
            state.joint_fault,
            getattr(state, "mode_name", state.mode),
        )
        arm.clear_faults()
        time.sleep(CLEAR_SETTLE_S)
        state = arm.get_state(
            refresh=True, timeout=self.config.feedback_timeout_s
        ).value
        if state is not None and state.mode == MODE_EMERGENCY:
            log.warning(
                "auto_clear_faults: still in EMERGENCY after clear_faults; resetting"
            )
            arm.reset()
            time.sleep(RESET_SETTLE_S)

    def _hold_current_pose(self, arm: Any) -> None:
        """Hand the arm off to a zero-displacement hold, leaving it enabled.

        This mirrors the normal shutdown hand-off: the firmware holds the
        measured pose with full stiffness instead of coasting. A ``movej`` the
        firmware rejects (for example while an axis is faulted) is logged and
        never masks the original safety error.
        """
        try:
            state = arm.get_state(
                refresh=True, timeout=self.config.feedback_timeout_s
            ).value
            if state is not None and state.enabled:
                arm.movej(list(state.q), speed=self.config.hold_speed)
        except BaseException:  # noqa: BLE001 - the original error is reported
            log.exception("Could not hand off to a zero-displacement hold")

    def _fail_into_hold(self, cause: BaseException) -> None:
        """Stop zero-gravity after a safety error and hold the measured pose.

        The arm is deliberately **not** emergency-stopped and **not** disabled:
        the physical emergency stop is the emergency path, and the session must
        leave the arm enabled in a controlled hold instead of latching the
        firmware in ``EMERGENCY``.
        """
        with self._trip_lock:
            if self._fail_closed or self._arm is None:
                return
            self._fail_closed = True
            log.error("Failing into hold after LiteArm safety error: %s", cause)
            try:
                self._arm.zero_g_stop(raise_on_lost=False)
            except BaseException:  # noqa: BLE001 - still attempt the hold
                log.exception("Could not stop zero-gravity before the hold hand-off")
            if self._gripper is not None:
                try:
                    self._gripper.fail_safe(cause)
                except BaseException:  # noqa: BLE001 - the arm hold still runs
                    log.exception("Could not stop the LiteGrip after a safety error")
            self._hold_current_pose(self._arm)

    def _cleanup_arm_after_failure(self) -> None:
        arm, self._arm = self._arm, None
        self._limits = None
        if arm is None:
            return
        try:
            arm.zero_g_stop(raise_on_lost=False)
            self._hold_current_pose(arm)
        except BaseException:  # noqa: BLE001
            log.exception("Failed-connect cleanup could not confirm a safe arm state")
        finally:
            arm.close()

    def assert_safe(self) -> None:
        if self._monitor is None:
            raise SafetyError("safety monitor is not running")
        self._monitor.raise_if_failed()
        try:
            self._read_safe_state(refresh=False, require_zero_g=True)
            self._assert_gripper_safe()
        except BaseException as exc:  # noqa: BLE001
            self._fail_into_hold(exc)
            raise

    def capture_sample(self) -> KinestheticSample:
        if self._monitor is None:
            raise SafetyError("KinestheticRobot is not connected")
        self._monitor.raise_if_failed()
        try:
            state = self._read_safe_state(refresh=True, require_zero_g=True)
        except BaseException as exc:  # noqa: BLE001
            self._fail_into_hold(exc)
            raise
        gripper_sample = None
        if self._gripper is not None:
            try:
                gripper_sample = self._gripper.capture_state()
            except BaseException as exc:  # noqa: BLE001
                self._fail_into_hold(exc)
                raise SafetyError(f"LiteGrip: {exc}") from exc
        values = tuple(float(value) for value in state.q)
        joint_values = dict(zip(self._joint_features, values, strict=True))
        observation: RobotObservation = dict(joint_values)
        candidate: RobotAction = dict(joint_values)
        if gripper_sample is not None:
            # The gripper follows the same q[t+1] rule as the arm and is
            # recorded as a calibrated opening angle in radians, so every
            # dimension of the state/action vector shares the same unit.
            observation[GRIPPER_FEATURE] = gripper_sample.position_rad
            candidate[GRIPPER_FEATURE] = gripper_sample.position_rad
        for name, camera in self.cameras.items():
            observation[name] = camera.read_latest()
        return KinestheticSample(
            observation=observation,
            future_action_candidate=candidate,
        )

    def apply_keyboard_events(self, events: dict[str, bool]) -> None:
        """Apply latched gripper keys; a missing gripper makes this a no-op."""
        if self._gripper is None:
            return
        try:
            if events.pop("gripper_open", False):
                self._gripper.request_open()
            if events.pop("gripper_close", False):
                self._gripper.request_close()
        except BaseException as exc:  # noqa: BLE001 - same fail-closed path
            self._fail_into_hold(exc)
            raise SafetyError(f"LiteGrip: {exc}") from exc

    def get_observation(self) -> RobotObservation:
        return self.capture_sample().observation

    def send_action(self, action: RobotAction) -> RobotAction:
        del action
        raise SafetyError("KinestheticRobot is read-only; actions come from q[t+1]")

    def disconnect(self) -> None:
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            monitor.stop()
        monitor_error = monitor.error if monitor is not None else None

        cleanup_error: BaseException | None = monitor_error
        gripper, self._gripper = self._gripper, None
        if gripper is not None:
            try:
                gripper.disconnect()
            except BaseException as exc:  # noqa: BLE001 - still close the arm
                cleanup_error = cleanup_error or exc

        arm, self._arm = self._arm, None
        if arm is not None:
            try:
                arm.zero_g_stop(raise_on_lost=True)
                if monitor_error is None:
                    if self.config.exit_wait_s:
                        time.sleep(self.config.exit_wait_s)
                    # Temporarily restore the handle so the shared validator can
                    # check freshness, faults, temperature, and limits.
                    self._arm = arm
                    state = self._read_safe_state(refresh=True, require_zero_g=False)
                    self._arm = None
                    arm.movej(list(state.q), speed=self.config.hold_speed)
            except BaseException as exc:  # noqa: BLE001
                cleanup_error = cleanup_error or exc
                self._arm = arm
                self._fail_into_hold(exc)
                self._arm = None
            finally:
                arm.close()

        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except BaseException as exc:  # noqa: BLE001
                    cleanup_error = cleanup_error or exc
        self._limits = None
        self._fail_closed = False
        if cleanup_error is not None:
            raise SafetyError(f"unsafe LiteArm shutdown: {cleanup_error}") from cleanup_error
