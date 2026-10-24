"""Offline lifecycle and safety tests for the keyboard-driven LiteGrip."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from litearm_vla.gripper import GripperError, LiteGripOptions
from litearm_vla.kinesthetic import (
    KinestheticRobot,
    KinestheticRobotConfig,
    SafetyError,
)


def _event_names(events):
    return [event[0] if isinstance(event, tuple) else event for event in events]


def _config(tmp_path, **overrides):
    gripper = overrides.pop(
        "gripper", LiteGripOptions(enabled=True, hold_hz=50.0)
    )
    values = {
        "calibration_dir": tmp_path / "calibration",
        "auto_enable": True,
        "monitor_period_s": 10.0,
        "exit_wait_s": 0.0,
        "gripper": gripper,
    }
    values.update(overrides)
    return KinestheticRobotConfig(**values)


def _wait_for(events, name, timeout=1.0):
    deadline = time.monotonic() + timeout
    while name not in _event_names(events) and time.monotonic() < deadline:
        time.sleep(0.005)
    return name in _event_names(events)


def test_gripper_disabled_keeps_the_arm_only_contract(fake_arm, tmp_path):
    robot = KinestheticRobot(
        _config(tmp_path, gripper=LiteGripOptions(enabled=False))
    )
    robot.connect()

    assert list(robot.observation_features) == [f"joint{i}.pos" for i in range(1, 8)]
    assert "gripper.pos" not in robot.action_features
    assert robot.gripper is None

    robot.disconnect()


def test_gripper_enabled_adds_features_and_command_action(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    assert list(robot.observation_features) == [
        *(f"joint{i}.pos" for i in range(1, 8)),
        "gripper.pos",
    ]
    assert list(robot.action_features) == list(robot.observation_features)

    sample = robot.capture_sample()
    assert sample.observation["gripper.pos"] == pytest.approx(20.0)
    # Before the first key press the command is a hold at the measured opening.
    assert sample.future_action_candidate["gripper.pos"] == pytest.approx(20.0)

    robot.disconnect()
    names = _event_names(gripper.events)
    assert names.index("connect") < names.index("enable")
    assert names[-1] == "disconnect"
    assert "stop" in names


def test_idle_session_streams_hold_frames_before_the_first_key(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    # No key has been pressed. The motor answers control frames rather than
    # reporting on its own, so the worker must hold from connect: otherwise the
    # feedback goes stale and the safety monitor trips the session.
    assert _wait_for(gripper.events, "mit")
    robot.assert_safe()

    robot.disconnect()


def test_keys_request_open_and_close(fake_arm, fake_gripper, tmp_path):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    events = {"gripper_open": True}
    robot.apply_keyboard_events(events)
    assert "gripper_open" not in events
    assert _wait_for(gripper.events, "open")
    assert robot.capture_sample().future_action_candidate["gripper.pos"] == pytest.approx(
        80.0
    )

    events = {"gripper_close": True}
    robot.apply_keyboard_events(events)
    assert "gripper_close" not in events
    assert _wait_for(gripper.events, "grasp")
    assert robot.capture_sample().future_action_candidate["gripper.pos"] == pytest.approx(
        0.0
    )

    robot.disconnect()


def test_repeated_key_does_not_restart_an_applied_move(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    robot.apply_keyboard_events({"gripper_open": True})
    assert _wait_for(gripper.events, "open")
    for _ in range(5):
        robot.apply_keyboard_events({"gripper_open": True})
        time.sleep(0.01)

    opens = [
        event
        for event in gripper.events
        if isinstance(event, tuple) and event[0] == "open"
    ]
    assert len(opens) == 1

    robot.disconnect()


def test_custom_open_position_uses_a_bounded_move(fake_arm, fake_gripper, tmp_path):
    robot = KinestheticRobot(
        _config(
            tmp_path,
            gripper=LiteGripOptions(
                enabled=True, hold_hz=50.0, open_position_mm=40.0
            ),
        )
    )
    robot.connect()
    gripper = fake_gripper.instances[-1]

    robot.apply_keyboard_events({"gripper_open": True})
    assert _wait_for(gripper.events, "move_at_speed")
    assert any(
        isinstance(event, tuple) and event[0] == "move_at_speed" and event[1] == 40.0
        for event in gripper.events
    )

    robot.disconnect()


def test_grasp_retries_a_slice_that_ends_in_free_travel(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    calls = 0

    def grasp(force_n=None, hold_s=0.0, *, progress=None):
        nonlocal calls
        calls += 1
        gripper.events.append(("grasp", force_n, hold_s))
        if calls == 1:
            # The SDK released the force while the jaws were still moving.
            return SimpleNamespace(
                ok=False, reached=False, stalled=False, state=gripper.state
            )
        gripper.state.position_mm = 0.0
        return SimpleNamespace(
            ok=True, reached=False, stalled=True, state=gripper.state
        )

    gripper.grasp = grasp
    robot.apply_keyboard_events({"gripper_close": True})

    deadline = time.monotonic() + 1.0
    while calls < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert calls >= 2, "a free-travel slice did not start another close ramp"

    sample = robot.capture_sample()
    assert sample.future_action_candidate["gripper.pos"] == pytest.approx(0.0)
    robot.assert_safe()
    robot.disconnect()


def test_persistent_grasp_failure_holds_position_and_keeps_the_session(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    calls = 0
    failing = [True]

    def grasp(force_n=None, hold_s=0.0, *, progress=None):
        nonlocal calls
        calls += 1
        gripper.events.append(("grasp", force_n, hold_s))
        if failing[0]:
            gripper.state.position_mm = 42.0
            return SimpleNamespace(
                ok=False, reached=False, stalled=False, state=gripper.state
            )
        gripper.state.position_mm = 0.0
        return SimpleNamespace(
            ok=True, reached=False, stalled=True, state=gripper.state
        )

    gripper.grasp = grasp
    robot.apply_keyboard_events({"gripper_close": True})

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        sample = robot.capture_sample()
        if sample.future_action_candidate["gripper.pos"] == pytest.approx(42.0):
            break
        time.sleep(0.005)
    else:
        pytest.fail("a failed grasp did not fall back to a position hold")

    # The slice was retried before giving up, then the worker stayed alive.
    assert calls == 3
    assert sample.observation["gripper.pos"] == pytest.approx(42.0)
    robot.assert_safe()

    # A new close press must restart the grasp, not be swallowed as an
    # auto-repeat of the command that failed.
    failing[0] = False
    robot.apply_keyboard_events({"gripper_close": True})
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        sample = robot.capture_sample()
        if calls >= 4 and sample.future_action_candidate["gripper.pos"] == pytest.approx(
            0.0
        ):
            break
        time.sleep(0.005)
    else:
        pytest.fail("the failed grasp command was not cleared for a retry")

    robot.disconnect()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda gripper: setattr(gripper.state, "error_code", 0x9), "not enabled"),
        (lambda gripper: setattr(gripper.state, "data_age_s", 1.0), "stale"),
        (lambda gripper: setattr(gripper.state, "temperature_coil", 80), "temperature"),
        (lambda gripper: setattr(gripper.state, "position_mm", 100.0), "outside"),
    ],
)
def test_capture_fails_closed_on_unsafe_gripper_feedback(
    fake_arm, fake_gripper, tmp_path, mutate, match
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]
    mutate(gripper)

    with pytest.raises(SafetyError, match=match):
        robot.capture_sample()

    assert "zero_g_stop" in _event_names(fake_arm.instances[-1].events)
    assert "stop" in _event_names(gripper.events)
    with pytest.raises(SafetyError, match="unsafe LiteArm shutdown"):
        robot.disconnect()


def test_grasp_excursion_inside_the_sdk_hold_guard_is_accepted(
    fake_arm, fake_gripper, tmp_path
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    gripper = fake_gripper.instances[-1]

    # A 20 N power grasp presses past the closed calibration limit. The SDK
    # allows hold_runaway_rad (0.05 rad ~= 3.1 mm) there, so the host check
    # must not trip on an excursion the motor is still holding correctly.
    gripper.state.position_mm = -2.5
    sample = robot.capture_sample()
    assert sample.observation["gripper.pos"] == pytest.approx(-2.5)

    # Beyond the configured margin is still a fault.
    gripper.state.position_mm = -5.0
    with pytest.raises(SafetyError, match="outside"):
        robot.capture_sample()
    with pytest.raises(SafetyError, match="unsafe LiteArm shutdown"):
        robot.disconnect()


def test_gripper_connect_failure_releases_the_arm(
    fake_arm, fake_gripper, monkeypatch, tmp_path
):
    original = fake_gripper.load_calibration

    def fail_calibration(self, path=None):
        original(self, path=path)
        return False

    monkeypatch.setattr(fake_gripper, "load_calibration", fail_calibration)
    robot = KinestheticRobot(_config(tmp_path))

    with pytest.raises(GripperError, match="calibration"):
        robot.connect()

    arm = fake_arm.instances[-1]
    names = _event_names(arm.events)
    assert "zero_g_stop" in names
    assert "movej" in names
    assert "emergency_stop" not in names
    assert arm.events[-1] == "close"
    assert fake_gripper.instances[-1].disconnected


def test_enable_failure_names_the_can_link(
    fake_arm, fake_gripper, monkeypatch, tmp_path
):
    def fail_enable(self, retries=None):
        return SimpleNamespace(
            ok=False, tries=3, state=SimpleNamespace(error_code=0)
        )

    monkeypatch.setattr(fake_gripper, "enable", fail_enable)
    robot = KinestheticRobot(_config(tmp_path))

    with pytest.raises(GripperError, match="can0 is UP"):
        robot.connect()

    assert fake_gripper.instances[-1].disconnected
    assert fake_arm.instances[-1].events[-1] == "close"


@pytest.mark.parametrize(
    "options",
    [
        LiteGripOptions(enabled=True, speed_mm_s=0.0),
        LiteGripOptions(enabled=True, grasp_force_n=50.0),
        LiteGripOptions(enabled=True, open_key="x", close_key="x"),
        LiteGripOptions(enabled=True, mount="sideways"),
        LiteGripOptions(enabled=True, open_position_mm=0.0),
        LiteGripOptions(enabled=True, position_tolerance_mm=0.0),
    ],
)
def test_invalid_gripper_options_are_rejected(options):
    with pytest.raises(ValueError):
        options.validate()
