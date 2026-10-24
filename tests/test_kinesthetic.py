"""Offline safety and lifecycle tests for direct kinesthetic collection."""
from __future__ import annotations

import time

import pytest

from litearm_vla.kinesthetic import (
    KinestheticRobot,
    KinestheticRobotConfig,
    SafetyError,
)


def _event_names(arm):
    return [event[0] if isinstance(event, tuple) else event for event in arm.events]


def _config(tmp_path, **overrides):
    values = {
        "calibration_dir": tmp_path / "calibration",
        "auto_enable": True,
        "monitor_period_s": 10.0,
        "exit_wait_s": 0.0,
    }
    values.update(overrides)
    return KinestheticRobotConfig(**values)


def test_disabled_arm_requires_explicit_auto_enable(fake_arm, tmp_path):
    robot = KinestheticRobot(_config(tmp_path, auto_enable=False))

    with pytest.raises(SafetyError, match="auto_enable=true"):
        robot.connect()

    arm = fake_arm.instances[-1]
    assert "enable" not in arm.events
    assert arm.events[-1] == "close"


def test_connect_reuses_base_driver_and_starts_zero_gravity(fake_arm, tmp_path):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    arm = fake_arm.instances[-1]

    assert arm.events.index("params") < arm.events.index("enable")
    enable_index = arm.events.index("enable")
    zero_g_index = next(
        index for index, event in enumerate(arm.events) if isinstance(event, tuple) and event[0] == "zero_g_start"
    )
    assert enable_index < zero_g_index
    assert arm.events[enable_index + 1 : zero_g_index] == []
    assert arm.zero_g_active
    assert list(robot.observation_features) == [f"joint{i}.pos" for i in range(1, 8)]
    assert robot.action_features == robot.observation_features

    sample = robot.capture_sample()
    assert list(sample.observation.values()) == pytest.approx(
        [index * 0.1 for index in range(7)]
    )
    assert sample.future_action_candidate == sample.observation
    with pytest.raises(SafetyError, match="read-only"):
        robot.send_action(sample.future_action_candidate)

    robot.disconnect()
    assert any(isinstance(event, tuple) and event[0] == "movej" for event in arm.events)
    assert arm.events[-1] == "close"


def test_connect_does_not_poll_state_between_enable_and_zero_gravity(
    fake_arm, monkeypatch, tmp_path
):
    original_get_state = fake_arm.get_state

    def trip_watchdog_on_idle_read(self, refresh=False, timeout=0.5):
        if "enable" in self.events and not any(
            isinstance(event, tuple) and event[0] == "zero_g_start"
            for event in self.events
        ):
            self.state.flags = 1 << 1
            self.state.flag_names = ["WD_TRIPPED"]
        return original_get_state(self, refresh=refresh, timeout=timeout)

    monkeypatch.setattr(fake_arm, "get_state", trip_watchdog_on_idle_read)
    robot = KinestheticRobot(_config(tmp_path))

    robot.connect()
    arm = fake_arm.instances[-1]
    assert arm.state.flag_names == []

    robot.disconnect()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda arm: setattr(arm.state, "joint_fault", 1), "fault"),
        (lambda arm: setattr(arm.state.joints[2], "t_mos", 80.0), "temperature"),
        (lambda arm: setattr(arm.state.joints[1], "q", 2.1), "limit margin"),
        (lambda arm: setattr(arm, "timestamp_offset", -1.0), "stale"),
    ],
)
def test_capture_fails_closed_on_unsafe_feedback(
    fake_arm, tmp_path, mutate, match
):
    robot = KinestheticRobot(_config(tmp_path))
    robot.connect()
    arm = fake_arm.instances[-1]
    mutate(arm)

    with pytest.raises(SafetyError, match=match):
        robot.capture_sample()

    names = _event_names(arm)
    assert "zero_g_stop" in names
    assert "emergency_stop" not in names
    with pytest.raises(SafetyError, match="unsafe LiteArm shutdown"):
        robot.disconnect()
    assert arm.events[-1] == "close"


def _latch_fault_on_connect(monkeypatch, fake_arm, *, enabled=False):
    original_connect = fake_arm.connect

    def connect_with_latched_fault(self):
        self.state.mode = 6
        self.state.enabled = enabled
        self.state.flags = 1
        self.state.flag_names = ["FAULT"]
        return original_connect(self)

    monkeypatch.setattr(fake_arm, "connect", connect_with_latched_fault)


def test_auto_clear_faults_recovers_a_latched_emergency(fake_arm, monkeypatch, tmp_path):
    _latch_fault_on_connect(monkeypatch, fake_arm)
    robot = KinestheticRobot(_config(tmp_path, auto_clear_faults=True))

    robot.connect()
    arm = fake_arm.instances[-1]

    zero_g_index = next(
        index
        for index, event in enumerate(arm.events)
        if isinstance(event, tuple) and event[0] == "zero_g_start"
    )
    assert arm.events.index("clear_faults") < arm.events.index("reset") < zero_g_index
    assert arm.zero_g_active

    robot.disconnect()
    assert arm.events[-1] == "close"


def test_latched_fault_still_fails_closed_by_default(fake_arm, monkeypatch, tmp_path):
    _latch_fault_on_connect(monkeypatch, fake_arm)
    robot = KinestheticRobot(_config(tmp_path))

    with pytest.raises(SafetyError, match="FAULT"):
        robot.connect()

    arm = fake_arm.instances[-1]
    assert "clear_faults" not in arm.events
    assert arm.events[-1] == "close"


def test_auto_clear_faults_leaves_non_fault_failures_alone(fake_arm, monkeypatch, tmp_path):
    original_connect = fake_arm.connect

    def connect_out_of_range(self):
        self.state.joints[1].q = 2.1
        return original_connect(self)

    monkeypatch.setattr(fake_arm, "connect", connect_out_of_range)
    robot = KinestheticRobot(_config(tmp_path, auto_clear_faults=True))

    with pytest.raises(SafetyError, match="limit margin"):
        robot.connect()

    arm = fake_arm.instances[-1]
    assert "clear_faults" not in arm.events
    assert arm.events[-1] == "close"


def test_auto_clear_faults_refuses_while_enabled(fake_arm, monkeypatch, tmp_path):
    _latch_fault_on_connect(monkeypatch, fake_arm, enabled=True)
    robot = KinestheticRobot(_config(tmp_path, auto_clear_faults=True))

    with pytest.raises(SafetyError, match="enabled"):
        robot.connect()

    arm = fake_arm.instances[-1]
    assert "clear_faults" not in arm.events
    assert arm.events[-1] == "close"


def test_ignored_watchdog_flag_does_not_block_capture(fake_arm, tmp_path):
    robot = KinestheticRobot(_config(tmp_path, ignored_flags=("wd_tripped",)))
    robot.connect()
    arm = fake_arm.instances[-1]

    arm.state.flags = 1 << 1
    arm.state.flag_names = ["WD_TRIPPED"]

    sample = robot.capture_sample()
    assert sample.future_action_candidate == sample.observation

    robot.disconnect()
    assert "emergency_stop" not in _event_names(arm)
    assert arm.events[-1] == "close"


@pytest.mark.parametrize("flag", ["FAULT", "FB_STALE", "POS_VIOL"])
def test_ignored_watchdog_keeps_other_flags_fatal(fake_arm, tmp_path, flag):
    robot = KinestheticRobot(_config(tmp_path, ignored_flags=("WD_TRIPPED",)))
    robot.connect()
    arm = fake_arm.instances[-1]

    arm.state.flags = 1 << 1
    arm.state.flag_names = ["WD_TRIPPED", flag]

    with pytest.raises(SafetyError, match=flag):
        robot.capture_sample()

    names = _event_names(arm)
    assert "zero_g_stop" in names
    assert "emergency_stop" not in names


def test_ignored_flags_are_normalized_and_validated(tmp_path):
    config = _config(tmp_path, ignored_flags=("wd_tripped",))
    assert config.ignored_flags == ("WD_TRIPPED",)
    config.validate()

    bad = _config(tmp_path, ignored_flags=("NOT_A_FLAG",))
    with pytest.raises(ValueError, match="unknown ignored_flags"):
        bad.validate()


def test_connect_failure_after_enable_hands_off_to_hold(
    fake_arm, monkeypatch, tmp_path
):
    original = fake_arm.zero_g_start

    def fail_zero_g(self, period=0.04):
        original(self, period)
        raise RuntimeError("zero-g rejected")

    monkeypatch.setattr(fake_arm, "zero_g_start", fail_zero_g)
    robot = KinestheticRobot(_config(tmp_path))

    with pytest.raises(RuntimeError, match="zero-g rejected"):
        robot.connect()

    arm = fake_arm.instances[-1]
    names = _event_names(arm)
    assert "zero_g_stop" in names
    assert "movej" in names
    assert "emergency_stop" not in names
    assert arm.events[-1] == "close"


def test_background_monitor_trips_while_main_thread_is_idle(fake_arm, tmp_path):
    robot = KinestheticRobot(_config(tmp_path, monitor_period_s=0.01))
    robot.connect()
    arm = fake_arm.instances[-1]
    arm.state.joints[0].t_coil = 80.0

    deadline = time.monotonic() + 1.0
    while "zero_g_stop" not in _event_names(arm) and time.monotonic() < deadline:
        time.sleep(0.01)

    names = _event_names(arm)
    assert "zero_g_stop" in names
    assert "emergency_stop" not in names
    with pytest.raises(SafetyError, match="safety monitor"):
        robot.assert_safe()
    with pytest.raises(SafetyError, match="unsafe LiteArm shutdown"):
        robot.disconnect()
