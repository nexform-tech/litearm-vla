"""Offline lifecycle tests for the policy-driven combined robot adapter."""
from __future__ import annotations

import time

import numpy as np
import pytest
from lerobot.cameras.opencv import OpenCVCameraConfig

import litearm_vla.policy_robot as policy_module
from litearm_vla import LiteArmPolicyRobot, LiteArmPolicyRobotConfig, LiteGripOptions
from litearm_vla.policy_robot import ACTION_NAMES

from conftest import FakeArm, FakeGripper


class FakeCamera:
    def __init__(self):
        self.is_connected = False
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.calls = []

    def connect(self):
        self.calls.append("connect")
        self.is_connected = True

    def read_latest(self):
        self.calls.append("read_latest")
        return self.image

    def disconnect(self):
        self.calls.append("disconnect")
        self.is_connected = False


def _named(events, name):
    return [
        event
        for event in events
        if event == name or (isinstance(event, tuple) and event[0] == name)
    ]


def _wait(predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def make_policy_robot(fake_arm, fake_gripper, monkeypatch, tmp_path):
    def _make(**overrides):
        camera = FakeCamera()
        monkeypatch.setattr(
            policy_module,
            "make_cameras_from_configs",
            lambda configs: {next(iter(configs)): camera},
        )
        fields = dict(
            id="policy-test",
            calibration_dir=tmp_path / "lerobot-calibration",
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path=0, fps=30, width=640, height=480
                )
            },
            enable_on_connect=True,
            gripper=LiteGripOptions(
                enabled=True,
                calibration_file=str(tmp_path / "can0_calibration.json"),
                grasp_hold_s=0.01,
            ),
        )
        fields.update(overrides)
        robot = LiteArmPolicyRobot(LiteArmPolicyRobotConfig(**fields))
        return robot, camera

    return _make


def _connected(make_policy_robot, **overrides):
    robot, camera = make_policy_robot(**overrides)
    robot.connect()
    gripper = FakeGripper.instances[-1]
    arm = FakeArm.instances[-1]
    return robot, arm, gripper, camera


def _action(value=0.0, gripper=0.0):
    action = {name: float(value) for name in ACTION_NAMES}
    action["gripper.pos"] = float(gripper)
    return action


def test_features_match_the_eight_dimensional_training_contract(make_policy_robot):
    robot, _ = make_policy_robot()
    assert list(robot.action_features) == list(ACTION_NAMES)
    assert robot.action_features == {name: float for name in ACTION_NAMES}
    assert robot.observation_features == {
        **{name: float for name in ACTION_NAMES},
        "wrist": (480, 640, 3),
    }


def test_lerobot_044_aggregates_named_features_into_the_expected_tensors(
    make_policy_robot,
):
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    robot, _ = make_policy_robot()
    action = hw_to_dataset_features(robot.action_features, ACTION, use_video=True)
    observation = hw_to_dataset_features(
        robot.observation_features, OBS_STR, use_video=True
    )
    assert action["action"]["shape"] == (8,)
    assert action["action"]["names"] == list(ACTION_NAMES)
    assert observation["observation.state"]["shape"] == (8,)
    assert observation["observation.state"]["names"] == list(ACTION_NAMES)
    assert observation["observation.images.wrist"]["shape"] == (480, 640, 3)


def test_connect_enables_the_arm_the_gripper_and_the_camera(make_policy_robot):
    robot, arm, gripper, camera = _connected(make_policy_robot)
    try:
        assert robot.is_connected
        assert _named(arm.events, "enable")
        assert _wait(lambda: bool(_named(arm.events, "joint_follow")))
        assert gripper.connected is True
        assert gripper.enabled is True
        assert camera.is_connected is True
    finally:
        robot.disconnect()


def test_observation_reports_the_gripper_opening_in_radians(make_policy_robot):
    robot, arm, gripper, camera = _connected(make_policy_robot)
    try:
        for index, joint in enumerate(arm.state.joints):
            joint.q = 0.1 * index
        gripper.state.position_mm = 42.5
        observation = robot.get_observation()
        assert [observation[f"joint{i}.pos"] for i in range(1, 8)] == pytest.approx(
            [0.1 * index for index in range(7)]
        )
        expected_rad = 42.5 / gripper.config.rad_to_mm
        assert observation["gripper.pos"] == pytest.approx(expected_rad)
        assert observation["wrist"] is camera.image
        assert "observation.state" not in observation
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    ("opening", "command"),
    [
        (0.0, "grasp"),
        (1.0, "open"),
    ],
)
def test_action_maps_the_opening_to_grasp_or_open(
    make_policy_robot, opening, command
):
    robot, _, gripper, _ = _connected(make_policy_robot)
    try:
        accepted = robot.send_action(_action(gripper=opening))
        assert accepted["gripper.pos"] == pytest.approx(opening)
        assert _wait(lambda: bool(_named(list(gripper.events), command)))
    finally:
        robot.disconnect()


def test_action_clamps_the_opening_to_calibrated_travel(make_policy_robot):
    robot, _, gripper, _ = _connected(make_policy_robot)
    try:
        accepted = robot.send_action(_action(gripper=999.0))
        travel = abs(gripper.config.pos_open_rad - gripper.config.pos_closed_rad)
        assert accepted["gripper.pos"] == pytest.approx(travel)
    finally:
        robot.disconnect()


def test_action_requires_all_named_dimensions_and_finite_values(make_policy_robot):
    robot, _, _, _ = _connected(make_policy_robot)
    try:
        missing = _action()
        missing.pop("joint7.pos")
        with pytest.raises(ValueError, match="joint7.pos"):
            robot.send_action(missing)
        invalid = _action()
        invalid["gripper.pos"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            robot.send_action(invalid)
    finally:
        robot.disconnect()


@pytest.mark.parametrize("failure", [RuntimeError("policy failed"), KeyboardInterrupt()])
def test_exception_or_ctrl_c_holds_the_arm_before_peripheral_cleanup(
    make_policy_robot, failure
):
    robot, arm, gripper, camera = _connected(make_policy_robot)
    order = []

    original_movej = arm.movej
    original_gripper_stop = gripper.stop
    original_camera_disconnect = camera.disconnect

    def tracked_movej(q, speed=1.0):
        order.append("arm_hold")
        return original_movej(q, speed=speed)

    def tracked_gripper_stop():
        order.append("gripper_stop")
        return original_gripper_stop()

    def tracked_camera_disconnect():
        order.append("camera_disconnect")
        return original_camera_disconnect()

    arm.movej = tracked_movej
    gripper.stop = tracked_gripper_stop
    camera.disconnect = tracked_camera_disconnect

    with pytest.raises(type(failure), match=str(failure) or None):
        with robot:
            raise failure

    assert order[0] == "arm_hold"
    assert order.index("arm_hold") < order.index("gripper_stop")
    assert order.index("arm_hold") < order.index("camera_disconnect")
    assert not _named(arm.events, "disable")


def test_config_requires_motion_flags_to_agree(tmp_path):
    config = LiteArmPolicyRobotConfig(
        id="policy-test",
        calibration_dir=tmp_path / "cal",
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480
            )
        },
        enable_on_connect=True,
        gripper=LiteGripOptions(
            enabled=False,
            calibration_file=str(tmp_path / "can0_calibration.json"),
        ),
    )
    with pytest.raises(ValueError, match="enable the arm and gripper together"):
        config.validate()


def test_config_requires_exactly_one_calibration_source(tmp_path):
    config = LiteArmPolicyRobotConfig(
        id="policy-test",
        calibration_dir=tmp_path / "cal",
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480
            )
        },
        enable_on_connect=True,
        gripper=LiteGripOptions(enabled=True),
    )
    with pytest.raises(ValueError, match="exactly one gripper calibration source"):
        config.validate()


def test_config_requires_exactly_one_camera():
    config = LiteArmPolicyRobotConfig(
        id="policy-test",
        enable_on_connect=True,
        gripper=LiteGripOptions(enabled=True),
    )
    with pytest.raises(ValueError, match="exactly one camera"):
        config.validate()
