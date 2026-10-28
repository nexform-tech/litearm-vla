"""Policy-driven LeRobot adapter for a LiteArm, a LiteGrip, and one camera.

The arm follows position targets through ``LiteArmRobot``'s background servo
loop. The gripper is owned by :class:`~litearm_vla.gripper.LiteGripController`
and is commanded in its two discrete modes: the policy output is compared with
the midpoint of the calibrated travel and becomes an open or grasp request.
The observation always reports the measured opening in radians (0 at the closed
stop), so the state vector matches the unit the recorder writes and the dataset
stores.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from litearm_lerobot import LiteArmRobot, LiteArmRobotConfig

from .gripper import (
    GRIPPER_FEATURE,
    GripperError,
    LiteGripController,
    LiteGripOptions,
)

log = logging.getLogger(__name__)

JOINT_ACTION_NAMES = tuple(f"joint{i}.pos" for i in range(1, 8))
GRIPPER_ACTION_NAME = GRIPPER_FEATURE
ACTION_NAMES = (*JOINT_ACTION_NAMES, GRIPPER_ACTION_NAME)


@RobotConfig.register_subclass("litearm_policy")
@dataclass(kw_only=True)
class LiteArmPolicyRobotConfig(LiteArmRobotConfig):
    """Configure a seven-joint LiteArm, one LiteGrip, and one camera.

    This adapter exists to run a trained policy; it has no hand-guided mode.
    ``enable_on_connect`` and ``gripper.enabled`` must agree, and a connected
    session requires both to be true.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    #: Override the base driver's motion-on-connect default. Policy deployment
    #: must opt in explicitly for both actuators.
    enable_on_connect: bool = False
    gripper: LiteGripOptions = field(default_factory=LiteGripOptions)

    def validate(self) -> None:
        super().validate()
        self.gripper.validate()
        if self.num_joints != 7:
            raise ValueError(
                "litearm_policy has a fixed seven-joint contract; "
                f"received num_joints={self.num_joints}"
            )
        if self.enable_on_connect != self.gripper.enabled:
            raise ValueError(
                "policy motion must enable the arm and gripper together; "
                "set enable_on_connect and gripper.enabled to the same value"
            )
        if len(self.cameras) != 1:
            raise ValueError(
                "litearm_policy requires exactly one camera; "
                f"received {len(self.cameras)}"
            )
        camera_name = next(iter(self.cameras))
        if not camera_name or "." in camera_name:
            raise ValueError(
                "camera name must be a non-empty LeRobot feature segment "
                f"without '.', received {camera_name!r}"
            )
        if (self.gripper.calibration_file is None) == (self.gripper.mount is None):
            raise ValueError(
                "configure exactly one gripper calibration source: "
                "calibration_file or mount"
            )
        if not str(self.gripper.channel).strip():
            raise ValueError("gripper.channel must not be empty")


class LiteArmPolicyRobot(LiteArmRobot):
    """Expose the training-time 8-D state/action contract to LeRobot 0.4.4."""

    name = "litearm_policy"
    config_class = LiteArmPolicyRobotConfig

    def __init__(self, config: LiteArmPolicyRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._camera_name = next(iter(config.cameras))
        self._gripper = (
            LiteGripController(config.gripper) if config.gripper.enabled else None
        )
        self._fully_connected = False

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        camera = self.config.cameras[self._camera_name]
        state = {name: float for name in ACTION_NAMES}
        return {**state, self._camera_name: (camera.height, camera.width, 3)}

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in ACTION_NAMES}

    @property
    def is_connected(self) -> bool:
        return bool(
            self._fully_connected
            and self._arm is not None
            and self._gripper is not None
            and self._gripper.is_connected
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # The arm reports absolute joint positions. The gripper calibration is
        # enforced by LiteGripController.connect() before the session opens.
        return True

    @property
    def is_enabled(self) -> bool:
        return super().is_enabled and self._gripper is not None

    def connect(self, calibrate: bool = True) -> None:
        if self._fully_connected:
            return
        self.config.validate()
        if not (self.config.enable_on_connect and self.config.gripper.enabled):
            raise RuntimeError(
                "litearm_policy is a policy adapter; set enable_on_connect and "
                "gripper.enabled to true to drive the arm and gripper"
            )
        try:
            super().connect(calibrate=False)
            self._gripper.connect()
            if not self._gripper.is_connected:
                raise RuntimeError(
                    "LiteGrip connect returned without an active connection"
                )
            for camera in self.cameras.values():
                camera.connect()
            if not all(camera.is_connected for camera in self.cameras.values()):
                raise RuntimeError(
                    "camera connect returned without an active connection"
                )
            self._fully_connected = True
            if calibrate:
                self.calibrate()
        except BaseException:
            self.disconnect()
            raise

    def calibrate(self) -> None:
        """No-op: both arm and gripper require pre-existing calibration."""
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("LiteArmPolicyRobot is not connected")
        arm_observation = super().get_observation()["observation.state"]
        observation = {
            name: float(value)
            for name, value in zip(JOINT_ACTION_NAMES, arm_observation, strict=True)
        }
        try:
            observation[GRIPPER_ACTION_NAME] = (
                self._gripper.capture_state().position_rad
            )
        except GripperError as exc:
            raise RuntimeError(f"LiteGrip: {exc}") from exc
        if not all(math.isfinite(observation[name]) for name in ACTION_NAMES):
            raise RuntimeError(
                "robot state contains a non-finite arm or gripper position"
            )
        image = self.cameras[self._camera_name].read_latest()
        expected_shape = self.observation_features[self._camera_name]
        if tuple(getattr(image, "shape", ())) != expected_shape:
            raise RuntimeError(
                f"camera {self._camera_name!r} returned shape "
                f"{getattr(image, 'shape', None)}, expected {expected_shape}"
            )
        observation[self._camera_name] = image
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        if not self.is_connected:
            raise RuntimeError("LiteArmPolicyRobot is not connected")
        missing = [name for name in ACTION_NAMES if name not in action]
        if missing:
            raise ValueError(f"action is missing required keys: {missing}")
        values = [float(action[name]) for name in ACTION_NAMES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all LiteArmPolicyRobot actions must be finite")

        # Validate the entire request before either actuator sees a new target.
        arm_target = values[:7]
        gripper_target = values[7]
        try:
            travel_rad = self._gripper.travel_rad
        except GripperError as exc:
            raise RuntimeError(f"LiteGrip: {exc}") from exc
        accepted_arm = super().send_action({"action": arm_target})["action"]
        accepted_gripper = min(max(gripper_target, 0.0), travel_rad)
        if accepted_gripper != gripper_target:
            log.warning(
                "LiteGrip target %.4f rad exceeded calibrated travel and was "
                "clamped to %.4f rad",
                gripper_target,
                accepted_gripper,
            )
        try:
            self._gripper.request_opening_rad(accepted_gripper)
        except GripperError as exc:
            raise RuntimeError(f"LiteGrip: {exc}") from exc
        accepted = {
            name: float(value)
            for name, value in zip(JOINT_ACTION_NAMES, accepted_arm, strict=True)
        }
        accepted[GRIPPER_ACTION_NAME] = accepted_gripper
        return accepted

    def disconnect(self) -> None:
        """Stop policy control and leave the arm enabled at its measured pose.

        LeRobot calls this method when a ``with robot`` block exits, including
        for ordinary exceptions and ``KeyboardInterrupt``. Stop and hand off the
        arm before waiting for the gripper or camera so the last policy target
        cannot keep running during peripheral cleanup.

        This is a controlled software stop, not ``Arm.emergency_stop()``. The
        SDK emergency stop latches a fault and removes holding torque; calling
        it here would contradict the requirement to prevent gravity drop. The
        base adapter instead stops the host servo and sends a zero-displacement
        ``movej`` to the measured pose, leaving the firmware enabled in its
        rigid position-hold mode.
        """
        self._fully_connected = False
        try:
            # The arm is the gravity-loaded actuator. Make it safe first;
            # camera shutdown and the gripper join must not delay the
            # controlled hold hand-off after an exception or Ctrl+C.
            super().disconnect()
        finally:
            gripper = self._gripper
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:  # noqa: BLE001 - all other cleanup still runs
                    log.exception("Failed to disconnect LiteGrip")
            for camera in self.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:  # noqa: BLE001 - continue releasing other devices
                    log.exception("Failed to disconnect camera %s", self._camera_name)

    def __str__(self) -> str:
        return f"{self.id} LiteArmPolicyRobot"
