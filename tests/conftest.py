"""Offline fixtures for the direct LiteArm collection path."""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@dataclass
class FakeJoint:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    t_mos: float = 30.0
    t_coil: float = 31.0
    err: int = 0


class FakeState:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.mode = 1
        self.flags = 0
        self.flag_names: list[str] = []
        self.joint_fault = 0
        self.joints = [FakeJoint(q=index * 0.1) for index in range(7)]

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def q(self) -> list[float]:
        return [joint.q for joint in self.joints]

    @property
    def faulted(self) -> bool:
        return bool(self.flags & 1) or self.mode == 6 or self.joint_fault != 0


class FakeArm:
    """The subset of ``litearm.Arm`` used by both adapters."""

    instances: list["FakeArm"] = []

    def __init__(self, port=None, move_timeout=15.0) -> None:
        self.port = port
        self.move_timeout = move_timeout
        self.n = 7
        self.state = FakeState()
        self.zero_g_active = False
        self.zero_g_error = None
        self.events: list[object] = []
        self.timestamp_offset = 0.0
        self.payload_mass = 0.0
        self.payload_com = [0.0, 0.0, 0.0]
        params = [
            SimpleNamespace(q_min=-2.0, q_max=2.0) for _ in range(self.n)
        ]
        self.params = SimpleNamespace(all_joint_params=self._all_joint_params)
        self._params = params
        type(self).instances.append(self)

    def _all_joint_params(self):
        self.events.append("params")
        return self._params

    def connect(self):
        self.events.append("connect")
        return self

    def get_state(self, refresh=False, timeout=0.5):
        self.events.append(("state", refresh))
        return SimpleNamespace(
            value=self.state,
            hz=100.0,
            timestamp=time.monotonic() + self.timestamp_offset,
        )

    def enable(self):
        self.events.append("enable")
        self.state.enabled = True

    def clear_faults(self):
        self.events.append("clear_faults")
        self.state.flags = 0
        self.state.flag_names = []
        self.state.joint_fault = 0

    def reset(self):
        self.events.append("reset")
        self.state.mode = 1
        self.state.flags = 0
        self.state.flag_names = []
        self.state.joint_fault = 0

    def zero_g_start(self, period=0.04):
        self.events.append(("zero_g_start", period))
        self.zero_g_active = True

    def zero_g_stop(self, raise_on_lost=False):
        self.events.append(("zero_g_stop", raise_on_lost))
        self.zero_g_active = False
        if raise_on_lost and self.zero_g_error is not None:
            raise self.zero_g_error

    def emergency_stop(self):
        self.events.append("emergency_stop")
        self.state.enabled = False

    def set_payload(self, mass, com=(0.0, 0.0, 0.0)):
        self.events.append(("set_payload", float(mass), tuple(com)))
        self.payload_mass = float(mass)
        self.payload_com = [float(value) for value in com]

    def get_ff_scalar(self, item, sub=0, timeout=1.0):
        self.events.append(("get_ff_scalar", item, sub))
        if item == 4:
            value = self.payload_mass
        elif item == 5:
            value = self.payload_com[sub]
        else:
            value = 0.0
        return SimpleNamespace(value=value, timestamp=time.monotonic())

    def movej(self, q, speed=1.0):
        self.events.append(("movej", list(q), speed))
        return self.state

    def joint_follow(self, q, dq, kp, kd):
        self.events.append(
            (
                "joint_follow",
                [float(value) for value in q],
                [float(value) for value in dq],
                [float(value) for value in kp],
                [float(value) for value in kd],
            )
        )
        for joint, value in zip(self.state.joints, q):
            joint.q = float(value)

    def move_js(self, q, dq=None, tau_ff=None):
        self.events.append(
            (
                "move_js",
                [float(value) for value in q],
                None if dq is None else [float(value) for value in dq],
            )
        )
        for joint, value in zip(self.state.joints, q):
            joint.q = float(value)

    def close(self):
        self.events.append("close")


@pytest.fixture
def fake_arm(monkeypatch):
    from litearm_lerobot import robot as base_robot

    FakeArm.instances.clear()
    monkeypatch.setattr(base_robot.litearm, "Arm", FakeArm)
    return FakeArm


class FakeGripperConfig:
    """The calibrated limits and gains the controller reads from ``litegrip``."""

    calibrated = True
    # 85 mm of travel at the shipped 61.01 mm/rad scale.
    pos_closed_rad = 1.3934
    pos_open_rad = 0.0
    rad_to_mm = 61.01
    work_stroke_mm = 80.0
    max_stroke_mm = 85.0
    close_sign = 1.0
    kp = 100.0
    kd = 2.0


class FakeGripperState:
    def __init__(self) -> None:
        self.position_mm = 20.0
        self.torque_nm = 0.0
        self.temperature_mos = 30
        self.temperature_coil = 31
        self.error_code = 1
        self.data_age_s = 0.0
        self.fresh = True

    @property
    def position_rad(self) -> float:
        return (
            FakeGripperConfig.pos_closed_rad
            - self.position_mm / FakeGripperConfig.rad_to_mm
        )


class FakeGripper:
    """The subset of ``litegrip.LiteGrip`` used by the controller."""

    instances: list["FakeGripper"] = []

    def __init__(
        self,
        channel="can0",
        can_id=0x08,
        mst_id=None,
        canfd_mode=None,
        motor_type=None,
        config=None,
        motion_config=None,
        disable_on_disconnect=True,
        mount=None,
    ) -> None:
        self.channel = channel
        self.can_id = can_id
        self.mount = mount
        self.config = FakeGripperConfig()
        self.state = FakeGripperState()
        self.events: list[object] = []
        self.connected = False
        self.enabled = False
        self.disconnected = False
        type(self).instances.append(self)

    def connect(self):
        self.events.append("connect")
        self.connected = True
        return True

    def load_calibration(self, path=None):
        self.events.append(("load_calibration", path))
        return True

    def enable(self, retries=None):
        self.events.append("enable")
        self.enabled = True
        return SimpleNamespace(ok=True, tries=1, state=self.state)

    def refresh_status(self):
        self.events.append("refresh_status")
        return True

    def get_state(self, wait=True):
        self.events.append(("get_state", wait))
        return self.state

    def open(self, speed_mm_s=None, *, progress=None):
        self.events.append(("open", speed_mm_s))
        self.state.position_mm = self.config.work_stroke_mm
        return SimpleNamespace(ok=True, reached=True, stalled=False)

    def close(self, speed_mm_s=None, *, progress=None):
        self.events.append(("close", speed_mm_s))
        self.state.position_mm = 0.0
        return SimpleNamespace(ok=True, reached=True, stalled=False)

    def move_at_speed(self, target_mm, speed_mm_s=30.0, kp=None, kd=None):
        self.events.append(("move_at_speed", target_mm, speed_mm_s))
        self.state.position_mm = float(target_mm)
        return SimpleNamespace(ok=True, reached=True, stalled=False)

    def grasp(self, force_n=None, hold_s=0.0, *, progress=None):
        self.events.append(("grasp", force_n, hold_s))
        self.state.position_mm = 0.0
        time.sleep(0.002)
        return SimpleNamespace(
            ok=True, reached=True, stalled=False, cycles=1, travel_guard=False
        )

    def send_mit_frame(self, q, kp, kd, dq=0.0, tau=0.0):
        self.events.append(("mit", q, kp, kd))
        return True

    def stop(self):
        self.events.append("stop")

    def disable(self):
        self.events.append("disable")
        self.enabled = False

    def disconnect(self):
        self.events.append("disconnect")
        self.connected = False
        self.disconnected = True


@pytest.fixture
def fake_gripper(monkeypatch):
    from litearm_vla import gripper as gripper_module

    FakeGripper.instances.clear()
    monkeypatch.setattr(gripper_module.litegrip, "LiteGrip", FakeGripper)
    return FakeGripper

