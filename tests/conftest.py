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

    def movej(self, q, speed=1.0):
        self.events.append(("movej", list(q), speed))
        return self.state

    def close(self):
        self.events.append("close")


@pytest.fixture
def fake_arm(monkeypatch):
    from litearm_lerobot import robot as base_robot

    FakeArm.instances.clear()
    monkeypatch.setattr(base_robot.litearm, "Arm", FakeArm)
    return FakeArm

