"""Offline LeRobot dataset and 30 Hz pairing tests."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from litearm_vla import record as recording
from litearm_vla.kinesthetic import (
    KinestheticRobot,
    KinestheticRobotConfig,
    KinestheticSample,
)


def _robot_config(tmp_path):
    return KinestheticRobotConfig(
        calibration_dir=tmp_path / "calibration",
        auto_enable=True,
        monitor_period_s=10.0,
        exit_wait_s=0.0,
    )


def test_cli_help_builds_without_opening_hardware():
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "litearm_vla.record", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "litearm_kinesthetic" in result.stdout


def test_loop_pairs_q_t_with_q_t_plus_one_and_drops_last(monkeypatch):
    names = [f"joint{i}.pos" for i in range(1, 8)]
    samples = []
    for offset in (0.0, 1.0, 2.0):
        values = {name: index + offset for index, name in enumerate(names)}
        samples.append(KinestheticSample(values, dict(values)))

    class SequenceRobot:
        def capture_sample(self):
            return samples.pop(0)

    class Dataset:
        fps = 30

    paired = []
    monkeypatch.setattr(
        recording,
        "_add_paired_frame",
        lambda dataset, observation, action, task: paired.append(
            (observation, action, task)
        ),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(recording.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(
        recording,
        "precise_sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    frames = recording.record_kinesthetic_loop(
        SequenceRobot(),
        {"exit_early": False},
        dataset=Dataset(),
        control_time_s=0.08,
        task="hand guide",
    )

    assert frames == 2
    assert list(paired[0][0].values()) == pytest.approx(range(7))
    assert list(paired[0][1].values()) == pytest.approx(
        [index + 1 for index in range(7)]
    )
    assert list(paired[1][1].values()) == pytest.approx(
        [index + 2 for index in range(7)]
    )
    assert not samples


def test_record_config_requires_30_hz_and_camera_for_video(tmp_path):
    robot_cfg = _robot_config(tmp_path)
    dataset_cfg = recording.DatasetRecordConfig(
        repo_id="tests/no-camera",
        single_task="hand guide",
        root=tmp_path / "unused",
        fps=30,
        video=True,
        push_to_hub=False,
    )
    with pytest.raises(ValueError, match="camera"):
        recording.KinestheticRecordConfig(robot=robot_cfg, dataset=dataset_cfg)

    dataset_cfg.video = False
    dataset_cfg.fps = 20
    with pytest.raises(ValueError, match="30"):
        recording.KinestheticRecordConfig(robot=robot_cfg, dataset=dataset_cfg)


def test_real_lerobot_dataset_reopens_with_future_joint_actions(
    monkeypatch, tmp_path
):
    import datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    robot = KinestheticRobot(_robot_config(tmp_path))
    features = recording.dataset_features(robot, use_video=False)
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "tests/litearm-kinesthetic",
        fps=30,
        root=root,
        robot_type=robot.name,
        features=features,
        use_videos=False,
    )
    names = list(robot.action_features)
    q_values = [[index * 0.1 + offset for index in range(7)] for offset in (0.0, 0.1, 0.2)]
    observations = [dict(zip(names, values, strict=True)) for values in q_values]

    recording._add_paired_frame(dataset, observations[0], observations[1], "hand guide")
    recording._add_paired_frame(dataset, observations[1], observations[2], "hand guide")
    dataset.save_episode()
    dataset.finalize()

    reopened = LeRobotDataset("tests/litearm-kinesthetic", root=root)
    assert len(reopened) == 2
    assert np.asarray(reopened[0]["observation.state"]) == pytest.approx(q_values[0])
    assert np.asarray(reopened[0]["action"]) == pytest.approx(q_values[1])
    assert np.asarray(reopened[1]["action"]) == pytest.approx(q_values[2])

    dataset_cfg = recording.DatasetRecordConfig(
        repo_id="tests/litearm-kinesthetic",
        single_task="hand guide",
        root=root,
        fps=30,
        video=False,
        push_to_hub=False,
    )
    resumed = recording._open_dataset(
        recording.KinestheticRecordConfig(
            robot=_robot_config(tmp_path),
            dataset=dataset_cfg,
            resume=True,
        ),
        robot,
    )
    recording._add_paired_frame(resumed, observations[0], observations[1], "hand guide")
    resumed.save_episode()
    resumed.finalize()

    reopened = LeRobotDataset("tests/litearm-kinesthetic", root=root)
    assert reopened.num_episodes == 2
    assert len(reopened) == 3


def test_recording_connects_before_dataset_and_always_disconnects(
    monkeypatch, tmp_path
):
    calls = []

    class Robot:
        name = "litearm_kinesthetic"
        cameras = {}

        def __init__(self, config):
            self.config = config

        def connect(self):
            calls.append("connect")

        def capture_sample(self):
            values = {f"joint{i}.pos": float(i) for i in range(1, 8)}
            return KinestheticSample(values, dict(values))

        def assert_safe(self):
            calls.append("safe")

        def disconnect(self):
            calls.append("disconnect")

    class Dataset:
        fps = 30
        num_episodes = 0

        def __init__(self):
            self.frames = []

        def clear_episode_buffer(self):
            self.frames.clear()

        def save_episode(self):
            calls.append("save")
            self.num_episodes += 1

        def finalize(self):
            calls.append("finalize")

    class Videos:
        def __init__(self, dataset):
            self.dataset = dataset

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    dataset = Dataset()
    monkeypatch.setattr(recording, "KinestheticRobot", Robot)
    monkeypatch.setattr(
        recording,
        "_open_dataset",
        lambda cfg, robot: calls.append("open_dataset") or dataset,
    )
    monkeypatch.setattr(recording, "VideoEncodingManager", Videos)
    monkeypatch.setattr(
        recording,
        "init_keyboard_listener",
        lambda: (
            None,
            {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            },
        ),
    )
    monkeypatch.setattr(recording, "is_headless", lambda: True)
    monkeypatch.setattr(recording, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        recording,
        "_add_paired_frame",
        lambda ds, observation, action, task: ds.frames.append((observation, action)),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(recording.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(
        recording,
        "precise_sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    dataset_cfg = recording.DatasetRecordConfig(
        repo_id="tests/session",
        single_task="hand guide",
        root=tmp_path / "unused",
        fps=30,
        episode_time_s=0.08,
        reset_time_s=0.01,
        num_episodes=1,
        video=False,
        push_to_hub=False,
    )
    cfg = recording.KinestheticRecordConfig(
        robot=_robot_config(tmp_path), dataset=dataset_cfg, play_sounds=False
    )

    assert recording.record_kinesthetic(cfg) is dataset
    assert calls.index("connect") < calls.index("open_dataset")
    assert calls.index("save") < calls.index("finalize") < calls.index("disconnect")
