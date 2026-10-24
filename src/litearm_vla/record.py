"""Record 30 Hz hand-guided LiteArm episodes with LeRobot 0.4.4."""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401 - register CLI camera choices
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.config import RobotConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from .gripper import LiteGripOptions
from .kinesthetic import RECORD_FPS, KinestheticRobot, KinestheticRobotConfig

log = logging.getLogger(__name__)


def init_gripper_keyboard_listener(
    options: LiteGripOptions, events: dict[str, bool]
) -> Any | None:
    """Latch open/close keys into ``events`` for the recording loop.

    A separate listener keeps LeRobot's arrow and escape bindings untouched.
    The loop applies the latch on the main thread, so the worker never reads
    the keyboard directly.
    """
    if is_headless():
        log.warning(
            "Headless environment: gripper keys %r/%r are unavailable",
            options.open_key,
            options.close_key,
        )
        return None

    from pynput import keyboard

    open_key = options.open_key.lower()
    close_key = options.close_key.lower()

    def on_press(key: Any) -> None:
        char = getattr(key, "char", None)
        if not isinstance(char, str):
            return
        char = char.lower()
        if char == open_key:
            log.info("Opening the gripper")
            events["gripper_open"] = True
        elif char == close_key:
            log.info("Closing the gripper")
            events["gripper_close"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


class SamplingRateError(RuntimeError):
    """The captured episode can no longer be represented honestly at 30 Hz."""


@dataclass
class KinestheticRecordConfig:
    """LeRobot recording options plus the direct kinesthetic Robot."""

    robot: RobotConfig
    dataset: DatasetRecordConfig
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.robot, KinestheticRobotConfig):
            raise TypeError("robot.type must be litearm_kinesthetic")
        self.robot.validate()
        if float(self.dataset.fps) != float(RECORD_FPS):
            raise ValueError(f"kinesthetic recording is fixed at {RECORD_FPS} Hz")
        if self.dataset.video and not self.robot.cameras:
            raise ValueError("dataset.video=true requires at least one configured camera")


def _add_paired_frame(
    dataset: Any,
    previous_observation: dict[str, Any],
    future_action: dict[str, Any],
    task: str,
) -> None:
    observation_frame = build_dataset_frame(
        dataset.features, previous_observation, prefix=OBS_STR
    )
    action_frame = build_dataset_frame(dataset.features, future_action, prefix=ACTION)
    dataset.add_frame({**observation_frame, **action_frame, "task": task})


def record_kinesthetic_loop(
    robot: KinestheticRobot,
    events: dict[str, bool],
    *,
    control_time_s: float,
    task: str,
    dataset: Any | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    fps: int = RECORD_FPS,
) -> int:
    """Capture one phase and pair ``observation[t]`` with ``q[t+1]``."""
    if float(fps) != float(RECORD_FPS):
        raise ValueError(f"kinesthetic recording only supports {RECORD_FPS} Hz")
    if dataset is not None and float(dataset.fps) != float(RECORD_FPS):
        raise ValueError(f"dataset fps must be {RECORD_FPS}, got {dataset.fps}")
    if not math.isfinite(float(control_time_s)) or control_time_s <= 0:
        raise ValueError("control_time_s must be finite and greater than zero")

    period_s = 1.0 / RECORD_FPS
    start_t = time.perf_counter()
    next_tick = start_t
    previous_observation: dict[str, Any] | None = None
    previous_sample_t: float | None = None
    consecutive_slow = 0
    frames = 0

    while time.perf_counter() - start_t < control_time_s:
        if events.get("exit_early", False):
            events["exit_early"] = False
            break

        sample_t = time.perf_counter()
        if dataset is not None and previous_sample_t is not None:
            interval = sample_t - previous_sample_t
            if interval > 2.0 * period_s:
                raise SamplingRateError(
                    f"sample gap {interval * 1000.0:.1f} ms exceeds "
                    f"{2.0 * period_s * 1000.0:.1f} ms"
                )
            consecutive_slow = consecutive_slow + 1 if interval > 1.25 * period_s else 0
            if consecutive_slow >= 3:
                raise SamplingRateError("three consecutive samples missed the 30 Hz target by 25%")
        previous_sample_t = sample_t

        sample = robot.capture_sample()
        if previous_observation is not None and dataset is not None:
            _add_paired_frame(
                dataset,
                previous_observation,
                sample.future_action_candidate,
                task,
            )
            frames += 1
        if display_data:
            log_rerun_data(
                observation=sample.observation,
                action=sample.future_action_candidate,
                compress_images=display_compressed_images,
            )
        previous_observation = sample.observation

        # Apply gripper keys after sampling so the command recorded with the
        # interval that just ended is the one that was in effect during it.
        apply_keys = getattr(robot, "apply_keyboard_events", None)
        if apply_keys is not None:
            apply_keys(events)

        next_tick += period_s
        precise_sleep(max(0.0, next_tick - time.perf_counter()))

    return frames


def dataset_features(robot: KinestheticRobot, use_video: bool) -> dict[str, dict]:
    """Build the native LeRobot state, action, and image feature schema."""
    return {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_video),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=use_video),
    }


def _open_dataset(
    cfg: KinestheticRecordConfig, robot: KinestheticRobot
) -> LeRobotDataset:
    features = dataset_features(robot, cfg.dataset.video)
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        if robot.cameras:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras)
                ),
            )
        sanity_check_dataset_robot_compatibility(
            dataset, robot, RECORD_FPS, features
        )
        return dataset

    sanity_check_dataset_name(cfg.dataset.repo_id, None)
    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        RECORD_FPS,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=(
            cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras)
        ),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )


@safe_stop_image_writer
def record_kinesthetic(cfg: KinestheticRecordConfig) -> LeRobotDataset:
    """Run a multi-episode direct LiteArm collection session."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(
            session_name="litearm-kinesthetic-recording",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )
    display_compressed = (
        True
        if cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None
        else cfg.display_compressed_images
    )

    robot = KinestheticRobot(cfg.robot)
    if getattr(robot, "gripper", None) is not None and is_headless():
        raise RuntimeError(
            "gripper keyboard control needs a graphical session; "
            "run with a display or disable --robot.gripper.enabled"
        )
    dataset: LeRobotDataset | None = None
    listener = None
    gripper_listener = None
    try:
        # Connection includes feedback checks, optional explicit enable,
        # zero-gravity confirmation, safety monitoring, and cameras. No dataset
        # directory is created until those gates pass.
        robot.connect()
        dataset = _open_dataset(cfg, robot)

        with VideoEncodingManager(dataset):
            listener, events = init_keyboard_listener()
            gripper_listener = (
                init_gripper_keyboard_listener(cfg.robot.gripper, events)
                if getattr(robot, "gripper", None) is not None
                else None
            )
            recorded_episodes = 0
            while (
                recorded_episodes < cfg.dataset.num_episodes
                and not events["stop_recording"]
            ):
                log_say(
                    f"Recording episode {dataset.num_episodes}", cfg.play_sounds
                )
                try:
                    frames = record_kinesthetic_loop(
                        robot,
                        events,
                        dataset=dataset,
                        control_time_s=cfg.dataset.episode_time_s,
                        task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed,
                    )
                except SamplingRateError as exc:
                    log.error("Discarding slow episode: %s", exc)
                    dataset.clear_episode_buffer()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    continue

                if frames == 0:
                    dataset.clear_episode_buffer()
                    if events["stop_recording"]:
                        break
                    log.warning("Episode has fewer than two samples; recording it again")
                    continue

                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1
                    or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_kinesthetic_loop(
                        robot,
                        events,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                robot.assert_safe()
                dataset.save_episode()
                robot.assert_safe()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        try:
            if dataset is not None:
                dataset.finalize()
                robot.assert_safe()
        finally:
            try:
                robot.disconnect()
            finally:
                for key_listener in (listener, gripper_listener):
                    if not is_headless() and key_listener is not None:
                        key_listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    log_say("Exiting", cfg.play_sounds)
    return dataset


@parser.wrap()
def record_from_cli(cfg: KinestheticRecordConfig) -> LeRobotDataset:
    return record_kinesthetic(cfg)


def main() -> None:
    record_from_cli()


if __name__ == "__main__":
    main()
