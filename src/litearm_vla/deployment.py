"""Fail-closed checkpoint validation and policy runner for the combined robot."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from litearm_lerobot import (
    ACTION_NAMES,
    GRIPPER_ACTION_NAME,
    JOINT_ACTION_NAMES,
    LiteGripOptions,
)

log = logging.getLogger(__name__)

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

FULL_MODEL_FILE = "model.safetensors"
PEFT_MODEL_FILES = ("adapter_config.json", "adapter_model.safetensors")

# These are policy-output sanity limits, not actuator speed limits. They sit
# after LeRobot's postprocessor has restored physical units and reject a target
# that is too far from the observation used to produce it. The actuator
# adapters retain ownership of physical limits and motion shaping.
DEFAULT_MAX_ARM_ACTION_DELTA_RAD = 0.10
# The gripper column shares the arm joints' unit. 0.05 rad is about 3.1 mm at
# the shipped 61.01 mm/rad calibration scale.
DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD = 0.05


class UnsafePolicyActionError(RuntimeError):
    """A postprocessed policy action failed the deployment sanity gate."""


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, received {value}")
    return number


def _finite_feature(mapping: dict[str, Any], name: str, source: str) -> float:
    if name not in mapping:
        raise UnsafePolicyActionError(f"{source} is missing required feature {name!r}")
    try:
        value = float(mapping[name])
    except (TypeError, ValueError) as exc:
        raise UnsafePolicyActionError(
            f"{source} feature {name!r} is not a scalar number: {mapping[name]!r}"
        ) from exc
    if not math.isfinite(value):
        raise UnsafePolicyActionError(
            f"{source} feature {name!r} is non-finite: {value}"
        )
    return value


def guard_postprocessed_action(
    requested: dict[str, Any],
    observation: dict[str, Any],
    *,
    max_arm_delta_rad: float = DEFAULT_MAX_ARM_ACTION_DELTA_RAD,
    max_gripper_delta_rad: float = DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD,
) -> dict[str, float]:
    """Reject a finite but implausible position jump before actuator dispatch.

    ``predict_action`` has already run the checkpoint postprocessor when this
    function is called, so both mappings use physical units: radians for the
    seven arm joints and radians for the calibrated gripper opening. This gate
    intentionally rejects rather than clips. Clipping a bad policy output would
    hide the model failure and create a trajectory the model did not request.
    """
    arm_limit = _positive_finite(max_arm_delta_rad, "max_arm_delta_rad")
    gripper_limit = _positive_finite(max_gripper_delta_rad, "max_gripper_delta_rad")

    guarded: dict[str, float] = {}
    violations: list[str] = []
    for name in JOINT_ACTION_NAMES:
        target = _finite_feature(requested, name, "postprocessed action")
        measured = _finite_feature(observation, name, "robot observation")
        delta = abs(target - measured)
        if delta > arm_limit:
            violations.append(
                f"{name}: |{target:.6f} - {measured:.6f}| = {delta:.6f} rad "
                f"> {arm_limit:.6f} rad"
            )
        guarded[name] = target

    target = _finite_feature(requested, GRIPPER_ACTION_NAME, "postprocessed action")
    measured = _finite_feature(observation, GRIPPER_ACTION_NAME, "robot observation")
    delta = abs(target - measured)
    if delta > gripper_limit:
        violations.append(
            f"{GRIPPER_ACTION_NAME}: |{target:.6f} - {measured:.6f}| = "
            f"{delta:.6f} rad > {gripper_limit:.6f} rad"
        )
    guarded[GRIPPER_ACTION_NAME] = target

    if violations:
        raise UnsafePolicyActionError(
            "postprocessed policy action rejected before actuator dispatch: "
            + "; ".join(violations)
        )
    return guarded


def validate_checkpoint_contract(
    checkpoint: str | Path,
    *,
    camera_name: str,
    camera_height: int,
    camera_width: int,
    camera_fps: float = 30.0,
    dataset_info: str | Path | None = None,
) -> dict[str, Any]:
    """Validate files and feature dimensions without opening any hardware."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete; missing files: {missing}")
    with (root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("use_peft"):
        missing_peft = [name for name in PEFT_MODEL_FILES if not (root / name).is_file()]
        if missing_peft:
            raise FileNotFoundError(
                f"PEFT checkpoint is incomplete; missing files: {missing_peft}"
            )
        with (root / "adapter_config.json").open(encoding="utf-8") as stream:
            adapter_config = json.load(stream)
        if not adapter_config.get("base_model_name_or_path"):
            raise ValueError(
                "PEFT adapter_config.json must declare base_model_name_or_path"
            )
    elif not (root / FULL_MODEL_FILE).is_file():
        raise FileNotFoundError(
            f"full checkpoint is incomplete; missing file: {FULL_MODEL_FILE}"
        )
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        with (root / processor_name).open(encoding="utf-8") as stream:
            processor = json.load(stream)
        state_files = [
            step["state_file"]
            for step in processor.get("steps", [])
            if step.get("state_file")
        ]
        missing_state = [name for name in state_files if not (root / name).is_file()]
        if missing_state:
            raise FileNotFoundError(
                f"checkpoint processor state is incomplete; missing files: {missing_state}"
            )

    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    image_key = f"observation.images.{camera_name}"
    expected_inputs = {"observation.state", image_key}
    if set(input_features) != expected_inputs:
        raise ValueError(
            f"checkpoint input features must be {sorted(expected_inputs)}, "
            f"received {sorted(input_features)}"
        )
    state_shape = tuple(input_features["observation.state"].get("shape", ()))
    image_shape = tuple(input_features[image_key].get("shape", ()))
    action_shape = tuple(output_features.get("action", {}).get("shape", ()))
    if state_shape != (len(ACTION_NAMES),):
        raise ValueError(
            f"checkpoint state shape must be ({len(ACTION_NAMES)},), "
            f"received {state_shape}"
        )
    expected_image_shape = (3, int(camera_height), int(camera_width))
    if image_shape != expected_image_shape:
        raise ValueError(
            f"checkpoint image shape must be {expected_image_shape}, received {image_shape}"
        )
    if action_shape != (len(ACTION_NAMES),):
        raise ValueError(
            f"checkpoint action shape must be ({len(ACTION_NAMES)},), "
            f"received {action_shape}"
        )

    info_path = _resolve_dataset_info(root, dataset_info)
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    dataset_features = info.get("features", {})
    expected_names = list(ACTION_NAMES)
    for key in ("observation.state", "action"):
        names = dataset_features.get(key, {}).get("names")
        if names != expected_names:
            raise ValueError(
                f"dataset {key} names must be {expected_names}, received {names}"
            )
    dataset_image = dataset_features.get(image_key, {})
    dataset_image_shape = tuple(dataset_image.get("shape", ()))
    expected_dataset_image_shape = (int(camera_height), int(camera_width), 3)
    if dataset_image_shape != expected_dataset_image_shape:
        raise ValueError(
            "dataset image shape must be "
            f"{expected_dataset_image_shape}, received {dataset_image_shape}"
        )
    if float(info.get("fps", -1)) != float(camera_fps):
        raise ValueError(
            f"dataset fps must match runtime fps {camera_fps}, received {info.get('fps')}"
        )
    return config


def _resolve_dataset_info(root: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
    else:
        train_config_path = root / "train_config.json"
        if not train_config_path.is_file():
            raise FileNotFoundError(
                "dataset meta/info.json is required to verify action names and order; "
                "pass --dataset-info or include train_config.json with an accessible dataset root"
            )
        with train_config_path.open(encoding="utf-8") as stream:
            train_config = json.load(stream)
        dataset_root = train_config.get("dataset", {}).get("root")
        if not dataset_root:
            raise FileNotFoundError(
                "train_config.json has no dataset.root; pass --dataset-info explicitly"
            )
        path = Path(dataset_root).expanduser().resolve() / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {path}")
    return path


def _dataset_features(camera_name: str, height: int, width: int) -> dict[str, dict]:
    names = list(ACTION_NAMES)
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": names},
        f"observation.images.{camera_name}": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (8,), "names": names},
    }


def _camera_source(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or run a LeRobot policy on LiteArm + LiteGrip + one camera."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--camera", default="0", help="OpenCV index or video-device path")
    parser.add_argument("--camera-name", default="wrist")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--dataset-info",
        type=Path,
        help="Training dataset meta/info.json. Auto-resolved from train_config.json when omitted.",
    )
    parser.add_argument("--arm-port")
    parser.add_argument("--gripper-channel", default="can0")
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--gripper-calibration", type=Path)
    calibration.add_argument("--gripper-mount", choices=("normal", "reverse"))
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--max-arm-action-delta-rad",
        type=float,
        default=DEFAULT_MAX_ARM_ACTION_DELTA_RAD,
        help=(
            "Reject a postprocessed joint target farther than this many radians "
            "from the latest measured joint position."
        ),
    )
    parser.add_argument(
        "--max-gripper-action-delta-rad",
        type=float,
        default=DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD,
        help=(
            "Reject a postprocessed gripper target farther than this many "
            "radians from the latest measured opening."
        ),
    )
    parser.add_argument(
        "--load-policy",
        action="store_true",
        help="Load the policy and processors without opening hardware.",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Open hardware and execute policy actions. Without this flag, only validate.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.task.strip():
        raise ValueError("task must not be empty")
    if not args.camera_name or "." in args.camera_name:
        raise ValueError("camera-name must be one non-empty feature segment without '.'")
    if args.camera_height <= 0 or args.camera_width <= 0 or args.fps <= 0.0:
        raise ValueError("camera width, height, and fps must be positive")
    arm_action_delta = _positive_finite(
        args.max_arm_action_delta_rad, "max-arm-action-delta-rad"
    )
    gripper_action_delta = _positive_finite(
        args.max_gripper_action_delta_rad, "max-gripper-action-delta-rad"
    )
    if args.gripper_calibration is not None and not args.gripper_calibration.is_file():
        raise FileNotFoundError(
            f"LiteGrip calibration file not found: {args.gripper_calibration}"
        )
    checkpoint_config = validate_checkpoint_contract(
        args.checkpoint,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        camera_fps=args.fps,
        dataset_info=args.dataset_info,
    )
    log.info(
        "Checkpoint contract is compatible: policy=%s state=8 action=8 camera=%s",
        checkpoint_config.get("type"),
        args.camera_name,
    )
    if not args.load_policy and not args.enable_motion:
        log.info("Validation-only mode: no arm, gripper, or camera was opened")
        return
    if args.enable_motion and args.duration_s <= 0.0:
        raise ValueError("duration-s must be positive")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    from litearm_lerobot import LiteArmGripCameraRobot, LiteArmGripCameraRobotConfig

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    policy_config.pretrained_path = checkpoint
    if args.device is not None:
        policy_config.device = args.device
    try:
        device = get_safe_torch_device(policy_config.device, log=True)
    except AssertionError as exc:
        raise RuntimeError(
            f"requested policy device is unavailable: {policy_config.device}"
        ) from exc
    policy_config.device = str(device)
    policy_class = get_policy_class(policy_config.type)
    if policy_config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(checkpoint, local_files_only=True)
        base_model = peft_config.base_model_name_or_path
        if not base_model:
            raise ValueError(
                "PEFT adapter_config.json must declare base_model_name_or_path"
            )
        policy = policy_class.from_pretrained(
            base_model,
            config=policy_config,
            local_files_only=True,
        )
        policy = PeftModel.from_pretrained(
            policy,
            checkpoint,
            config=peft_config,
            local_files_only=True,
        )
        policy.to(policy_config.device)
        policy.eval()
    else:
        policy = policy_class.from_pretrained(
            checkpoint,
            config=policy_config,
            local_files_only=True,
        )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": policy_config.device},
        },
    )
    log.info(
        "Policy and processors loaded: policy=%s device=%s peft=%s",
        policy_config.type,
        policy_config.device,
        policy_config.use_peft,
    )
    log.info(
        "Postprocessed action guard: arm_delta<=%.3f rad gripper_delta<=%.3f rad",
        arm_action_delta,
        gripper_action_delta,
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    if not args.enable_motion:
        log.info("Model-load validation finished: no hardware was opened")
        return

    camera_config = OpenCVCameraConfig(
        index_or_path=_camera_source(args.camera),
        fps=int(args.fps),
        width=args.camera_width,
        height=args.camera_height,
    )
    robot_config = LiteArmGripCameraRobotConfig(
        port=args.arm_port,
        cameras={args.camera_name: camera_config},
        enable_on_connect=True,
        gripper=LiteGripOptions(
            enabled=True,
            channel=args.gripper_channel,
            calibration_file=(
                str(args.gripper_calibration)
                if args.gripper_calibration is not None
                else None
            ),
            mount=args.gripper_mount,
        ),
    )
    robot = LiteArmGripCameraRobot(robot_config)
    features = _dataset_features(args.camera_name, args.camera_height, args.camera_width)

    period = 1.0 / args.fps
    overruns = 0
    steps = 0
    with robot:
        deadline = time.monotonic() + args.duration_s
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            observation = robot.get_observation()
            frame = build_dataset_frame(features, observation, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            requested = make_robot_action(action_tensor, features)
            guarded = guard_postprocessed_action(
                requested,
                observation,
                max_arm_delta_rad=arm_action_delta,
                max_gripper_delta_rad=gripper_action_delta,
            )
            accepted = robot.send_action(guarded)
            log.debug("accepted action: %s", accepted)
            steps += 1

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                overruns += 1
                next_tick = time.monotonic()
    log.info("Policy run finished: steps=%d deadline_overruns=%d", steps, overruns)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
