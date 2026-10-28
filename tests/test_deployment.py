"""Hardware-free tests for deployment checkpoint validation."""
from __future__ import annotations

import json

import pytest

from litearm_vla.deployment import (
    DEFAULT_MAX_ARM_ACTION_DELTA_RAD,
    DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD,
    UnsafePolicyActionError,
    build_parser,
    guard_postprocessed_action,
    run,
    validate_checkpoint_contract,
)
from litearm_lerobot import ACTION_NAMES


def _positions(value=0.0):
    return {name: float(value) for name in ACTION_NAMES}


def _checkpoint(
    tmp_path,
    *,
    state=8,
    action=8,
    image=(3, 480, 640),
    camera="wrist",
    omit=(),
    names=None,
    use_peft=False,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config = {
        "type": "smolvla",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [state]},
            f"observation.images.{camera}": {"type": "VISUAL", "shape": list(image)},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [action]},
        },
        "use_peft": use_peft,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if use_peft:
        if "adapter_config.json" not in omit:
            (root / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "lerobot/smolvla_base"}),
                encoding="utf-8",
            )
        if "adapter_model.safetensors" not in omit:
            (root / "adapter_model.safetensors").write_bytes(b"test")
    elif "model.safetensors" not in omit:
        (root / "model.safetensors").write_bytes(b"test")
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if processor_name not in omit:
            (root / processor_name).write_text(
                json.dumps({"name": processor_name, "steps": []}),
                encoding="utf-8",
            )
    dataset_root = tmp_path / "dataset"
    info_dir = dataset_root / "meta"
    info_dir.mkdir(parents=True)
    feature_names = names or [
        "joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos",
        "joint5.pos", "joint6.pos", "joint7.pos", "gripper.pos",
    ]
    info = {
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [8], "names": feature_names,
            },
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": [image[1], image[2], image[0]],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32", "shape": [8], "names": feature_names,
            },
        },
    }
    (info_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "train_config.json").write_text(
        json.dumps({"dataset": {"root": str(dataset_root)}}),
        encoding="utf-8",
    )
    return root


def test_compatible_checkpoint_passes_without_loading_model_or_hardware(tmp_path):
    root = _checkpoint(tmp_path)
    config = validate_checkpoint_contract(
        root,
        camera_name="wrist",
        camera_height=480,
        camera_width=640,
    )
    assert config["type"] == "smolvla"


def test_compatible_peft_checkpoint_passes_without_loading_model_or_hardware(tmp_path):
    root = _checkpoint(tmp_path, use_peft=True)
    config = validate_checkpoint_contract(
        root,
        camera_name="wrist",
        camera_height=480,
        camera_width=640,
    )
    assert config["use_peft"] is True


@pytest.mark.parametrize("missing", ["adapter_config.json", "adapter_model.safetensors"])
def test_incomplete_peft_checkpoint_is_rejected(tmp_path, missing):
    root = _checkpoint(tmp_path, use_peft=True, omit=(missing,))
    with pytest.raises(FileNotFoundError, match=missing):
        validate_checkpoint_contract(
            root,
            camera_name="wrist",
            camera_height=480,
            camera_width=640,
        )


def test_peft_checkpoint_requires_a_base_model(tmp_path):
    root = _checkpoint(tmp_path, use_peft=True)
    (root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": ""}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        validate_checkpoint_contract(
            root,
            camera_name="wrist",
            camera_height=480,
            camera_width=640,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state": 7}, "state shape"),
        ({"action": 7}, "action shape"),
        ({"image": (3, 224, 224)}, "image shape"),
        ({"camera": "front"}, "input features"),
    ],
)
def test_incompatible_checkpoint_fails_closed(tmp_path, kwargs, message):
    root = _checkpoint(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_contract(
            root,
            camera_name="wrist",
            camera_height=480,
            camera_width=640,
        )


def test_incomplete_checkpoint_is_rejected(tmp_path):
    root = _checkpoint(tmp_path, omit=("policy_postprocessor.json",))
    with pytest.raises(FileNotFoundError, match="policy_postprocessor"):
        validate_checkpoint_contract(
            root,
            camera_name="wrist",
            camera_height=480,
            camera_width=640,
        )


def test_dataset_action_order_is_verified(tmp_path):
    names = [
        "gripper.pos", "joint1.pos", "joint2.pos", "joint3.pos",
        "joint4.pos", "joint5.pos", "joint6.pos", "joint7.pos",
    ]
    root = _checkpoint(tmp_path, names=names)
    with pytest.raises(ValueError, match="names must be"):
        validate_checkpoint_contract(
            root,
            camera_name="wrist",
            camera_height=480,
            camera_width=640,
        )


def test_cli_defaults_to_validation_only(tmp_path):
    root = _checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--checkpoint", str(root),
            "--task", "pick the cube",
            "--gripper-mount", "normal",
        ]
    )
    assert args.enable_motion is False
    assert args.load_policy is False
    assert args.camera_name == "wrist"
    assert (args.camera_height, args.camera_width) == (480, 640)
    assert args.max_arm_action_delta_rad == DEFAULT_MAX_ARM_ACTION_DELTA_RAD
    assert args.max_gripper_action_delta_rad == DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD
    run(args)


def test_validation_rejects_a_missing_explicit_gripper_calibration(tmp_path):
    root = _checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--checkpoint", str(root),
            "--task", "pick the cube",
            "--gripper-calibration", str(tmp_path / "missing.json"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="calibration file"):
        run(args)


def test_postprocessed_action_guard_preserves_a_safe_action():
    observation = _positions()
    requested = _positions()
    requested["joint2.pos"] = 0.09
    requested["gripper.pos"] = 0.04

    guarded = guard_postprocessed_action(requested, observation)

    assert guarded == requested


@pytest.mark.parametrize("source", ["requested", "observation"])
def test_postprocessed_action_guard_rejects_non_finite_values(source):
    observation = _positions()
    requested = _positions()
    (requested if source == "requested" else observation)["joint3.pos"] = float("nan")

    with pytest.raises(UnsafePolicyActionError, match="non-finite"):
        guard_postprocessed_action(requested, observation)


def test_postprocessed_action_guard_rejects_an_arm_jump():
    observation = _positions()
    requested = _positions()
    requested["joint6.pos"] = DEFAULT_MAX_ARM_ACTION_DELTA_RAD + 0.001

    with pytest.raises(UnsafePolicyActionError, match=r"joint6\.pos.*rad"):
        guard_postprocessed_action(requested, observation)


def test_postprocessed_action_guard_rejects_a_gripper_jump():
    observation = _positions()
    requested = _positions()
    requested["gripper.pos"] = DEFAULT_MAX_GRIPPER_ACTION_DELTA_RAD + 0.001

    with pytest.raises(UnsafePolicyActionError, match=r"gripper\.pos.*rad"):
        guard_postprocessed_action(requested, observation)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_arm_delta_rad": 0.0}, "max_arm_delta_rad"),
        ({"max_gripper_delta_rad": float("inf")}, "max_gripper_delta_rad"),
    ],
)
def test_postprocessed_action_guard_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        guard_postprocessed_action(_positions(), _positions(), **kwargs)
