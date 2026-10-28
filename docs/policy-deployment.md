This guide shows deployment engineers how to validate and run a LeRobot checkpoint on one seven-joint LiteArm, one LiteGrip, and one camera.

# Deploy a LeRobot policy

## 1. Match the hardware and feature contract

Use `LiteArmGripCameraRobot` from `litearm-lerobot`, registered as
`litearm_grip_camera`, for exactly one LiteArm, one LiteGrip, and one camera.
It exposes scalar raw features that LeRobot 0.4.4 aggregates in this fixed
order:

```text
state/action: joint1.pos ... joint7.pos, gripper.pos
image:        observation.images.wrist, shape [3, 480, 640]
```

Every value is in radians. `gripper.pos` is the calibrated opening angle
(`position_mm / rad_to_mm`, 0 at the closed stop), the same unit the recorder
writes, so the state vector has one unit throughout. The camera name is
configurable, but the adapter requires exactly one camera and the name must
match the checkpoint.

Do not use the arm-only `LiteArmRobot` seven-dimensional vector contract with an
eight-dimensional arm-and-gripper checkpoint. Do not convert `gripper.pos` to
millimetres: datasets recorded before the radian rule stored millimetres under
the same feature name, and LeRobot checks names and shapes, not units.

## 2. Understand command ownership

```text
LeRobot policy (30 Hz)
  └─ postprocessed-action delta gate
       ├─ LiteArm ServoLoop (250 Hz) → litearm-python → USB CDC → STM32
       └─ LiteGripController (200 Hz hold stream) → litegrip-python → SocketCAN → DM4310
```

The policy thread only replaces targets. Each background loop is the only
sustained command writer for its actuator.

The arm follows position targets continuously. The gripper is commanded in two
discrete modes: a target at or below the midpoint of the calibrated travel is a
grasp, and a target above it opens the jaws. The observation still reports the
measured opening continuously in radians. The grasp uses the SDK force
setpoint, which is a nominal torque conversion, not a calibrated force
measurement. This path does not claim contact hold or slip recovery.

After LeRobot restores physical units, the deployment runner compares every
target with the observation that produced it. It rejects the complete action
before actuator dispatch when any joint changes by more than `0.10 rad` or the
gripper changes by more than `0.05 rad` (about 3.1 mm at the shipped
61.01 mm/rad scale). It does not clip an unsafe output into a different
trajectory. These are model-output sanity limits, not replacements for actuator
limits, collision detection, or an E-stop.

The gripper controller defaults to a 200 Hz hold stream, a 50 mm/s open speed, a
20 N grasp setpoint, a 0.2-second grasp slice, and a 0.25-second feedback
timeout. It stops on a motor fault, stale feedback, or 70 °C at the hottest of
the MOS and coil sensors. These are conservative software stops, not
motor-rating or thermal-safety certification.

## 3. Install and validate without hardware

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../litearm-lerobot
python -m pip install -e .
```

The package pins LeRobot 0.4.4 and the tested LiteGrip SDK commit. Validate a
checkpoint without opening the arm, CAN bus, or camera:

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json
```

Without `--enable-motion`, the command only checks checkpoint completeness,
processor state files, model dimensions, image name and shape, dataset FPS, and
the semantic state/action name order. It reads dataset metadata through
`train_config.json`. If the dataset moved, add:

```bash
--dataset-info /absolute/path/to/dataset/meta/info.json
```

Checkpoint completeness accepts either a full `model.safetensors`, or the
`adapter_config.json` and `adapter_model.safetensors` pair produced by
LoRA/PEFT training. A PEFT adapter must name its base model, and the deployment
machine must already cache that model or be able to retrieve it.

Load the model, PEFT adapter, normalization processors, and target device
without opening hardware:

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --dataset-info /absolute/path/to/dataset/meta/info.json \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json \
  --device cuda \
  --load-policy
```

Continue only after this prints `Model-load validation finished`. This check
does not prove that policy actions are safe or that the task will succeed.

## 4. Clear the hardware gates

Before policy motion, verify each layer separately:

1. Verify the E-stop, support the arm, clear the workspace, and remove fragile objects from the gripper.
2. Read fresh seven-joint state, limits, flags, faults, and temperatures through `litearm-python` without commanding motion.
3. Load the measured LiteGrip calibration and read position, error code, and temperature through `litegrip-python` without commanding motion.
4. Verify one RGB camera produces `480×640×3` frames at 30 Hz under the `wrist` key.
5. Run separate low-speed, small-range, unloaded arm and gripper motion tests and verify physical direction and units.
6. Repeat the validation-only command before allowing the policy to take control.

Prefer a measured `--gripper-calibration` file. `--gripper-mount normal|reverse`
loads a direction template and does not prove the travel of the installed unit.

## 5. Run a bounded motion session

The following command moves real hardware:

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --dataset-info /absolute/path/to/dataset/meta/info.json \
  --task "pick the red cube" \
  --camera /dev/video0 \
  --camera-name wrist \
  --camera-width 640 \
  --camera-height 480 \
  --fps 30 \
  --gripper-channel can0 \
  --gripper-calibration /absolute/path/to/can0_calibration.json \
  --max-arm-action-delta-rad 0.10 \
  --max-gripper-action-delta-rad 0.05 \
  --duration-s 10 \
  --enable-motion
```

Only `--enable-motion` enables both actuators. Keep the first run unloaded and
limited to 10 seconds with an operator at the E-stop. Add
`--arm-port /dev/serial/by-id/...` only when automatic detection is unsuitable.
Do not hard-code `/dev/ttyACM0`; CDC numbers can swap across boots.

Do not raise the action-delta limits merely to make a rejected checkpoint run.
First measure the training dataset's per-frame `abs(action - observation)`
distribution and investigate normalization, feature order, FPS, and camera
mismatches. A rejection means no part of that action reaches `send_action()`.

`send_action()` returns the accepted arm targets and the clamped gripper
opening, not the unbounded policy output.

An exception or `Ctrl+C` stops the arm policy servo first, then sends a
zero-displacement `movej` to the measured joint pose. This hands control to the
firmware's rigid position-hold mode without disabling the arm. Gripper and
camera cleanup runs only after that hand-off. This is a controlled software
stop, not `Arm.emergency_stop()`: the SDK emergency stop latches a fault and
removes holding torque. The runner does not automatically reset a fault and
re-enable the arm.

This path cannot guarantee holding after `kill -9`, process death, USB loss,
power loss, or a failed hold hand-off. Keep physical support and an operator at
the hardware E-stop during initial live tests.

## 6. Know the evidence boundary

Offline tests cover the feature contract, lifecycle, clamping, timeout, and
error propagation. They do not prove USB, SocketCAN, camera, physical motion,
load handling, timing stability, or task success. A short smoke-trained
checkpoint validates the loading path only; it is not evidence of task ability.

中文步骤见 [policy-deployment.zh-CN.md](policy-deployment.zh-CN.md)。
