English | [中文](README.zh-CN.md)

This repository records policy-training LiteArm demonstrations and runs trained LeRobot policy checkpoints, for developers using LeRobot 0.4.4.

# litearm-vla

Two workflows share one data contract: direct, single-arm kinesthetic
collection, and policy deployment on the same arm, gripper, and camera. Neither
starts, connects to, or imports `litearm-vr-follow`.

```text
Record:  LeRobot dataset/cameras
                  ↑
         litearm-vla (30 Hz q[t] -> q[t+1])
                  ↑
         litearm-lerobot (Robot lifecycle and firmware limits)
                  ↑
         litearm-python (USB CDC and zero-gravity keepalive)

Deploy:  LeRobot policy (30 Hz)
                  ↑
         litearm-vla.deployment (delta gate, checkpoint validation)
                  ↑
         litearm-lerobot (LiteArmGripCameraRobot + servo loops)
                  ↑
         litearm-python (USB CDC) + litegrip-python (SocketCAN)
```

The gripper controller and the combined adapter live in `litearm-lerobot`:
`litearm_lerobot.gripper` → `litegrip-python` → SocketCAN. The recording
adapter reuses that controller and keeps its own safety monitor.

## Data contract

- Seven arm joints. The LiteGrip is optional and off by default.
- `observation.state[t]` is the measured joint vector `q[t]`, plus
  `gripper.pos` when the gripper is enabled. Every component is in radians;
  `gripper.pos` is the calibrated opening angle (`position_mm / rad_to_mm`,
  0 rad at the closed stop).
- `action[t]` is the next measured vector `q[t+1]`, including
  `gripper.pos` in the same radians when the gripper is enabled. The gripper
  follows the same rule as the joints, so its action is continuous while the
  jaws move.
- Datasets recorded before this unit rule stored `gripper.pos` in millimetres
  under the same feature name. Do not resume or mix them with radian datasets;
  re-record them, or convert the column with the gripper's calibrated
  `rad_to_mm` and regenerate the dataset statistics.
- A policy command uses the same radians. `LiteArmGripCameraRobot` does not stream
  a continuous gripper position: a target at or below the midpoint of the
  calibrated travel becomes a grasp, and a target above it opens the jaws. The
  observation still reports the measured opening continuously.
- The final observation is dropped because it has no future action.
- Recording is fixed at 30 Hz and slow episodes are discarded.
- LeRobot owns cameras, dataset features, video encoding, episode save, resume,
  re-record controls, and optional Hub upload.

## Install and test

Use the existing project environment; do not install into the system Python:

```bash
conda activate litearm-lerobot
cd /home/yd/wl/VLA/litearm-lerobot
python -m pip install -e .
cd /home/yd/wl/VLA/litearm-vla
python -m pip install -e .

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q -p no:cacheprovider -rs
```

`litegrip` is not on PyPI. `litearm-lerobot` pins the upstream Git commit and
installs it, so `litearm-vla` gets the SDK through
`pip install -e ../litearm-lerobot` above.

The tests fake the arm, cameras, and gripper. They do not open serial, camera,
CAN, or network robot devices.

## Hardware preflight

The recorder fails closed, so validate both devices before the first episode.

### Camera node

`index_or_path` must point at the **color** node. An Intel RealSense D435i
exposes six V4L2 nodes, and `0` is the depth stream (`Z16`), not color:

```bash
for d in /sys/class/video4linux/video*; do
  echo "$(basename "$d"): $(cat "$d/device/interface")"
done
```

Use the node whose interface ends in `RGB` (often `/dev/video4`, but the number
is not stable across replugs) and pass the device path rather than a bare index.
Confirm the camera is on USB 3 and that frames actually arrive:

```bash
for d in /sys/bus/usb/devices/*/product; do
  p=$(cat "$d" 2>/dev/null)
  case "$p" in
    *RealSense*) echo "$(basename "$(dirname "$d")"): speed=$(cat "$(dirname "$d")/speed")" ;;
  esac
done
timeout 15 gst-launch-1.0 -q v4l2src device=/dev/video4 num-buffers=30 ! fakesink
echo "exit=$?"
```

The camera's `speed` should be `5000` (USB 3) and the capture test must print
`exit=0` immediately. If the node exists but no frames arrive, unplug and
replug the camera to power-cycle it; a software USB reset did not recover it.

### Arm starting pose

Before enabling, the recorder requires every joint to be inside the
firmware-derived limits with a 0.01 rad margin. A disabled arm sags under its
own weight — J4 has only about 0.0175 rad of positive travel — so a limp arm
often fails this gate and the recorder refuses to start. Home the arm through
the SDK and leave it enabled:

```python
from litearm import Arm

arm = Arm(port="/dev/ttyACM0")
arm.connect()
if arm.get_state(refresh=True).value.enabled:
    arm.disable()
state = arm.get_state(refresh=True).value
if state.flag_names or state.faulted:
    arm.clear_faults()
    state = arm.get_state(refresh=True).value
    if state.faulted:
        arm.reset()
arm.enable()
arm.home(timeout=60)   # firmware moves at its fixed 0.10 speed
arm.close()            # close() stops the link; it does not disable the arm
```

`Arm.home()` is allowed to start from a soft-limit pose, and leaving the arm
enabled means the recorder skips `enable()` and enters zero-gravity
immediately. Do not disable the arm to park it: it will sag out of the limits.

### Gripper CAN link

The LiteGrip uses SocketCAN, not the arm's USB CDC. Bring the interface up at
1 Mbit/s before every session and keep it exclusive to this process:

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

Then check the SDK's calibrated limits and enable path without moving the jaws:

```bash
python - <<'PY'
from litegrip import LiteGrip

with LiteGrip("can0") as gripper:
    print("calibration:", gripper.load_calibration())
    print("travel mm:", abs(gripper.config.pos_open_rad - gripper.config.pos_closed_rad)
          * gripper.config.rad_to_mm)
    print("enabled:", gripper.enable().ok)
PY
```

If `load_calibration()` returns `False`, the factory file is for a different
channel (it declares `can0`) or the local calibration is missing; pass
`--robot.gripper.mount=reverse` or `--robot.gripper.calibration_file` for a
non-standard unit. Do not start recording on placeholder limits.

## Record

Do not run this command until `litearm-vr-follow`, LiteArm Studio, and every
other arm controller are stopped. Keep a physical emergency stop within reach.
Starting the command enters zero-gravity within a few seconds.

```bash
cd /home/yd/wl/VLA/litearm-vla
litearm-vla-record \
  --robot.type=litearm_kinesthetic \
  --robot.id=armA-hand-guide \
  --robot.auto_enable=true \
  --robot.auto_clear_faults=true \
  --robot.ignored_flags='["WD_TRIPPED"]' \
  --robot.payload_kg=0.62 \
  --robot.payload_com='[0, 0, 0.03]' \
  --robot.gripper.enabled=true \
  --robot.gripper.channel=can0 \
  --robot.cameras='{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=local/litearm-hand-guide-smoke-v1 \
  --dataset.root=data/litearm-hand-guide-smoke-v1 \
  --dataset.single_task='move the arm by hand through the task' \
  --dataset.fps=30 \
  --dataset.episode_time_s=8 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=1 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --play_sounds=false
```

`auto_enable` defaults to `false`. Setting it to `true` is an explicit request
to energize the arm. `--robot.gripper.enabled=true` is the equivalent opt-in for
the gripper: it opens the CAN link, loads the calibration, and energizes the
gripper motor before the dataset is created.

`--robot.payload_kg=0.62` writes the tool payload into the firmware's gravity
feed-forward before the arm is enabled, then reads item 4 back with
`get_ff_scalar()` and fails closed if the firmware kept a different value.
`--robot.payload_com='[0, 0, 0.03]'` places its centre of mass 3 cm along the
flange `+z` axis; `set_payload()` writes both values. The write is RAM-only;
call `Arm.save_params()` yourself if it must survive a power cycle. The process
opens the arm exclusively, validates fresh feedback and firmware limits, enters
zero-gravity once, and keeps it active across recording, reset, discard, and
save phases.

The terminal marks `[PHASE: RECORDING]` with `data capture ON` and
`[PHASE: RESET]` with `data capture OFF`. Routine LiteGrip `INFO` messages are
hidden so the phase remains clear; warnings and errors remain visible. Add
`--verbose_gripper_logs=true` when diagnosing the gripper SDK.

- Right arrow: finish the current recording or reset phase early.
- Left arrow: discard the current episode and record it again.
- Escape: stop the session, finalize the dataset, leave zero-gravity, hold the
  measured pose, and close USB CDC.
- `o`: open the gripper (`--robot.gripper.open_key`).
- `c`: close and hold the gripper with `grasp()`
  (`--robot.gripper.close_key`, force `--robot.gripper.grasp_force_n`).

A normal exit stops zero-gravity, verifies a fresh safe state, performs a
zero-displacement `movej` hand-off, and closes the link. It does **not** disable
the arm, so the arm stays enabled holding the measured pose. A safety fault
takes the same path: zero-gravity stops and the arm holds its measured pose. This
program never emergency-stops or disables the arm; the physical emergency stop
stays the emergency path.

The gripper follows the SDK lifecycle instead: between moves a background
worker streams zero-displacement hold frames, and shutdown sends a zero-torque
frame and closes the CAN session, which disables the motor. The jaws release on
exit, so do not rely on the gripper to hold a part after the process ends.
Keyboard control needs a graphical session; with `--robot.gripper.enabled=true`
a headless run stops before opening either device.

See [the recording guide](docs/kinesthetic-recording.md) for the staged hardware
gate and dataset checks.

## Deploy a policy

`LiteArmGripCameraRobot` from `litearm-lerobot` (registered as
`litearm_grip_camera`) runs an eight-dimensional arm-and-gripper checkpoint on
the same hardware. Validate the checkpoint without opening the arm, CAN bus, or
camera:

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json
```

Add `--enable-motion` only after the hardware gates in
[the policy deployment guide](docs/policy-deployment.md). That guide covers the
delta gate (`0.10 rad` per joint, `0.05 rad` for the gripper), the staged
hardware checks, the bounded first run, and the controlled stop on `Ctrl+C`.

## Troubleshooting

**`WD_TRIPPED`.** The firmware's 0.10 s command watchdog expired between
zero-gravity keepalives. An enabled arm with no keepalive — for example right
after LiteArm Studio exits — latches it, and the keepalive re-asserts `ZERO_G`
on the next period, so the trip is transient. Pass
`--robot.ignored_flags='["WD_TRIPPED"]'` to log it as a warning instead of
failing the session. Every other flag, the temperature and limit checks, and the
host-side keepalive health (`zero_g_error` / `zero_g_active`) stay fatal.

While the arm is enabled `clear_faults()` does not stick (the watchdog re-trips
immediately); disable first, then `clear_faults()`, then `reset()` if the mode
stays `EMERGENCY`.

**Latched faults block recording.** `FAULT`, `POS_VIOL`, `FB_STALE`, and
`joint_fault` are real safety signals and the recorder refuses to start while
any of them is set. Pass `--robot.auto_clear_faults=true` to clear a fault left
latched by a previous fail-closed session: it only runs on a disabled arm, logs
what it cleared, and re-runs the full safety check. It never clears while the arm
is enabled, and a fault whose physical cause persists simply re-latches.
Repeated `FB_STALE` on different joints while enabling points at the CAN side:
check the motor cabling and power-cycle the arm.

**The gripper fails to enable with `[Errno 100] Network is down`.** The
SocketCAN interface exists but is administratively down. `LiteGrip.connect()`
can open a raw socket on a down interface; the first transmit fails, so
`enable()` retries three times and returns `err=0`. Bring it up first:

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

`state UP` and `can state ERROR-ACTIVE` are required. If the interface drops
again, check the USB adapter (`gs_usb`) with `dmesg`, the cable, and the 120 Ω
bus termination.

**The gripper ignored the local calibration.** A no-argument
`load_calibration()` reads only the shipped factory file; the SDK deliberately
does not pick up `~/.litegrip/can0_08_calibration.json`. Pass it explicitly:

```bash
--robot.gripper.calibration_file=/home/yd/.litegrip/can0_08_calibration.json
```

or set `LITEGRIP_CALIB` for the session. Use the factory file only when the
unit is the shipped normal-mount configuration.

**The episode is static.** `unique rows: 1` in the dataset check means the arm
was never moved. Zero-gravity makes the arm easy to drag, but it does not move
on its own.

**`lerobot-edit-dataset` cannot edit a local dataset.** Without `--root` it only
checks `~/.cache/huggingface/lerobot/<repo_id>`, then queries the Hub, so
`RepositoryNotFoundError` is not an authentication problem. With `--root`,
LeRobot 0.4.4 handles local paths inconsistently: deletion treats the path as
both the dataset and its parent, and merge applies one root to every source.
Call `delete_episodes()` or `merge_datasets()` directly with explicit paths; see
[section 6](docs/kinesthetic-recording.md#6-remove-a-failed-episode) and
[section 7](docs/kinesthetic-recording.md#7-merge-datasets) of the recording
guide.

**Video encoding and the watchdog.** LeRobot's default `libsvtav1` was measured
stalling the recording process for more than 100 ms while encoding an episode.
Sessions using both `libsvtav1` and `h264` have still failed closed at save time
on the reference machine while the host-side keepalive gap stayed below 44 ms,
so the cause is under investigation; USB bus contention between the camera and
the arm is one suspect. `--dataset.vcodec=h264` shortens the encoding window and
produces video that OpenCV can decode directly, but it did not by itself
eliminate the failure. Keep the camera on a USB 3 port (ideally a separate
controller from the arm). With `--robot.ignored_flags='["WD_TRIPPED"]'` a trip
during save is logged and the session continues; without it the session fails
closed.

## Safety boundary

Offline tests prove the data contract and fail-closed call order only. They do
not prove zero-gravity behavior, physical holding, camera synchronization, USB
stability, or safe operation on a real arm. Start with an unloaded, constrained
workspace and one short episode.

## Repository standards

This repository follows the shared NEXFORM ROBOTICS repository standards: the
agent operating rules in [`.nexform-standards.json`](.nexform-standards.json), Conventional Commits, and
automated semantic-release versioning on every merge to `main`.

## License

Copyright © 2026 NEXFORM ROBOTICS. Licensed under the
[Apache License 2.0](LICENSE).
