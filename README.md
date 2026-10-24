English | [中文](README.zh-CN.md)

This repository records policy-training LiteArm demonstrations for developers using LeRobot 0.4.4.

# litearm-vla

The first supported workflow is direct, single-arm kinesthetic collection. It
does not start, connect to, or import `litearm-vr-follow`.

```text
LeRobot dataset/cameras
          ↑
litearm-vla (30 Hz q[t] -> q[t+1])
          ↑
litearm-lerobot (Robot lifecycle and firmware limits)
          ↑
litearm-python (USB CDC and zero-gravity keepalive)
```

## Data contract

- Seven arm joints, no gripper in this first version.
- `observation.state[t]` is the measured joint vector `q[t]`.
- `action[t]` is the next measured vector `q[t+1]`.
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

The tests fake the arm and cameras. They do not open serial, camera, CAN, or
network robot devices.

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
to energize the arm. The process opens the arm exclusively, validates fresh
feedback and firmware limits, enters zero-gravity once, and keeps it active
across recording, reset, discard, and save phases.

- Right arrow: finish the current recording or reset phase early.
- Left arrow: discard the current episode and record it again.
- Escape: stop the session, finalize the dataset, leave zero-gravity, hold the
  measured pose, and close USB CDC.

A normal exit stops zero-gravity, verifies a fresh safe state, performs a
zero-displacement `movej` hand-off, and closes the link. It does **not** disable
the arm, so the arm stays enabled holding the measured pose. A safety fault
takes the same path: zero-gravity stops and the arm holds its measured pose. This
program never emergency-stops or disables the arm; the physical emergency stop
stays the emergency path.

See [the recording guide](docs/kinesthetic-recording.md) for the staged hardware
gate and dataset checks.

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

**The episode is static.** `unique rows: 1` in the dataset check means the arm
was never moved. Zero-gravity makes the arm easy to drag, but it does not move
on its own.

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

## License

Copyright © 2026 NEXFORM ROBOTICS. Licensed under the
[Apache License 2.0](LICENSE).
