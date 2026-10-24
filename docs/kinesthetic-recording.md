English | [中文](kinesthetic-recording.zh-CN.md)

This guide helps LiteArm operators validate and run direct 30 Hz LeRobot kinesthetic collection safely.

# Direct kinesthetic recording

## 1. Confirm the software gate

Work from `/home/yd/wl/VLA/litearm-vla` in the
`litearm-lerobot` Conda environment:

```bash
python -c 'import lerobot, litearm, litearm_lerobot; print(lerobot.__version__, litearm.__version__)'
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q -p no:cacheprovider -rs
```

Expected versions are LeRobot `0.4.4` and litearm-python `2.1.0`. All offline
tests must pass before hardware work.

## 2. Prepare the hardware

1. Stop `litearm-vr-follow`, LiteArm Studio, and every other process that can
   open the arm. The recorder must be the only CDC owner.
2. Fix the arm base, remove payloads, clear the workspace, and keep a second
   person at the physical emergency stop.
3. Confirm all seven joints report finite positions, no fault bits, fresh
   timestamps, and plausible motor and coil temperatures with the SDK's
   read-only diagnostics.
4. Confirm the configured camera is the intended color node. On an Intel
   RealSense D435i, `index_or_path: 0` is the depth stream (`Z16`), not color:

   ```bash
   for d in /sys/class/video4linux/video*; do
     echo "$(basename "$d"): $(cat "$d/device/interface")"
   done
   ```

   Pass the `RGB` node as a device path. Check that the camera is on USB 3
   (`speed` should be `5000`) and that frames actually arrive:

   ```bash
   timeout 15 gst-launch-1.0 -q v4l2src device=/dev/video4 num-buffers=30 ! fakesink
   echo "exit=$?"
   ```

   `exit=0` means frames are flowing. If the node exists but no frames arrive,
   unplug and replug the camera to power-cycle it; a software USB reset does not
   recover it. Do not run a tool that kills camera owners automatically.

5. Put the arm inside its firmware limits before the recorder starts. The
   recorder checks every joint against the limits minus a 0.01 rad margin while
   the arm is still disabled, and a limp arm sags out of range — J4 has only
   about 0.0175 rad of positive travel. Home the arm through the SDK and leave
   it enabled; `Arm.close()` does not disable:

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
   arm.home(timeout=60)
   arm.close()
   ```

   An already-enabled arm is fine: the recorder skips `enable()` and enters
   zero-gravity immediately.

Do not continue from an ACK alone. The gate requires fresh physical feedback
and a known emergency-stop path.

## 3. Validate zero-gravity separately

Use litearm-python's public `Arm.zero_g()` API in a dedicated low-risk check
before recording. Keep travel small and slow. Verify that the arm does not
jump, self-move, overheat, or fall when zero-gravity ends.

This repository has not run that hardware check automatically. A Python test
cannot prove real gravity compensation, friction, backlash, load behavior, or
the firmware's exit handoff.

## 4. Record one smoke episode

Run the command from the README with `episode_time_s=8`, `num_episodes=1`, and
`push_to_hub=false`. `--robot.auto_enable=true` is required only when the arm is
currently disabled; it is an explicit energizing action. If the arm is already
enabled and fault-free (for example right after `home()`), the recorder skips
enabling and enters zero-gravity straight away.

The command enters zero-gravity within a few seconds of starting. Announce it,
stand next to the arm, and actually drag the arm during the episode; a
zero-gravity arm is easy to move but never moves by itself.

A latched fault blocks the run. Any of `FAULT`, `POS_VIOL`, `FB_STALE`, or
`joint_fault` makes `robot.connect()` raise before the arm is energized. The
README command passes `--robot.auto_clear_faults=true`, which clears a fault
latched by a previous fail-closed session before re-running the safety check; it
only runs on a disabled arm and never while the arm is enabled. Clear the
physical cause too — while the arm is enabled, `clear_faults()` does not stick
because the command watchdog re-trips immediately.

The README command passes `--robot.ignored_flags='["WD_TRIPPED"]'`, which logs a
transient command-watchdog trip as a warning instead of failing the session.
Remove the flag to run in strict fail-closed mode. Every other flag, the
temperature and limit checks, and the host-side keepalive health stay fatal
either way.

The process creates the dataset only after CDC connection, firmware-limit
reads, a disabled-pose safety check, immediate zero-gravity entry after enable,
zero-gravity confirmation, safety-monitor startup, and camera connection all
succeed. The immediate handoff is required because the firmware command
watchdog starts when the arm is enabled.

The safety monitor remains active while LeRobot saves and encodes an episode.
It checks feedback age, enable state, zero-gravity keepalive, fault bits,
temperatures, and the firmware-derived limits. A violation stops zero-gravity,
hands the arm off to a zero-displacement hold (the arm stays enabled), closes the
session, and reports the cause. The program never emergency-stops or disables the
arm; use the physical emergency stop if the arm does not hold safely.

On the reference machine this save/encode phase has failed closed with
`WD_TRIPPED` even though the host-side keepalive gap stayed below 44 ms. Treat
such a fault as real, not as a fluke, and see the README troubleshooting section
before retrying.

## 5. Check the dataset

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/litearm-hand-guide-smoke-v1")
dataset = LeRobotDataset("local/litearm-hand-guide-smoke-v1", root=root)
observation = np.asarray([dataset[i]["observation.state"] for i in range(len(dataset))])
action = np.asarray([dataset[i]["action"] for i in range(len(dataset))])
episodes = dataset.meta.episodes
bounds = list(zip(episodes["dataset_from_index"], episodes["dataset_to_index"]))

print("fps:", dataset.fps)
print("episodes:", dataset.num_episodes)
print("frames:", len(dataset))
print("observation:", observation.shape, "action:", action.shape)
print("unique states:", len(np.unique(np.round(observation, 6), axis=0)))
print("per-joint range:", np.round(observation.max(0) - observation.min(0), 4))
print("finite:", np.isfinite(observation).all() and np.isfinite(action).all())
print(
    "max pairing error:",
    max(
        np.abs(action[start:stop - 1] - observation[start + 1:stop]).max()
        for start, stop in bounds
        if stop - start > 1
    ),
)
PY
```

Pass only when the dataset reopens, reports 30 Hz, contains finite `(N, 7)`
state and action arrays, and the video matches the physical task. The pairing
error should be near zero because `action[t]` is `observation.state[t+1]`.

Check pairing **within each episode**. The last frame of an episode has no
future action in that episode, so its `action` is the episode's final measured
state. Comparing `action[:-1]` with `observation[1:]` over the whole dataset
pairs the end of one episode with the start of the next and reports a large
error that is not a data-contract violation.

`unique states: 1` (and an all-zero per-joint range) means the arm never moved:
re-record instead of keeping the episode. Recording the arm at rest still
produces a structurally valid dataset, so the contract checks alone cannot
catch it.

Do not mix this smoke episode into a production dataset. Use `--resume=true`
only for an existing dataset with the same robot, camera, task schema, and 30 Hz
contract.
