# LiteArm VLA development rules

This document defines the reproducible development and hardware-safety rules
for contributors working on LiteArm VLA data collection.

## Environment

- Reference OS: Ubuntu 22.04 LTS.
- Reference Python: 3.10 in a project virtual environment or Conda environment.
- Do not install project dependencies into the system Python environment.
- The validated local environment is
  `/home/yd/miniconda3/envs/litearm-lerobot` with `lerobot==0.4.4`,
  `litearm-python==2.1.0`, and an editable `litearm-lerobot` checkout.
- Runtime data belongs under a user-selected dataset root. Tests and temporary
  files belong under pytest temporary directories and must not enter Git.

## Architecture and ownership

- `litearm-python` owns the USB CDC connection and zero-gravity keepalive.
- `litearm-lerobot` supplies the LeRobot Robot adapter, firmware limits, and
  direct-arm lifecycle that this repository extends.
- LeRobot supplies camera adapters, `LeRobotDataset`, episode handling, video
  encoding, keyboard events, and Hub upload.
- The collection process is the only LiteArm hardware owner. Do not run
  `litearm-vr-follow`, LiteArm Studio, or another robot controller at the same
  time.
- The recorder supports a seven-joint arm and an optional keyboard-driven
  LiteGrip. Record the gripper action as the target actually commanded (open
  endpoint, closed stop, or a hold command), never as a value inferred from
  measured position or a sentinel.

## Data contract

- Record at a fixed 30 Hz.
- Store measured joint state as `observation.state[t]`.
- Store the next measured joint state as `action[t] = q[t+1]`.
- Drop the final observation of each episode because it has no future action.
- Reject and re-record an episode when timing cannot honestly represent 30 Hz.
- Keep one zero-gravity session active across recording, environment reset,
  discard, and episode save operations.

## Hardware safety

- Offline tests must fake `litearm.Arm` and cameras. They must never open
  `/dev/ttyACM*`, `/dev/video*`, CAN, or a network robot service.
- Automatic motor enable is opt-in and defaults to false.
- Before recording, require fresh finite feedback, seven joints, no fault bits,
  acceptable motor and coil temperatures, and positions inside firmware-derived
  limits with a configured margin.
- Monitor feedback freshness, zero-gravity keepalive, enable state, faults,
  temperature, and position limits for the entire session, including dataset
  save and video encoding.
- On a safety fault, stop zero-gravity and hand off to a zero-displacement
  hold. Do not emergency-stop or disable the arm from this program: the physical
  emergency stop is the emergency path. Never treat an ACK as evidence of safe
  physical behavior.
- Normal shutdown stops zero-gravity, verifies a fresh safe state, performs a
  zero-displacement `movej` handoff, and then closes the CDC link.
- Real-hardware testing requires a physical emergency stop and proceeds one
  gate at a time: connection and passive feedback, explicit enable,
  zero-gravity without recording, then one short episode.

## Development workflow

Create and activate the environment outside this repository, then install the
three local projects in editable mode as needed. Run offline validation from the
repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q -p no:cacheprovider
python -m compileall -q src tests
git diff --check
```

Use English for source, documentation, commits, and pull requests. Keep public
configuration fields typed and documented. Do not add hidden hardware actions,
global dependency changes, credentials, generated datasets, or recorded media
to the repository.
