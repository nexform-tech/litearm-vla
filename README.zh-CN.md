[English](README.md) | 中文

本仓库面向使用 LeRobot 0.4.4 的开发者，用于采集用于策略训练的 LiteArm 示教数据。

# litearm-vla

首个支持的工作流是直接的单臂手动引导（动觉）采集。它不会启动、连接或导入
`litearm-vr-follow`。

```text
LeRobot 数据集/相机
          ↑
litearm-vla (30 Hz q[t] -> q[t+1])
          ↑
litearm-lerobot (Robot 生命周期与固件限位)
          ↑
litearm-python (USB CDC 与零重力保活)
```

## 数据契约

- 七个机械臂关节，首版不包含夹爪。
- `observation.state[t]` 是测得的关节向量 `q[t]`。
- `action[t]` 是下一拍测得的向量 `q[t+1]`。
- 每条 episode 的最后一个观测被丢弃，因为它没有对应的后续动作。
- 录制固定为 30 Hz，采样率不达标的 episode 会被丢弃。
- 相机、数据集特征、视频编码、episode 保存、续录、重录控制以及可选的 Hub
  上传均由 LeRobot 负责。

## 安装与测试

使用现有的项目环境；不要安装到系统 Python 中：

```bash
conda activate litearm-lerobot
cd /home/yd/wl/VLA/litearm-lerobot
python -m pip install -e .
cd /home/yd/wl/VLA/litearm-vla
python -m pip install -e .

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q -p no:cacheprovider -rs
```

测试会伪造机械臂和相机。它们不会打开串口、相机、CAN 或网络机器人设备。

## 硬件预检

录制程序是故障关闭（fail-closed）的，所以第一次录制前先把两件设备都验证好。

### 相机节点

`index_or_path` 必须指向**彩色**节点。Intel RealSense D435i 会暴露六个 V4L2
节点，其中 `0` 是深度流（`Z16`），不是彩色流：

```bash
for d in /sys/class/video4linux/video*; do
  echo "$(basename "$d"): $(cat "$d/device/interface")"
done
```

使用接口名以 `RGB` 结尾的那个节点（常见是 `/dev/video4`，但不要假设编号永不
变化），并且传设备路径而不是裸索引。确认相机接在 USB 3 上、且确实出帧：

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

相机的 `speed` 应为 `5000`（USB 3），取帧测试必须立即打印 `exit=0`。如果节点
存在但收不到帧，拔插相机给它断电重启；软件层的 USB reset 无法恢复。

### 机械臂起始姿态

使能之前，录制程序要求每个关节都落在**固件限位内缩 0.01 rad** 的范围内。失能
状态下的机械臂会因自重下垂——尤其 J4 的正向行程只有约 0.0175 rad——所以软塌
的姿态经常过不了这道门禁，录制会被拒绝启动。请通过 SDK 回零并**保持使能**：

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
arm.home(timeout=60)   # 固件以固定的 0.10 速度运动
arm.close()            # close() 只关链路，不会失能
```

`Arm.home()` 允许从贴着软限位的位置起步；保持使能后，录制程序会跳过
`enable()` 直接进入零重力。**不要用失能来"停放"机械臂**：它会下垂并掉出限位。

## 录制

在 `litearm-vr-follow`、LiteArm Studio 以及所有其他机械臂控制器停止之前，不要
运行此命令。请将物理急停保持在触手可及的位置。命令启动后几秒内机械臂就会进入
零重力。

```bash
cd /home/yd/wl/VLA/litearm-vla
litearm-vla-record \
  --robot.type=litearm_kinesthetic \
  --robot.id=armA-hand-guide \
  --robot.auto_enable=true \
  --robot.auto_clear_faults=true \
  --robot.ignored_flags='["WD_TRIPPED"]' \
  --robot.cameras='{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=local/litearm-hand-guide-smoke-v2 \
  --dataset.root=data/litearm-hand-guide-smoke-v2 \
  --dataset.single_task='抓取方块并放在另一边' \
  --dataset.fps=30 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=5 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --play_sounds=false
```

`auto_enable` 默认为 `false`。将其设为 `true` 即明确请求为机械臂上电。进程会独占
打开机械臂，校验最新反馈与固件限位，进入一次零重力模式，并在录制、复位、丢弃和
保存各阶段保持该模式有效。

- 右方向键：提前结束当前录制或复位阶段。
- 左方向键：丢弃当前 episode 并重录。
- Esc：结束会话、finalize 数据集、退出零重力、保持实测姿态，然后关闭 USB CDC。

正常退出会停止零重力、校验新的安全状态、执行**零位移 `movej` 交接**、再关闭
链路；它**不会失能**，所以机械臂会保持使能并停在实测姿态。发生安全故障时走同一条
路径：停止零重力并由固件保持实测姿态。本程序**不会**触发急停或失能机械臂；物理
急停仍然是唯一的应急手段。

有关分阶段硬件准入与数据集检查，请参阅[录制指南](docs/kinesthetic-recording.zh-CN.md)。

## 故障排查

**`WD_TRIPPED`。** 固件的 0.10 s 命令看门狗在两次零重力保活之间超时。使能但收不到
保活的臂（例如 LiteArm Studio 刚退出）会锁存该标志，而保活会在下一个周期重新下发
`ZERO_G`，所以这个触发是瞬态的。在命令里加上
`--robot.ignored_flags='["WD_TRIPPED"]'` 即可把它降级为警告日志、不再中断会话。
其他所有故障位、温度与限位检查、以及主机侧保活健康度（`zero_g_error` /
`zero_g_active`）仍然是致命的。

**在使能状态下 `clear_faults()` 清不掉**（看门狗会立刻重新触发）；先失能，再
`clear_faults()`，若模式仍停在 `EMERGENCY` 再 `reset()`。

**锁存故障会阻止录制。** `FAULT`、`POS_VIOL`、`FB_STALE`、`joint_fault` 都是真实
的安全信号，只要有任何一项存在，录制程序就拒绝启动。加上
`--robot.auto_clear_faults=true` 可以在启动时清除上一次故障关闭留下的锁存：它只在
机械臂**失能**时执行，会记录清除了什么，然后重新做完整的安全检查；使能状态下绝不
清除。如果物理原因仍在，故障会立即重新锁存。使能时反复出现不同关节的 `FB_STALE`，
指向 CAN 侧问题：检查电机线缆并重新给机械臂上电。

**episode 是静止的。** 数据集检查里的 `unique rows: 1` 表示整个录制期间机械臂根本
没被拖动过。零重力下很容易拖动，但它不会自己动。

**视频编码与看门狗。** 实测 LeRobot 默认的 `libsvtav1` 在编码 episode 时会让录制
进程停顿超过 100 ms。在参考机上也观察到使用 `libsvtav1` 和 `h264` 的会话都在保存
阶段故障关闭，而主机侧保活间隔始终低于 44 ms，因此根因仍在排查中；相机与机械臂之间
的 USB 总线竞争是疑点之一。`--dataset.vcodec=h264` 能缩短编码窗口，产生的视频也能
被 OpenCV 直接解码，但它本身并不能消除该故障。请把相机接到 USB 3 口（最好与机械臂
不在同一控制器上）。加上 `--robot.ignored_flags='["WD_TRIPPED"]'` 后，保存阶段的
该故障只会记录警告并继续；不加则会故障关闭。

## 安全边界

离线测试只能证明数据契约和故障关闭（fail-closed）的调用顺序。它们不能证明零重力
行为、物理保持能力、相机同步、USB 稳定性，也不能证明真实机械臂上的安全运行。请从
无负载、受约束的工作空间和一个短 episode 开始。

## 许可证

Copyright © 2026 NEXFORM ROBOTICS。基于
[Apache License 2.0](LICENSE) 授权。
