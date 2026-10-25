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

可选的夹爪链路是独立的：`litearm_vla.gripper` → `litegrip-python` → SocketCAN。
它复用录制程序的安全监控，但不占用机械臂的 USB CDC 链路。

## 数据契约

- 七个机械臂关节。LiteGrip 夹爪可选，默认关闭。
- `observation.state[t]` 是测得的关节向量 `q[t]`；启用夹爪时还包含
  `gripper.pos`。所有分量统一使用弧度：`gripper.pos` 是标定后的开度角
  （`position_mm / rad_to_mm`，闭合限位为 0 rad）。
- `action[t]` 是下一拍测得的向量 `q[t+1]`，启用夹爪时也包含同样以弧度表示的
  `gripper.pos`。夹爪与关节遵循同一规则，因此夹爪动作在运动过程中是连续的。
- 在此单位规则之前录制的数据集把 `gripper.pos` 以毫米存放在同一个特征名下。
  不要对它们使用 `--resume`，也不要与新弧度数据集混用；请重录，或按夹爪标定的
  `rad_to_mm` 转换该列并重新生成数据集统计量。
- 策略输出的 `gripper.pos` 也是弧度；下发给 LiteGrip 前必须用
  `position_mm = position_rad * rad_to_mm` 换回毫米。
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

`litegrip` 尚未发布到 PyPI，因此 `litearm-vla` 从上游 Git commit 安装夹爪 SDK：

```bash
python -m pip install \
  "git+https://github.com/nexform-tech/litegrip-python.git@0d46d6d8c598c0caa1ca863748dffb0328d5b80e"
```

测试会伪造机械臂、相机和夹爪。它们不会打开串口、相机、CAN 或网络机器人设备。

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

### 夹爪 CAN 链路

LiteGrip 走 SocketCAN，不占用机械臂的 USB CDC。每次会话前把接口以 1 Mbit/s 拉起，
并保证本进程独占：

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

然后在不移动夹爪的前提下检查标定和使能路径：

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

如果 `load_calibration()` 返回 `False`，说明出厂标定文件对应的是别的通道（文件里
写死 `can0`），或本机标定缺失；非标准安装方向请传
`--robot.gripper.mount=reverse`，或通过 `--robot.gripper.calibration_file` 指定
文件。**不要**在占位限位上开始录制。

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
  --robot.payload_kg=0.62 \
  --robot.payload_com='[0, 0, 0.03]' \
  --robot.gripper.enabled=true \
  --robot.gripper.channel=can0 \
  --robot.cameras='{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=local/litearm-pick-place-v10 \
  --dataset.root=data/litearm-pick-place-v10 \
  --dataset.single_task='Pick up the block and place it on the other side.' \
  --dataset.fps=30 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=20 \
  --dataset.num_episodes=10 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --resume=true \
  --play_sounds=false
```

`auto_enable` 默认为 `false`。将其设为 `true` 即明确请求为机械臂上电。
`--robot.gripper.enabled=true` 则是夹爪的对应开关：它会在创建数据集之前打开 CAN
链路、加载标定并使能夹爪电机。

`--robot.payload_kg=0.62` 会在使能机械臂之前把末端负载质量写入固件的重力前馈，
然后用 `get_ff_scalar()` 读回 item 4 校验；读回值与请求值不一致就故障关闭。
`--robot.payload_com='[0, 0, 0.03]'` 把质心放在法兰 `+z` 方向 3 cm 处；
`set_payload()` 会同时写入这两个值。该写入只改 RAM；如需掉电保留，请自行调用
`Arm.save_params()`。

进程会独占
打开机械臂，校验最新反馈与固件限位，进入一次零重力模式，并在录制、复位、丢弃和
保存各阶段保持该模式有效。

终端用 `[PHASE: RECORDING]` 与 `data capture ON` 标记录制阶段，用
`[PHASE: RESET]` 与 `data capture OFF` 标记环境复位阶段。程序默认隐藏 LiteGrip
常规 `INFO`，但保留所有告警和错误；诊断夹爪 SDK 时可加
`--verbose_gripper_logs=true`。

- 右方向键：提前结束当前录制或复位阶段。
- 左方向键：丢弃当前 episode 并重录。
- Esc：结束会话、finalize 数据集、退出零重力、保持实测姿态，然后关闭 USB CDC。
- `o`：张开夹爪（`--robot.gripper.open_key`）。
- `c`：闭合夹爪并用 `grasp()` 保持夹持力（`--robot.gripper.close_key`，力由
  `--robot.gripper.grasp_force_n` 指定）。

正常退出会停止零重力、校验新的安全状态、执行**零位移 `movej` 交接**、再关闭
链路；它**不会失能**，所以机械臂会保持使能并停在实测姿态。发生安全故障时走同一条
路径：停止零重力并由固件保持实测姿态。本程序**不会**触发急停或失能机械臂；物理
急停仍然是唯一的应急手段。

夹爪遵循 SDK 的生命周期：移动之间由后台线程持续发送零位移保持帧；退出时先发零力矩
帧，再关闭 CAN 会话（这会失能电机）。因此**夹爪会在进程退出时松开**，不要指望它
在会话结束后继续夹住工件。键盘控制需要图形会话；启用
`--robot.gripper.enabled=true` 时，无显示环境会在打开任何设备之前直接报错退出。

有关分阶段硬件准入与数据集检查，请参阅[录制指南](docs/kinesthetic-recording.zh-CN.md)。
数据集准备完成后，请按照 [SmolVLA 训练教程](docs/smolvla-training.zh-CN.md)先完成
离线字段检查和 10-step 冒烟训练。当前 `smoke-v3` 数据只用于验证训练链路。

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

**夹爪使能失败，报 `[Errno 100] Network is down`。** SocketCAN 接口存在但处于
管理性 DOWN 状态。`LiteGrip.connect()` 能在 DOWN 接口上打开原始套接字，但第一次
发送就失败，于是 `enable()` 重试 3 次后返回 `err=0`。先把接口拉起：

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

必须看到 `state UP` 和 `can state ERROR-ACTIVE`。如果接口再次掉线，用 `dmesg`
检查 USB CAN 适配器（`gs_usb`）、线缆和 120 Ω 总线终端电阻。

**夹爪没有使用本地标定。** 无参 `load_calibration()` 只读 SDK 自带的出厂标定；
SDK 刻意不会自动拾取 `~/.litegrip/can0_08_calibration.json`。请显式传入：

```bash
--robot.gripper.calibration_file=/home/yd/.litegrip/can0_08_calibration.json
```

或在会话中设置 `LITEGRIP_CALIB`。只有在该单元确实就是出厂正装配置时才用出厂标定。

**episode 是静止的。** 数据集检查里的 `unique rows: 1` 表示整个录制期间机械臂根本
没被拖动过。零重力下很容易拖动，但它不会自己动。

**`lerobot-edit-dataset` 改不了本地数据集。** 不传 `--root` 时它只查
`~/.cache/huggingface/lerobot/<repo_id>`，然后转向 Hub，因此
`RepositoryNotFoundError` 不是认证问题。传 `--root` 后，LeRobot 0.4.4 对本地路径的
处理前后不一致：删除把该路径既当数据集目录又当父目录，合并则把同一个 root 发给所有
源数据集。请直接调用 `delete_episodes()` 或 `merge_datasets()` 并显式指定路径，见
[录制指南第 6 节](docs/kinesthetic-recording.zh-CN.md#6-删除失败的-episode)和
[第 7 节](docs/kinesthetic-recording.zh-CN.md#7-合并数据集)。

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
