本文说明如何用 `litearm-vla` 在一台七轴 LiteArm、一个 LiteGrip 和一个相机上安全地验证并运行 LeRobot 训练检查点，供部署和实机联调人员使用。

# 部署 LeRobot 策略

## 1. 先确认部署契约

策略适配器是 `litearm_policy`，Python 类是 `LiteArmPolicyRobot`。它严格表示以下
硬件组合：

```text
一个 LiteArm + 一个 LiteGrip + 一个相机
```

LeRobot 0.4.4 看到的原始 feature 为：

```text
observation:
  joint1.pos ... joint7.pos   float，rad
  gripper.pos                 float，rad
  wrist                       uint8[480, 640, 3]

action:
  joint1.pos ... joint7.pos   float，rad
  gripper.pos                 float，rad
```

`gripper.pos` 是标定后的开口弧度（`position_mm / rad_to_mm`，闭合端为 0），与
机械臂关节同一单位。LeRobot 会按上述顺序组成 `observation.state[8]` 和
`action[8]`，并把 `wrist` 组成 `observation.images.wrist`。相机名字可配置，但必须
只有一个，且必须与模型的 image feature 名完全一致。

**不要**把 `LiteArmRobot` 的 7 维向量接口用于 8 维模型。**不要**把
`gripper.pos` 换算成毫米：早期数据集在同一个 feature 名下存的是毫米，LeRobot 只
校验名称和形状，不校验单位。

## 2. 理解实际控制链

```text
LeRobot policy (30 Hz)
  └─ 后处理动作跳变门禁
       ├─ LiteArm ServoLoop (250 Hz) → litearm-python → USB CDC → STM32
       └─ LiteGripController (200 Hz 保持流) → litegrip-python → SocketCAN → DM4310
```

策略线程只更新目标，不直接维持电机流。每个执行器只有一个持续写入者。相机通过
LeRobot 的 camera API 读取。

机械臂连续跟随位置目标。夹爪只使用控制器已有的两种离散模式：目标不超过标定行程
中点时执行 grasp，高于中点时执行 open。观测仍然连续上报实测开口弧度。grasp 使用的
`force_n` 是 SDK 的名义力矩换算，不是测力标定结果；这条路径不声称接触保持或滑移
恢复。

LeRobot 把模型输出恢复为物理单位后，部署程序会先用生成该动作的最新实测状态做
跳变检查。任一关节单步变化超过 `0.10 rad`，或夹爪单步变化超过 `0.05 rad`（在
出货标定 61.01 mm/rad 下约 3.1 mm），就会在 `send_action()` 之前拒绝整个动作。
程序不会把异常输出静默钳位成模型未请求的另一条轨迹。这是模型输出合理性门禁，
不代替执行器限位、碰撞检测和实体急停。

夹爪控制器的关键默认值：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `hold_hz` | `200 Hz` | 无新命令时的保持帧周期 |
| `speed_mm_s` | `50 mm/s` | open 动作速度 |
| `grasp_force_n` | `20 N` | grasp 的名义力设定 |
| `grasp_hold_s` | `0.2 s` | grasp 分片长度，便于尽快响应新命令 |
| `feedback_timeout_s` | `0.25 s` | CAN 新鲜反馈超时 |
| `max_temperature_c` | `70 °C` | MOS 与线圈中较热者的软件停止阈值 |

温度值是本适配器的保守软件停止阈值，不是对电机额定温度或机构热安全的认证。

## 3. 安装固定版本

工作目录：`litearm-vla` 仓库根目录。先安装它依赖的 `litearm-lerobot`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../litearm-lerobot
python -m pip install -e .
```

本仓固定使用 LeRobot `0.4.4`，并把 LiteGrip SDK 固定到仓库验证过的提交。不要把
适配器装进一个环境、LeRobot 装进另一个环境。

验证解释器与包来源：

```bash
python -c "import lerobot, litearm, litegrip, litearm_vla; print(lerobot.__version__); print(litearm_vla.__file__)"
```

## 4. 先做零硬件检查

下面的命令**不会**打开机械臂、CAN 或相机，因为没有 `--enable-motion`：

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json
```

检查项包括：

- `config.json`、preprocessor 和 postprocessor 是否齐全。
- 完整模型的 `model.safetensors`，或 LoRA/PEFT 的 `adapter_config.json` 与 `adapter_model.safetensors` 是否齐全。
- LoRA/PEFT 的 `base_model_name_or_path` 是否存在；部署机必须已缓存或能够取得该基础模型。
- processor 引用的 `.safetensors` 状态文件是否齐全。
- 模型输入是否是 8 维 state 和一个 `3×480×640` 的 `wrist` 图像。
- 模型输出是否是 8 维 action。
- 训练数据集的 state/action 名称和顺序是否为 `joint1.pos` 到 `joint7.pos`，最后为 `gripper.pos`。
- 训练数据集的相机名字、分辨率和帧率是否与运行参数一致。

默认从检查点的 `train_config.json` 读取训练数据集根目录，再打开
`meta/info.json`。若训练数据集已移动，显式指定：

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --dataset-info /absolute/path/to/dataset/meta/info.json \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json
```

看到 `Validation-only mode` 才表示静态契约检查通过。它不证明模型能完成任务，也不
证明任何硬件链路正常。

继续做一次完整的模型与 LoRA 加载检查，但仍不打开机械臂、夹爪或相机：

```bash
litearm-vla-run-policy \
  --checkpoint /absolute/path/to/pretrained_model \
  --dataset-info /absolute/path/to/dataset/meta/info.json \
  --task "pick the red cube" \
  --gripper-calibration /absolute/path/to/can0_calibration.json \
  --device cuda \
  --load-policy
```

看到 `Model-load validation finished` 才表示基础模型、LoRA adapter、归一化处理器
和目标计算设备均能加载。该检查仍不证明策略动作安全或任务能够完成。

## 5. 分关通过实机门槛

不要直接运行完整策略。依次确认：

1. 机械急停有效，机械臂有支撑，工作区无人，夹爪内没有易碎物。
2. `litearm-python` 能连接并只读到七轴新鲜反馈；核对软限位、故障位和温度。
3. `litegrip-python` 能在指定 CAN 通道加载**本机实测**标定，并只读到位置、错误码和温度。
4. 相机单独输出 `480×640×3` RGB 图像，名字为 `wrist`，实际帧率达到 30 Hz。
5. 分别对机械臂和夹爪做低速、小范围、无负载动作，确认真实位移方向和单位。
6. 零硬件检查再次通过后，才允许策略接管。

`--gripper-mount normal|reverse` 只声明安装方向并加载模板，不能替代当前机构的行程
实测。正式策略部署优先使用 `--gripper-calibration`。

## 6. 受限运行策略

以下命令会真实驱动机械臂和夹爪：

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

只有 `--enable-motion` 同时使能机械臂和夹爪。首轮保持 `10 s`、无负载、低风险
工位，并由操作员握住急停。需要指定机械臂串口时再加
`--arm-port /dev/serial/by-id/...`；不要写死会互换的 `/dev/ttyACM0`。

不要为了让被拒绝的检查点继续运行而盲目放大跳变阈值。先统计训练数据每帧
`abs(action - observation)` 的分布，再检查归一化、feature 顺序、FPS 和相机映射。
发生拒绝时，该次动作不会进入 `send_action()`。

运行过程中：

- 后处理动作与当前实测状态的差值超阈值时，整个动作在下发前被拒绝。
- 机械臂目标由原有软限位、速度限制和加速度限制约束。
- 夹爪目标钳在已标定行程内，并按中点映射为 open 或 grasp。
- 夹爪反馈超过 `0.25 s` 未刷新、错误码异常或温度达到配置阈值时，控制环报错停止。
- `send_action()` 返回钳位后的实际接受目标；调试日志记录该值，不把模型原始输出伪装成已执行动作。
- 任意异常或 `Ctrl+C` 都先停止机械臂策略伺服，再用当前实测关节位置执行一次零位移 `movej`，把机械臂交给固件的刚性持位模式；该路径不调用 `disable()`，机械臂保持使能，完成机械臂交接后才清理夹爪和相机。

这里的“停止”是受控的软件停机，不是 `litearm.Arm.emergency_stop()`。SDK 急停会进入
锁存故障并移除保持力矩，不能同时满足“急停后仍使能持位”。程序也不会在异常路径里
自动执行 `reset()` 后重新使能，因为这会在未排除故障时恢复动力。实体急停仍用于人员
和设备安全；`kill -9`、进程崩溃、USB 断线、断电或持位交接本身失败时，软件无法保证
机械臂不下落，因此实机首轮仍必须有机械支撑和操作员看守。

## 7. 代码方式接入

`LiteArmPolicyRobot` 只有策略模式，连接时必须同时使能机械臂和夹爪：

```python
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from litearm_vla import LiteArmPolicyRobot, LiteArmPolicyRobotConfig, LiteGripOptions

config = LiteArmPolicyRobotConfig(
    cameras={
        "wrist": OpenCVCameraConfig(
            index_or_path=Path("/dev/video0"),
            fps=30,
            width=640,
            height=480,
        )
    },
    enable_on_connect=True,
    gripper=LiteGripOptions(
        enabled=True,
        channel="can0",
        calibration_file="/absolute/path/to/can0_calibration.json",
    ),
)

with LiteArmPolicyRobot(config) as robot:
    observation = robot.get_observation()
    robot.send_action(action)  # action 是 8 维，gripper.pos 用弧度
```

不要在未完成第 5 节硬件门槛时构造这个配置。`connect()` 会立刻使能机械臂并启动
夹爪保持流。

## 8. 证据边界

离线测试能证明配置校验、feature 契约、生命周期、夹爪映射与钳位、超时和异常传播；
它不能证明 USB、SocketCAN、相机、真实位移、持物能力、控制周期或长时间稳定性。

当前 smoke 检查点若只训练了少量步骤，只能用于验证加载与数据流，不能宣称具备任务
能力。真机验收必须记录检查点、数据集、标定文件、固件版本、控制频率、故障位、温度、
真实位移和任务成功率。
