[English](kinesthetic-recording.md) | 中文

本指南帮助 LiteArm 操作员安全地验证并运行基于 LeRobot 的直接 30 Hz 动觉采集。

# 直接手动引导录制

## 1. 确认软件准入

在 `litearm-lerobot` Conda 环境中，从 `/home/yd/wl/VLA/litearm-vla` 开始操作：

```bash
python -c 'import lerobot, litearm, litearm_lerobot, litegrip; print(lerobot.__version__, litearm.__version__, litegrip.__file__)'
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q -p no:cacheprovider -rs
```

预期版本为 LeRobot `0.4.4` 和 litearm-python `2.1.0`。`litegrip` 从 commit
`0d46d6d` 安装；适配器依赖它的状态新鲜度字段（`fresh` / `data_age_s`）。在进行
硬件操作之前，所有离线测试都必须通过。

## 2. 准备硬件

1. 停止 `litearm-vr-follow`、LiteArm Studio 以及所有可能打开机械臂的其他进程。
   录制程序必须是唯一的 CDC 占用者。
2. 固定机械臂底座，卸下负载，清空工作空间，并让第二个人守在物理急停旁。
3. 使用 SDK 的只读诊断功能确认全部七个关节都报告有限位置、无故障位、时间戳
   新鲜，且电机和线圈温度处于合理范围。
4. 确认所配置的相机正是预期的彩色节点。Intel RealSense D435i 的
   `index_or_path: 0` 是深度流（`Z16`），不是彩色流：

   ```bash
   for d in /sys/class/video4linux/video*; do
     echo "$(basename "$d"): $(cat "$d/device/interface")"
   done
   ```

   把 `RGB` 节点以设备路径传入。确认相机接在 USB 3 上（`speed` 应为 `5000`），
   并确认确实出帧：

   ```bash
   timeout 15 gst-launch-1.0 -q v4l2src device=/dev/video4 num-buffers=30 ! fakesink
   echo "exit=$?"
   ```

   `exit=0` 表示有帧。如果节点存在但收不到帧，拔插相机给它断电重启；软件层的
   USB reset 无法恢复。不要运行会自动杀掉相机占用进程的工具。

5. 如果启用夹爪，先拉起它独立的 CAN 链路，并在会话前确认标定行程：

   ```bash
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   python - <<'PY'
   from litegrip import LiteGrip

   with LiteGrip("can0") as gripper:
       assert gripper.load_calibration()
       assert gripper.enable().ok
       print("calibrated:", gripper.config.calibrated)
   PY
   ```

   `can0` 必须由本进程独占。`ip -details link show can0` 必须显示 `state UP` 和
   `can state ERROR-ACTIVE`；接口处于 DOWN 时仍能连接，但每次发送都会报
   `[Errno 100] Network is down`。标定返回 `False` 说明出厂文件对应的是别的通道，
   或该单元需要模板/标定文件；见 README。不要在占位限位上继续。无参加载只读 SDK
   自带的出厂文件——该单元有本机标定时请传 `--robot.gripper.calibration_file`。

6. 在启动录制之前，把机械臂放到固件限位以内。录制程序会在臂仍处于失能状态时，
   按固件限位内缩 0.01 rad 逐轴检查；软塌的臂会掉出范围——J4 的正向行程只有约
   0.0175 rad。请通过 SDK 回零并保持使能；`Arm.close()` 不会失能：

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

   已经使能的臂也可以：录制程序会跳过 `enable()` 直接进入零重力。

不要仅凭 ACK 就继续操作。准入门槛要求新鲜的物理反馈和已知的急停路径。

## 3. 单独验证零重力

在录制之前，使用 litearm-python 公开的 `Arm.zero_g()` API 进行一次专门的低风险
检查。保持行程小且慢。验证机械臂不会跳变、自走、过热，以及退出零重力时不会
坠落。

本仓库尚未自动执行该硬件检查。Python 测试无法证明真实的力补偿、摩擦、回差、
负载行为或固件的退出交接。

## 4. 录制一个冒烟 episode

使用 README 中的命令，并设置 `episode_time_s=8`、`num_episodes=1` 和
`push_to_hub=false`。仅当机械臂当前处于禁用状态时才需要
`--robot.auto_enable=true`；这是一个明确的上电操作。如果机械臂已经使能且无故障
（例如刚执行完 `home()`），录制程序会跳过使能、直接进入零重力。

命令启动后几秒内就会进入零重力。请提前告知在场人员、站到臂旁，并在 episode
期间**实际拖动机械臂**——零重力下它很容易被拖动，但它不会自己动。

启用 `--robot.gripper.enabled=true` 后，按 `o` 张开、按 `c` 闭合并保持夹持。
首次按键走到标定端点；新的按键会在一个切片（`grasp_hold_s`，默认 0.2 s）内打断
正在进行的夹取。夹爪在创建数据集之前完成连接，并与机械臂共用同一套安全监控。

终端在写入数据帧时显示 `[PHASE: RECORDING]`，在只复位环境、不写数据时显示
`[PHASE: RESET]`。程序默认隐藏 LiteGrip 常规 `INFO`，避免这些信息淹没阶段标记；
所有 `WARNING` 和 `ERROR` 仍会显示。只有诊断夹爪连接或运动细节时才传入
`--verbose_gripper_logs=true`。

`--robot.payload_kg=0.62 --robot.payload_com='[0, 0, 0.03]'` 会在使能前把末端负载
质量与质心（法兰 `+z` 方向 3 cm）写入固件的重力前馈，并通过读回校验质量。
`set_payload()` 会同时写两个值；不传 `payload_com` 时默认 `(0, 0, 0)` 会清除已有
偏移。该写入只改 RAM。

锁存故障会阻止运行。只要存在 `FAULT`、`POS_VIOL`、`FB_STALE` 或 `joint_fault`
中的任何一项，`robot.connect()` 就会在使能之前抛错。README 中的命令带有
`--robot.auto_clear_faults=true`，它会在重新执行安全检查之前，清除上一次故障关闭
留下的锁存；只在机械臂**失能**时执行，使能状态下绝不清除。同时仍要排除物理原因；
在使能状态下 `clear_faults()` 清不掉，因为命令看门狗会立刻重新触发。

README 中的命令带有 `--robot.ignored_flags='["WD_TRIPPED"]'`，它把瞬态的命令看门狗
触发记录为警告而不中断会话。去掉该参数即回到严格的故障关闭模式。无论是否加这个
参数，其他故障位、温度与限位检查以及主机侧保活健康度仍然是致命的。

只有在 CDC 连接、固件限位读取、失能姿态安全检查、使能后立即进入零重力、
零重力确认、夹爪连接（启用时）、安全监控启动以及相机连接全部成功之后，进程才会
创建数据集。
使能后必须立即交接给零重力保活，因为固件命令看门狗会在使能时开始计时。

LeRobot 保存并编码 episode 期间，安全监控保持活动。它检查反馈时效、使能状态、
零重力保活、故障位、温度以及固件推导出的限位。一旦违反，就会停止零重力、执行
零位移交接（机械臂保持使能）、关闭会话并报告原因。本程序不会触发急停或失能机械臂；
如果机械臂无法稳定保持姿态，请使用物理急停。

参考机上已观察到这一保存/编码阶段以 `WD_TRIPPED` 故障关闭，即使主机侧保活间隔
始终低于 44 ms。请把这类故障当作真实故障，而不是偶发，并在重试前阅读 README 的
故障排查一节。

## 5. 检查数据集

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/litearm-hand-guide-smoke-v2")
dataset = LeRobotDataset("local/litearm-hand-guide-smoke-v2", root=root)
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
if observation.shape[1] > 7:
    print("gripper opening range:", np.round(
        observation[:, 7].max() - observation[:, 7].min(), 2
    ), "rad")
PY
```

只有当数据集能重新打开、报告 30 Hz、包含有限的 `(N, 7)` 或 `(N, 8)` 状态与动作
数组，并且视频与物理任务相符时，才算通过。配对误差应接近零，因为每一列都满足
`action[t]` 等于 `observation.state[t+1]`，启用夹爪时包括 `gripper.pos`。
`gripper.pos` 是标定后的开度角（弧度，闭合限位为 0），因此整个状态向量单位一致。
夹爪动作在运动过程中是连续的，而不是下发的目标值。

配对误差必须**按 episode 分别检查**。每个 episode 的最后一帧在本 episode 内没有
未来动作，它的 `action` 就是该 episode 的最后一个实测状态。若对整个数据集直接用
`action[:-1]` 对比 `observation[1:]`，会把上一 episode 的结尾与下一 episode 的开头
配成一对，从而报出很大的误差——那不是数据契约违例。

`unique states: 1`（以及逐关节行程全为 0）说明机械臂根本没动过：应当重录而不是
保留这条 episode。静止录制同样会产出结构合法的数据集，所以只靠契约检查发现不了。

不要把这个冒烟 episode 混入生产数据集。仅当已存在数据集且其机器人、相机、任务
schema 和 30 Hz 契约都一致时，才使用 `--resume=true`。

## 6. 删除失败的 episode

第 5 节确认某条 episode 失败或静止后，删除它应当可复现。LeRobot 会重新编码保留的
视频并重写 episode 元数据，因此结果会写入一个新的数据集目录。

在 LeRobot 0.4.4 上，`lerobot-edit-dataset` 只能编辑默认缓存布局
（`~/.cache/huggingface/lerobot/<repo_id>`）里的数据集，改不了放在 `data/` 下的
本地数据集：

```bash
# 失败：没有本地 root 时，它先查 Hub 缓存，再查 Hub。
lerobot-edit-dataset \
    --repo_id local/litearm-pick-place-v5 \
    --operation.type delete_episodes \
    --operation.episode_indices "[3, 9]"
```

不传 `--root` 时，它在
`~/.cache/huggingface/lerobot/<repo_id>/meta/info.json` 找不到元数据，随后把
`repo_id` 当成 Hub 仓库查询并抛出 `RepositoryNotFoundError`。这不是认证问题，本地
数据集不需要登录 Hugging Face。

传 `--root=<数据集目录>` 确实能加载数据，但 `handle_delete_episodes` 又把同一路径
当成父目录：它把输出算成 `<root>/<repo_id>`，并把已加载数据集的 `root` 改成
`<root>_old`。删除阶段于是去读从未创建的 `<root>_old/videos/...`，以
`FileNotFoundError` 结束。这些数据集在 0.4.4 上无法通过这个 CLI 删除。

请直接调用数据集工具，并显式指定一个尚不存在的输出目录；源数据集保持不动：

```bash
cd /home/yd/wl/VLA/litearm-vla

python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

source = Path("data/litearm-pick-place-v9")
output = Path("data/litearm-pick-place-v9-edited")  # 不能已存在

dataset = LeRobotDataset("local/litearm-pick-place-v9", root=source)
edited = delete_episodes(
    dataset,
    episode_indices=[4, 5, 7, 9],
    output_dir=output,
    repo_id="local/litearm-pick-place-v9",
)
print("episodes:", edited.meta.total_episodes)
print("frames:", edited.meta.total_frames)
PY
```

`repo_id` 只是逻辑标识，`root` 才决定读写哪个目录。删除后保留下来的 episode 会重新
连续编号，请对新目录重跑第 5 节的检查，并在后续录制或训练中用 `--dataset.root`
指向该目录。

## 7. 合并数据集

`lerobot-edit-dataset` 的 merge 操作有同样的限制：`handle_merge` 把同一个
`--root` 传给每个源数据集，而 `LeRobotDataset` 把 `root` 当成每个数据集自己的目录。
一个 root 只能指向一个目录，因此这个 CLI 无法合并 `data/` 下互不相同的本地目录。
不传 `--root` 时它先查 Hub 缓存再查 Hub，同样以 `RepositoryNotFoundError` 结束。

请为每个源数据集分别指定 `root`，直接调用数据集工具。输出目录必须尚不存在，源数据集
保持不动：

```bash
cd /home/yd/wl/VLA/litearm-vla

python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

sources = [
    ("local/litearm-pick-place-v9-edited", Path("data/litearm-pick-place-v9-edited")),
    ("local/litearm-pick-place-v6", Path("data/litearm-pick-place-v6")),
]
datasets = [LeRobotDataset(repo_id, root=root) for repo_id, root in sources]

merged = merge_datasets(
    datasets,
    output_repo_id="local/litearm-pick-place-v10",
    output_dir=Path("data/litearm-pick-place-v10"),  # 不能已存在
)
print("episodes:", merged.meta.total_episodes)
print("frames:", merged.meta.total_frames)
PY
```

所有源数据集必须具有相同的 `robot_type`、`fps`、特征名和形状；`gripper.pos` 单位
不同的数据集不能合并。合并会复制数据和视频文件，并把 episode 从 0 开始重新连续编号，
因此请对新目录重跑第 5 节的检查，并在后续录制或训练中用 `--dataset.root` 指向它。
