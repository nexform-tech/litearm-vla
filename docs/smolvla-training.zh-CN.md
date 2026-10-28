本文指导已有 LeRobot 数据集的 LiteArm 开发者使用 LeRobot 0.4.4 微调 SmolVLA，
并在正式训练前验证数据、GPU 和动作契约。

# 使用 SmolVLA 训练 LiteArm 策略

本教程先使用仓库中的 `litearm-hand-guide-smoke-v3` 完成数据检查和 10-step
冒烟训练，再说明如何迁移到正式数据集。训练过程只读取数据集和写入模型文件，
不连接机械臂、相机或 CAN 设备。

完成后，你将得到一个包含模型权重、归一化统计和训练配置的 LeRobot policy
checkpoint。这个 checkpoint 仍需经过离线检查和受限真机评估，不能仅凭 training
loss 判定可安全控制机械臂。

## 1. 确认版本和工作流

本教程固定使用以下环境：

- Ubuntu 22.04 LTS。
- Python 3.10 虚拟环境。
- LeRobot 0.4.4。
- 预训练模型 `lerobot/smolvla_base`。
- LeRobot Dataset v3.0。

推荐微调预训练模型，不要从零训练：

```text
LeRobot 数据集
  -> 任务文本、当前图像和当前状态
  -> 未来 50 帧动作序列
  -> SmolVLA 预处理和归一化
  -> 预训练 SmolVLA + action expert
  -> checkpoint
```

SmolVLA 默认使用一个当前观测和 50 帧动作块。对于 30 Hz 数据，动作块覆盖约
`50 / 30 = 1.67 s`。`n_action_steps` 只控制推理时连续取用多少个已预测动作，不会
改变训练标签的 50 帧长度。

官方建议从约 50 个高质量 episode 开始，并让每种物体位置、姿态和其他任务变化
重复出现。50 不是硬性门槛，但少量、静止或重复数据通常无法训练出可用策略。

## 2. 检查 Python 和 GPU

从仓库根目录激活已有环境。不要把依赖安装到系统 Python：

```bash
cd /home/yd/wl/VLA/litearm-vla
conda activate litearm-lerobot
python -m pip install "lerobot[smolvla]==0.4.4"
```

确认命令使用的是虚拟环境中的解释器和 LeRobot 0.4.4：

```bash
which python
python - <<'PY'
from importlib.metadata import version
import sys

print("python:", sys.executable)
print("lerobot:", version("lerobot"))
for package in ("torch", "transformers", "accelerate", "num2words"):
    print(f"{package}:", version(package))
PY
```

检查 CUDA：

```bash
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`CUDA available` 必须为 `True`。如果为 `False`，先修复 NVIDIA 驱动、CUDA runtime
或容器的 GPU 映射。不要退回 CPU 正式训练；它虽然可能启动，但通常不具备实用速度。

同时为模型缓存和 checkpoint 预留空间：

```bash
df -h /home/yd
du -sh /home/yd/.cache/huggingface
```

建议至少预留数十 GB。基础模型、VLM、优化器状态和多个 checkpoint 会同时占用磁盘。

## 3. 检查数据集契约

本仓库当前用于训练链路验证的数据集位于：

```text
/home/yd/wl/VLA/litearm-vla/data/litearm-hand-guide-smoke-v3
```

它使用以下本地逻辑 ID：

```text
local/litearm-hand-guide-smoke-v3
```

`dataset.repo_id` 是数据集的逻辑标识，`dataset.root` 才是本地文件目录。使用本地数据时，
两者必须同时传入。如果本地目录错误、缺少元数据或加载失败，LeRobot 可能把
`local/...` 当作 Hugging Face Hub 仓库名继续查询，最终报 `Repository Not Found`。
这不表示本地数据集需要登录 Hugging Face。

先执行只读检查：

```bash
cd /home/yd/wl/VLA/litearm-vla
conda activate litearm-lerobot

DATASET_ROOT=/home/yd/wl/VLA/litearm-vla/data/litearm-hand-guide-smoke-v3
DATASET_REPO_ID=local/litearm-hand-guide-smoke-v3

python - "$DATASET_REPO_ID" "$DATASET_ROOT" <<'PY'
import sys
from pathlib import Path

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

repo_id, root = sys.argv[1:]
root = Path(root).resolve()
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")

meta = LeRobotDatasetMetadata(repo_id, root=root)
config = SmolVLAConfig(
    input_features=None,
    output_features=None,
    device="cpu",
    push_to_hub=False,
)
delta_timestamps = resolve_delta_timestamps(config, meta)
dataset = LeRobotDataset(
    repo_id,
    root=root,
    delta_timestamps=delta_timestamps,
    video_backend="torchcodec",
)

sample = dataset[len(dataset) // 2]
print("episodes:", dataset.num_episodes)
print("frames:", dataset.num_frames)
print("fps:", dataset.fps)
print("camera keys:", dataset.meta.camera_keys)
print("state shape:", tuple(sample["observation.state"].shape))
print("action shape:", tuple(sample["action"].shape))
print("task:", sample["task"])
PY
```

该数据集的实测输出是：

```text
episodes: 3
frames: 897
fps: 30
camera keys: ['observation.images.wrist']
state shape: (1, 8)
action shape: (50, 8)
task: 抓取方块并放在另一边
```

这 3 个 episode 只适合验证训练链路，不足以证明能够训练出可用策略。正式训练前应采集
更多高质量成功示范，并保留独立验证集。

开始训练前必须满足：

- 至少存在一个 RGB 图像字段 `observation.images.*`。
- `observation.state` 和 `action` 是有限的 `float32` 数值。
- 所有 episode 的状态维度、动作维度、字段名和字段顺序一致。
- `task` 非空，同一任务使用一致、明确的描述。
- 图像能够由 `torchcodec` 解码。
- 状态和动作统计不是大量零标准差或异常尖峰。
- episode 包含完整、成功的任务过程，而不是仅有夹爪动作或静止画面。

SmolVLA 模型卡标注的主要语言是 English。中文任务文本可以被 tokenizer 处理，但在
数据量有限时，优先使用一致的英文指令，并在推理时使用相同语义。

## 4. 决定七维或八维动作契约

训练前必须决定最终 policy 是否控制夹爪。

七维模型使用：

```text
observation.state = 7 joints
action = 7 joints
```

八维模型使用：

```text
observation.state = 7 joints (rad) + gripper.pos (rad)
action = 7 joints (rad) + gripper.pos (rad)
```

八个分量统一使用弧度。`gripper.pos` 是标定后的夹爪开度角
（`position_mm / rad_to_mm`，闭合限位为 0），不是毫米。

注意：本仓库 `data/` 下已有的数据集（例如 `smoke-v3`、`pick-place-v3`）是在改单位
之前录制的，第八维仍是毫米。它们不能与新的弧度数据集混用，也不能用 `--resume`
续录；请重录，或先按夹爪标定的 `rad_to_mm` 转换该列并重新生成数据集统计量。

当前 `litearm-vla` 录制器能够生成八维数据，但当前 `litearm-lerobot` 的
`LiteArmRobot` 只声明七个关节动作。八维数据可以训练 SmolVLA，但不能直接通过这个
七维 adapter 完整部署。

如果模型需要控制夹爪，请先把“统一的 LiteArm + LiteGrip LeRobot Robot adapter”列为
部署前置任务。不要丢弃第八维后直接发送剩余七维，因为这样会让训练动作契约和实际执行
契约不一致。

## 5. 让 SmolVLA 使用数据集字段

`lerobot/smolvla_base` 的预训练配置自带六维状态、六维动作以及三个固定相机键。
LiteArm 数据通常使用七维或八维状态，以及 `observation.images.wrist`。

训练命令必须包含：

```bash
--policy.input_features=null \
--policy.output_features=null
```

这两个参数让 LeRobot 从数据集元数据推断输入和输出字段。省略后，模型可能继续查找
预训练配置中的相机键，并在字段或形状检查阶段失败。

LiteArm 的七维或八维状态和动作均低于 SmolVLA 的默认上限 32 维。模型会在内部补零，
不需要人为给数据集添加无意义维度。

## 6. 运行 10-step 冒烟训练

先用独立输出目录验证完整的数据加载、模型加载、前向传播、反向传播和日志链路：

```bash
cd /home/yd/wl/VLA/litearm-vla
conda activate litearm-lerobot

lerobot-train \
  --dataset.repo_id=local/litearm-pick-place-v10-edited \
  --dataset.root=data/litearm-pick-place-v10-edited \
  --dataset.video_backend=torchcodec \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_action_steps=10 \
  --batch_size=2 \
  --steps=10 \
  --num_workers=2 \
  --log_freq=1 \
  --save_checkpoint=true \
  --output_dir=/home/yd/vla-runs/litearm-smolvla-smoke-v3-save \
  --job_name=litearm_smolvla_smoke_v3 \
  --wandb.enable=false
```

冒烟训练通过必须同时满足：

- 日志显示正确的 episode、frame 和 effective batch size。
- 推断出的图像键、状态维度和动作维度与数据集一致。
- 10 个 step 全部完成。
- loss 和 gradient norm 是有限值，没有 `NaN` 或 `inf`。
- 没有 CUDA OOM、视频解码错误或字段缺失错误。
- 输出目录在启动前不存在；LeRobot 默认拒绝覆盖已有目录。

这一步只能证明训练链路可以运行，不代表模型已经学会任务。

## 7. 启动长程训练

下面的命令仍使用 `smoke-v3`，用于验证长程训练配置和 checkpoint 保存链路。该数据集只有
3 个 episode，直接运行 20k step 很可能过拟合，不能把结果当作可部署策略。真正的正式训练
应先把数据集替换为包含足够任务变化和成功示范的数据集，再同步修改 `dataset.repo_id`、
`dataset.root`、`output_dir` 和 `job_name`。

```bash
cd /home/yd/wl/VLA/litearm-vla
conda activate litearm-lerobot

lerobot-train \
  --dataset.repo_id=local/litearm-pick-place-v10-edited \
  --dataset.root=data/litearm-pick-place-v10-edited \
  --dataset.video_backend=torchcodec \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_action_steps=10 \
  --peft.method_type=LORA \
  --peft.r=16 \
  --batch_size=16 \
  --steps=20000 \
  --num_workers=4 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir=/home/yd/vla-runs/litearm-smolvla-v10-edited-29 \
  --job_name=litearm_smolvla_v10_edited_29 \
  --wandb.enable=false
```

从 `batch_size=4` 开始。出现 CUDA OOM 时依次降为 `2`、`1`，不要通过降低图像质量、
删除输入字段或修改动作维度来掩盖显存问题。

默认设置会冻结 vision encoder，并主要训练 action expert、状态投影和相关层。除非已经
建立可靠基线，不要同时修改 learning rate、冻结策略、动作块长度和图像尺寸；否则训练
失败时无法判断是哪项改动造成的。

如果显存不足或希望进一步缩小 checkpoint，可以改用 LoRA；但需要先理解它的适用范围和
LeRobot 0.4.4 的恢复训练限制，见下一节。

## 8. 可选：使用 LoRA 微调

LeRobot 0.4.4 原生支持 PEFT，LoRA 可用于降低优化器显存和 checkpoint 体积。需要注意：
SmolVLA 默认的 `train_expert_only=True` 加 `freeze_vision_encoder=True` 已经只训练
action expert 和投影层；LoRA 会进一步把 expert 的 q/v 投影冻结为低秩适配器，可训练
容量更小。因此 LoRA 不能替代更多数据，也不保证效果优于默认配方；它的主要收益是省
显存、缩小 checkpoint，以及减少对预训练权重的改动。

先安装 PEFT。不要把依赖装进系统 Python：

```bash
cd /home/yd/wl/VLA/litearm-vla
conda activate litearm-lerobot
python -m pip install "peft>=0.18.0,<1.0.0"
python -c "import peft; print(peft.__version__)"
```

在第 7 节的训练命令上增加两行即可启用 LoRA：

```bash
  --peft.method_type=LORA \
  --peft.r=16 \
```

不要手动传 `--policy.use_peft=true`。该字段由训练脚本在包装完成后自动置真；提前传入会
让模型在加载基座权重时误走 adapter 加载分支并失败。

SmolVLA 自带默认 LoRA 目标层，不需要手写：

```text
model.vlm_with_expert.lm_expert.<layer>.self_attn.{q,v}_proj
model.{state_proj,action_in_proj,action_out_proj,action_time_mlp_in,action_time_mlp_out}
```

需要覆盖默认值时可显式传 `--peft.target_modules`（字符串、列表或正则）。`--peft.r` 越大，
可训练容量越接近全量微调，显存和 adapter 也越大；没有建立基线时不要同时调整 `r`、
learning rate 和冻结策略。

已在本机 RTX 4060 8 GB 上使用 `smoke-v3` 完成 10-step LoRA 冒烟：可训练参数
742,656 / 450,788,832（约 0.165%），`batch_size=2` 时显存峰值约 2.1 GB，10 步全部完成，
loss 和 gradient norm 均为有限值。该结果只证明训练链路可以运行，不代表策略有效，也不
代表 `batch_size=4` 一定可用。

LoRA 训练的 `pretrained_model/` 结构与普通训练不同：

```text
checkpoints/000010/pretrained_model/
├── adapter_config.json       # LoRA 配置和基座模型路径
├── adapter_model.safetensors # 仅 adapter 权重（本机实测约 2.9 MB）
├── config.json               # policy 配置，use_peft=true
├── train_config.json
├── policy_preprocessor.json
└── policy_postprocessor.json
```

这里没有 `model.safetensors`。加载时 LeRobot 先读取 `adapter_config.json` 中的
`base_model_name_or_path`（这里是 `lerobot/smolvla_base`），再加载基座模型并套用 adapter，
因此基座模型必须在本地缓存或可访问。不要只复制 adapter 文件而遗漏处理器和归一化统计。

**在 LeRobot 0.4.4 上不要对 LoRA run 使用 `--resume=true`。** 该版本的恢复路径会先加载
已保存的 adapter（`is_trainable=False`），随后又对已包装的模型执行一次
`wrap_with_peft()`，从而新建一个从零初始化的 adapter。训练步数和优化器状态会继续，但
之前的 adapter 权重不会被继续训练，也不会被保存。日志中会出现：

```text
You are trying to modify a model with PEFT for a second time...
Already found a `peft_config` attribute in the model. This will lead to having multiple adapters...
```

实测恢复 2 步后，`lora_B` 中位范数从 0.032 降到 0.0002，说明 adapter 已被重置。LoRA
长程训练应在修复该问题前一次跑完；需要可靠恢复时，优先使用不启用 PEFT 的默认配方。

## 9. 检查 checkpoint 和恢复训练

正式训练会生成：

```text
/home/yd/vla-runs/litearm-smolvla-v3/
└── checkpoints/
    ├── 005000/
    ├── 010000/
    ├── 015000/
    ├── 020000/
    └── last -> 020000
```

每个 checkpoint 的 `pretrained_model/` 包含 policy config、模型权重、训练配置、
preprocessor 和 postprocessor。普通训练保存完整的 `model.safetensors`；LoRA 训练保存
`adapter_model.safetensors` 和 `adapter_config.json`，没有完整的 `model.safetensors`
（见第 8 节）。不要只复制权重文件；归一化统计和处理器也是推理契约的一部分。

从最后一个 checkpoint 恢复普通训练：

```bash
RUN_DIR=/home/yd/vla-runs/litearm-smolvla-v3

lerobot-train \
  --config_path="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true
```

`resume=true` 会恢复模型、优化器、scheduler、随机数状态和 step。不要把恢复训练误写成
重新使用 `--policy.path` 启动一个新 run；后者不会恢复优化器状态。

上述恢复流程只适用于未启用 PEFT 的训练。LeRobot 0.4.4 的 LoRA 恢复会静默重置 adapter
（见第 8 节），在修复前不要对 LoRA run 使用 `--resume=true`。

## 10. 可选地上传模型

本教程默认 `--policy.push_to_hub=false`，因此不需要 Hugging Face 写权限。如果需要上传，
先在本机登录，不要把 token 写入脚本、文档或 Git：

```bash
hf auth login
```

然后在正式训练命令中改为：

```bash
--policy.push_to_hub=true \
--policy.repo_id=your-hf-user/litearm-smolvla-v3
```

数据集也可以保持在本地；`policy.repo_id` 是模型仓库，不是数据集路径。

## 11. 评估训练结果

不要用 training loss 代替任务成功率。至少保留一组未参与训练的任务条件，并记录：

- 完整任务成功率。
- 抓取成功率和放置成功率。
- 不同初始物体位置的成功率。
- 超限动作、饱和动作和安全停止次数。
- 每轮推理延迟和实际动作发送频率。
- 相机掉帧、反馈超时和通信故障。

在真机执行前，先检查 checkpoint 的输入键和输出维度与部署 adapter 完全一致。真机阶段
必须从只读观测开始，再使用低速、低加速度、受限空间和较小的 `n_action_steps`。物理急停
必须在触手可及的位置。

**不要**用 `litearm_vla.kinesthetic.KinestheticRobot` 执行策略。它是只读录制 adapter，
`send_action()` 会故障关闭。策略执行必须使用经过验证的控制 adapter。

## 12. 常见失败

**出现 `Repository Not Found`，URL 中包含 `/api/datasets/local/.../refs`。** 这通常不是
认证问题，而是本地数据加载失败后，LeRobot 又把 `dataset.repo_id` 当成 Hub 仓库名查询。
确认命令同时包含正确的绝对路径
`--dataset.root=/home/yd/wl/VLA/litearm-vla/data/litearm-hand-guide-smoke-v3` 和逻辑 ID
`--dataset.repo_id=local/litearm-hand-guide-smoke-v3`，并检查
`data/litearm-hand-guide-smoke-v3/meta/info.json` 存在。不要为这个本地数据集执行
`hf auth login`。

**找不到 `camera1`、`camera2` 或 `camera3`。** 训练命令缺少
`--policy.input_features=null`，模型仍在使用预训练数据的相机键。

**要求 `policy.repo_id`。** `lerobot/smolvla_base` 的配置默认允许上传。离线训练请显式传
`--policy.push_to_hub=false`。

**CUDA OOM。** 先减小 `batch_size`，再减少 `num_workers`。不要改变动作维度或删除任务
关键相机来绕过显存不足。

**输出目录已经存在。** 为新 run 使用新目录；只有恢复 checkpoint 时使用
`--resume=true`。

**LoRA 恢复训练后效果回退。** LeRobot 0.4.4 在 `--resume=true` 时会二次包装 PEFT 模型，
新建一个从零初始化的 adapter；训练步数继续，但之前的 adapter 权重被丢弃。日志会出现
`modify a model with PEFT for a second time` 和 `multiple adapters` 警告。修复前不要
恢复 LoRA run，见第 8 节。

**loss 下降但真机不工作。** 优先检查数据是否存在静止 episode、失败示范、任务变化覆盖
不足、相机视角变化、动作单位错误，以及训练和部署的动作维度是否一致。

**训练完成但八维动作无法发送。** 当前控制 adapter 只接受七个关节。实现统一的机械臂和
夹爪 adapter，或重新训练明确的七维模型；不要静默截断夹爪维度。

## 13. 验证边界

截至本文编写时，已离线验证 LeRobot 0.4.4 能从上述绝对路径读取 3 个 episode、897 帧、
30 Hz 的单腕部相机数据，并把八维状态和八维动作组织为 SmolVLA 所需的 50 帧 action
chunk。该检查没有访问 Hugging Face Hub。

另外，已在本机 RTX 4060 8 GB 上完成 10-step LoRA 冒烟训练、adapter checkpoint 保存和
加载推理：可训练参数 742,656 / 450,788,832，`batch_size=2` 时显存峰值约 2.1 GB，
`predict_action_chunk` 返回 `(1, 50, 8)` 的有限值。恢复训练路径也在同一环境验证过，并
确认存在 adapter 被重置的问题（见第 8 节）。

以下项目不属于上述验证结果：

- 20k-step 训练结果。
- `batch_size=4` 的 LoRA 训练。
- checkpoint 的任务成功率。
- LoRA 恢复训练问题的修复验证。
- 八维机械臂和夹爪控制 adapter。
- 真机推理延迟、控制稳定性和安全性。

完成每一项后，应把实际命令、数据集版本、checkpoint、硬件条件和结果单独记录，不能用
“脚本启动成功”代替训练完成，也不能用“模型输出动作”代替真机执行成功。

## 参考资料

- [SmolVLA guide](https://huggingface.co/docs/lerobot/main/en/smolvla)
- [SmolVLA base model](https://huggingface.co/lerobot/smolvla_base)
- [LeRobot repository](https://github.com/huggingface/lerobot)
