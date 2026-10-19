# EGO Action Lab

**面向第一视角动作建模与机制研究的可复现实验工具。**

EGO Action Lab extracts an independent, reproducible action-modeling pipeline from an ongoing egocentric research project. It provides EgoDex preprocessing, a frozen visual encoder, a 131M MLX action model, resumable training and machine-readable evaluation.

研究问题是：**视觉表征中哪些信息真正有助于动作预测？这些收益来自外观、当前状态，还是时间变化？** 本仓库先提供可运行的实验基础，便于逐步开展机制分析和自动化对照实验。

## 当前提供什么

- **数据与标签**：EgoDex HDF5/视频配对读取，原生时间戳对齐，当前相机坐标系下的双手动作，部位级有效性掩码，会话级训练/验证隔离。
- **动作模型**：受 H-RDT 架构启发的 131M 参数 MLX 实现；冻结的 DINOv3 ViT-S/16 当前帧特征；残差 Flow Matching 与 5 步 Euler 采样。
- **复现流程**：配置文件、数据/权重校验值、训练日志、模型与优化器及随机数状态恢复。
- **自动评分**：多次采样误差、静止基线、分预测时距指标、姿态与速度指标，以及按会话重采样的置信区间。结果输出 JSON，可由 Agent 或脚本调度、汇总。
- **可运行示例与测试**：无真实数据的合成样例、CPU 数据测试、MLX 目标函数与恢复测试。

首版实现是 **48 维双手轨迹预测基线**。视觉历史、恒速度/状态历史基线和更换视觉表征的机制对照仍为后续工作，见 [研究路线](docs/research.md)。本项目不是 EgoDex 或 H-RDT 官方实现，也不把人体轨迹预测视为已经验证的机器人控制能力。

## 安装

Python **3.10–3.12**；推荐单独创建环境。

```bash
git clone https://github.com/LilacZhang/ego-action-lab.git
cd ego-action-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

CPU/Linux 可运行数据工具、合成样例和基础测试。**真实视觉特征提取与模型训练目前使用 Apple Silicon macOS + MLX**，尚未提供 CUDA/PyTorch 训练后端：

```bash
pip install -e '.[mlx]'
```

依赖用途、已验证环境与资源说明见 [依赖说明](docs/dependencies.md)。安装不会下载数据或模型权重；运行默认无远程实验追踪。

## 五分钟内验证数据流程

```bash
ego-action demo --output runs/toy
ego-action inspect --data runs/toy/prepared
pytest -q
```

该命令生成小型合成 HDF5/视频，走完整的数据准备流程。其图像特征是**仅用于流程测试的占位特征**，不能用于研究结论。

在 Apple Silicon 上继续验证真实 131M 模型训练和恢复：

```bash
ego-action train --data runs/toy/prepared --output runs/smoke \
  --config configs/smoke.json --stop-after 1
ego-action train --data runs/toy/prepared --output runs/smoke \
  --config configs/smoke.json --resume
ego-action evaluate --data runs/toy/prepared --weights runs/smoke/best.safetensors \
  --output runs/smoke/evaluation.json --samples 1
```

首次运行包括模型初始化、编译及检查点写入，所需时间取决于机器；这不是训练效果演示。

## 使用真实 EgoDex 数据

1. 从 [EgoDex 官方仓库](https://github.com/apple-aiml-research/ml-egodex) 按其条款获取 HDF5 和对应视频。
2. 自行获取 [timm 格式 DINOv3 ViT-S/16 权重](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)。默认编码器需要该仓库的 `model.safetensors`，不能直接使用键名不同的其他格式。参考修订：`3bf4720a82ec2066db88137180ff1f83a675cef0`。请遵守模型自己的使用条款。
3. 复制 `configs/episodes.example.json` 为 `configs/episodes.local.json`，填写数据路径及真实 `session_name`，按会话划分 train/validation。同一会话不得跨集合；官方 test 应保留给最终评估，本命令只处理显式提供的 train/validation。

```bash
ego-action prepare --manifest configs/episodes.local.json \
  --weights weights/dinov3_vits16_model.safetensors --output data/prepared
ego-action inspect --data data/prepared
ego-action train --data data/prepared --output runs/baseline --config configs/train.json
```

路径相对于清单文件所在目录解析。输出目录必须不存在，以防误覆盖。准备过程保留输入文件哈希、窗口索引、原生 PTS、归一化参数和特征映射；详见 [数据约定](docs/data.md)。

```bash
# 恢复时使用同一份配置；stop-after 是总更新步数，不改变实验配置。
ego-action train --data data/prepared --output runs/baseline \
  --config configs/train.json --resume

# 在全部验证窗口上评测；更换 noise-seed 可检查采样波动。
ego-action evaluate --data data/prepared --weights runs/baseline/best.safetensors \
  --output runs/baseline/evaluation.json --samples 8 --noise-seed 747
```

默认训练每次验证使用固定、均匀选取的最多 64 个验证窗口，并据此选最佳检查点；最终命令默认评测全部验证窗口。**验证集分数不是独立测试集泛化结果。**

## 指标与复现边界

主指标 `sample_rmse_m` 为有效手腕坐标上、多次采样平方误差平均后的 RMSE，单位米；`mean_rmse_m` 为多次采样平均轨迹的误差。静止基线重复当前状态。比值 `sample_to_hold_ratio < 1` 表示在相同有效坐标上优于静止基线；静止误差为零时比值返回 `null`。

- 当前图像是唯一视觉条件，预测阶段不读取未来图像或真实未来动作。
- 采样种子控制推理随机性，**不等同于独立训练种子**。可靠研究需独立训练、匹配预算与多会话评估。
- 历史项目的内部结果未作为本仓库可复现成绩发布。这里使用新的可移植数据清单、窗口选择和 MLX 初始化；不保证重现历史实验的逐位数值。
- 相同代码、数据、依赖和设备上可恢复优化器与 NumPy RNG；跨设备/系统版本不承诺逐位一致。MLX 模型本身不含运行时 dropout。
- 置信区间按会话聚合；会话数很少时只作诊断。原始标签是追踪结果，存在噪声和缺失。

详细说明：[复现流程](docs/reproducibility.md) · [数据约定](docs/data.md) · [研究路线](docs/research.md) · [第三方来源](THIRD_PARTY_NOTICES.md)

## 目录

```text
src/ego_action_lab/   数据、几何、视觉编码、动作模型、训练与评测
configs/             训练配置、快速测试配置、数据清单示例
tests/               数据边界、掩码、指标、优化器/RNG 恢复测试
docs/                依赖、数据规范、复现与研究路线
```

## 许可与引用

代码采用 [MPL-2.0](LICENSE)，H-RDT 架构来源与适配说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。数据和预训练权重不随仓库分发，适用各自条款。引用本工具可使用 [CITATION.cff](CITATION.cff)，并请引用实际使用的 EgoDex、H-RDT、DINOv3 工作。
