# 复现与自动化实验

## 一次实验的最小记录

1. 明确研究假设，固定数据清单、会话划分及主指标。
2. 准备数据并保存 `manifest.json` 与 `normalization.json`。
3. 复制训练 JSON，为每个独立训练种子创建不同配置、输出目录。
4. 用同一验证窗口与同一采样种子比较模型，记录全部试验，包括负结果。
5. 分别报告多训练种子变动和同一模型多采样种子变动；使用未参与模型选择的数据做最终泛化评估。

```bash
ego-action train --data data/prepared --output runs/seed47 --config configs/train.json
ego-action evaluate --data data/prepared --weights runs/seed47/best.safetensors \
  --output runs/seed47/evaluation-747.json --samples 8 --noise-seed 747
ego-action evaluate --data data/prepared --weights runs/seed47/best.safetensors \
  --output runs/seed47/evaluation-748.json --samples 8 --noise-seed 748
```

这些命令可由 Agent 执行。Agent 适合协助生成配置、部署消融、诊断失败、汇总 JSON；实验假设、数据划分、评分口径和结果解释仍需研究者审查。首版提供 CLI 和机器可读产物，尚不包含自主生成假设或自行修改算法的 Agent 服务。

## 训练产物

- `contract.json`：完整配置、数据清单哈希、Python 源码哈希、主要运行时版本。
- `training.jsonl`：逐更新 loss 与裁剪前梯度范数。
- `validation.jsonl`：固定验证面板上的完整评分。
- `best.safetensors` / `best.json`：按验证面板主指标选择的模型与步数。
- `resume.json` / `resume_*.safetensors` / `optimizer_*.safetensors`：交替写入两个槽位，JSON 是完成提交记录；包含步数、NumPy RNG 状态与优化器状态。
- `summary.json`：已完成步数、目标步数、参数量、是否为合成数据。

AdamW：β=(0.9,0.999)、eps=1e-8、weight decay=0.01、梯度范数上限 1。默认线性预热 200 更新，随后固定学习率。批次按固定 NumPy RNG 有放回采样。模型使用 MLX 随机初始化，不导入原项目的 Torch 初始权重。

`--resume` 严格核验代码、配置、数据清单及运行时版本，不接受静默改变实验条件。暂停应使用 `--stop-after N` 保持完整训练配置；意外终止后回到最近提交的检查点。输出日志中超出已提交检查点的行在恢复时移除。

## 评测口径

- 对每个窗口生成 K 条轨迹。主指标先对 K 次平方误差平均，再按全部有效手腕坐标加权开根号；不是 best-of-K。
- 对照是当前手腕位置重复至所有未来时刻。模型和对照使用相同的未来有效掩码，并要求相应当前坐标有效。
- 指尖/手腕三维位置、6D 旋转转 SO(3) 后的角误差、原生时间差下的速度误差作为诊断。退化的预测旋转按 180° 计。
- 会话 bootstrap 以会话为采样单位、保留其全部相关窗口，共 2000 次，给出模型/静止误差比的 95% 区间。静止总误差为零时不定义该比值；重采样中零分母项被排除。
- 数据中只有少数会话时，区间不能作为可靠的显著性证据。所有模型调参后的验证指标都属于开发集结果。

## 发布验证

基础 CI 在 Linux/Python 3.10 与 3.12 上运行数据管线测试、合成样例、格式检查与包构建。MLX 训练需 Apple Silicon，在本地另行验证。详见 [首版验证记录](validation.md)。
