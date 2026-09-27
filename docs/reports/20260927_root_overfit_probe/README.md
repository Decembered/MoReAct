# 固定训练序列的 root 过拟合探针

2026-09-27。目的：检验现有扩散模型与损失能否把一条训练集动作的 **120 帧自身历史生成** 拟合到厘米以下，以及该做法是否改善未训练序列。它是单样本能力测试，不是新配方的通用质量验证。

## 方法

- 起点：`runs/diffusion_stage2_b512_stratified_20260926/step_027000.pt`，SHA256 `dd4a77b5e2e4443c93fd3cd5081834beae6d0cd97d833db8212b5068657b687d`。仅更新 denoiser；CVAE、归一化、数据 digest、架构与推理路径固定。沿用该 checkpoint 的几何损失权重，**没有加入 `root_position`**。
- 固定训练 episode `G001T000A000R000`，12 个 4-primitive 窗口，起点 0、8、…、88，覆盖正式评估的 120 个未来帧。每次从这 12 个窗口随机选 2 个；共 300 次更新，AdamW 从 checkpoint 恢复优化器状态，恒定 LR `1e-4`，EMA `0.95`。训练时 `rollout_probability=1`，每个窗口内的后 3 段使用自身生成历史。历史随机平移概率设为 0，`feature_root_weight=5`、GT 近距离阈值 1.5 m、边界 delta 重算开启。随机采样扩散时间、噪声、caption 与 dropout 仍存在。脚本及完整配置见 [证据](evidence/)。
- 同一 episode 用正式 `ReactionGenerator.step()` 每 8 帧反馈生成 120 帧，并以相同种子 0、1、2 比较起点、100 更新、300 更新。`teacher` 每段重置为 GT 历史，仅作诊断；`feedback` 无 GT future 输入。额外挑选一条未参与更新的训练 episode 和一条 val episode，各用同样的 3 个种子复测。

## 结果

单位 cm，3 个种子的均值。root ADE 是世界系骨盆逐帧欧氏距离均值；root 对齐 MPJPE 则先逐帧去掉骨盆位置。

| 模型 | 固定训练序列 feedback root ADE | teacher root ADE | feedback root FDE | feedback root 对齐 MPJPE |
| --- | ---: | ---: | ---: | ---: |
| 起点 step 27000 | 22.85 | 0.71 | 23.30 | 11.39 |
| 100 更新 | 11.65 | 2.28 | 7.99 | 4.87 |
| **300 更新** | **0.78** | **0.29** | **2.46** | **0.53** |

训练记录中的随机 batch loss 从第 1 次 `0.5295` 降到第 300 次 `0.0585`；由于批次、噪声与 caption 变化，这两个值不是严格配对指标。固定训练序列的反馈边界速度跳变从 `5.73` 降至 `0.10 m/s`。100 次更新时 teacher 指标暂时恶化，说明只看短训中途点会误判最终过拟合能力。

| 未训练 episode，feedback root ADE | 起点 | 300 更新 |
| --- | ---: | ---: |
| 另一条 train：`G001T000A000R001` | 20.31 cm | 18.58 cm |
| val：`G002T001A006R009` | 9.25 cm | **61.26 cm** |

这个结果证明：**在相同的现有损失下，通过固定覆盖整段动作的窗口、全概率历史反馈和 300 次集中更新，模型可以把一条训练序列的 120 帧 root ADE 压到 0.78 cm**。它也清楚显示单样本记忆会破坏泛化；该 checkpoint 不应替代通用生成模型。上述两条未训练序列只是故障探针，不代表全训练集或验证集平均性能。A/B/C 的 40 类配对短训评估在另一项实验进行，结果须独立判读。

## 自回归动作可视化

按原评估的逐段随机数规则重新生成 3 个 seed，root ADE 分别为 0.934、0.901、0.514 cm；三者均值 `0.782949 cm` 与归档评估在浮点精度内一致。每段只用**真实 actor 的已观测历史**和**reactor 自身生成的历史**；reactor 仅最初 2 帧来自 GT，后续 120 帧没有读取 GT reactor future。它是条件反应生成，不是双方都从零自主生成。

- [seed 0 并排视频](../../../outputs/root_overfit_probe_20260927/visual/seed0/comparison.mp4)：左侧蓝色 actor + 橙色自回归 reactor，右侧相同 actor + GT reactor；4 秒，30 FPS 数据以 15 FPS 抽帧播放。
- [动图](../../../outputs/root_overfit_probe_20260927/visual/seed0/comparison.gif)及[第 2 秒静帧](../../../outputs/root_overfit_probe_20260927/visual/seed0/comparison_still.png)。这是 22 关节骨架可视化，不是 SMPL-X 网格。
- 逐帧生成数据和每个 seed 的指标在 `outputs/root_overfit_probe_20260927/visual/`。复现脚本快照为 [make_rollout.py](evidence/make_rollout.py)，使用与原 `rollout_root_study.evaluate_one()` 相同的逐段 seed 规则。

## 资产与复现

- 300 更新 checkpoint：`outputs/root_overfit_probe_20260927/overfit_300.pt`，SHA256 `5dfafd9f2c72abcb4e22bc5521a570a9a6e4490a73e264d5139c6cd62c14b01a`，保留供逐帧复核。
- 小型证据：[run.py](evidence/run.py)、[result.json](evidence/result.json)、[holdout.json](evidence/holdout.json)、[losses.jsonl](evidence/losses.jsonl)、[selection.json](evidence/selection.json)。`run.py` 是一次性脚本快照；其路径常量要求它位于 `tmp/root_overfit_probe_20260927/run.py`。
- 在项目根目录复现：`mkdir -p tmp/root_overfit_probe_20260927 && cp docs/reports/20260927_root_overfit_probe/evidence/run.py tmp/root_overfit_probe_20260927/run.py && /data/users/autovla/.envs/remogen-motion-only/bin/python -u tmp/root_overfit_probe_20260927/run.py`。复现会重新读取源 checkpoint 并产生两个约 309 MiB 的临时 checkpoint；先选新的临时目录或移走旧文件，避免覆盖需要保留的实验。
- 运行中没有更改或停止正式长训和 A/B/C 任务。临时 `overfit_100.pt` 仅用于读取上述 100 更新指标，无后续依赖；`overfit_300.pt` 已硬链接转存正式输出。检查活跃进程和引用后，已精确删除整个临时目录；正式输出与源 checkpoint 保留。删除清单与大小见 [cleanup.json](evidence/cleanup.json)。
