# 八卡完整 diffusion 续训启动与 batch 对照

2026-09-27。用户要求八卡全力推进完整训练。本次目标是**跨 episode 的条件反应生成**，不使用仅记住一条序列的 `root_overfit_probe` 权重。A/B/C 短训的 40 类测试显示 B/C 的 root ADE 对 A 没有稳定收益，因此沿用 A 几何配方；现有 A 四卡长训在不可变 step 32500 保存后切换到八卡。

## 续训起点

两个候选 checkpoint 在同一组 40 类 val episode、每条 seed 0/1/2、120 帧完整反馈上比较：

| checkpoint | root ADE | root FDE | root 对齐 MPJPE | 边界速度跳变 |
| --- | ---: | ---: | ---: | ---: |
| step 30500 | 23.13 cm | 42.39 cm | 11.35 cm | 4.75 m/s |
| **step 32500** | **22.70 cm** | **41.47 cm** | **11.05 cm** | **4.37 m/s** |

32500 相对 30500 的 episode 配对 root ADE 差值为 −0.43 cm，95% bootstrap CI [−1.14,+0.29] cm，不能据此声称显著改善；但其余观察指标也未恶化，且保留额外训练进度，所以选 step 32500。源 SHA256 为 `6b31f4f0ca8d72834a19715eb8bd98961dec5346d27baa47192a9f0615ccd208`，固定 val 选择和结果见 [source_val.json](evidence/source_val.json)。未用 test 集筛选。

## 八卡 batch 预检

四档均从相同 step 32500 模型、EMA、优化器与配置启动，GPU 0–7，每档 8 次更新；第 2 次起使用完整生成历史反馈。下表取第 3–7 次稳定更新的中位数，排除启动、终点验证和 checkpoint I/O。

| 全局 batch | 每卡 | 秒/步 | 样本/秒 | 按剩余样本量估计的纯训练时间 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 64 | 1.312 | 390 | 7.47 h |
| **1024** | **128** | **2.264** | **452** | **6.45 h** |
| 1536 | 192 | 3.257 | 472 | 6.18 h |
| 2048 | 256 | 4.244 | 483 | 6.04 h |

选 **1024**：相对 512 吞吐增加约 16%；相对 2048 只少约 6% 吞吐，却保留约两倍的参数更新，并在每卡约 10 GB 的实际运行显存下留有余量。2048 预检时每卡约 17–18 GB，也通过，但短测速不能证明它的最终动作质量更好。因此“最佳”指本次**吞吐、更新次数和显存余量**的工程折中，不是已测得的泛化最优 batch。原计划剩余样本曝光量为 `10,496,000`，1024 对应 10,250 次更新，目标 step **42750**。详见 [benchmark.json](evidence/benchmark.json) 和 [plan.json](evidence/plan.json)。
四份完整预检配置 `evidence/batch_*.yaml` 已归档；`plan.json` 中的 `tmp/` 配置路径是运行时原始记录，临时目录现已清理，应使用归档副本复核。

## 正式运行

- 位置：`runs/diffusion_geometry_8gpu_b1024_20260927/`；GPU 0–7，八个 DDP rank，启动 launcher PID `2426542`。配置与命令见 [launch.yaml](evidence/launch.yaml)、[provenance_start.json](evidence/provenance_start.json)。直接从四卡 A 配方 step 32500 恢复模型、EMA、冻结 CVAE 和 AdamW 状态；loss 未变。
- `--stage2` 在新目录切换全局 batch，学习率从原 step 的 `3.93605e-5` 继续按剩余样本量线性退火。新阶段的第 1 次更新历史反馈概率为 0，此后为 1；其余 10,249 次都是完整的 4 primitive 反馈训练。起点验证使用 1024 个分层窗口，因 batch 改变已重置旧 best 分数并生成新的 `best.pt` / `best_rollout.pt`。
- 启动验证：step 32500 的 `val_loss=0.16580`、`val_rollout_loss=0.66282`；step 32501 与 32550 的训练损失分别为 `0.15521`、`0.50415`（前者无反馈，不能直接比较），均有限；step 32550 的反馈概率为 1，八卡显存约 9.8–10.0 GB/卡。见 [start_metrics.json](evidence/start_metrics.json)。后续指标、checkpoint 和 `train.log` 保存在正式运行目录。
- 独立 root 评估 watcher PID `2427407` 等待 step 35000、38000、41000、42750 的不可变 checkpoint；每次用固定 40 类 val episode × 3 seed × 120 帧生成，记录在 `outputs/diffusion_geometry_8gpu_b1024_20260927/root_val/`。此评估读取 checkpoint，不改变训练目标或随机数。监控脚本为 `scripts/watch_root_rollout.py`。
- 为检查**全量训练是否使训练样本越来越准确**，另建固定 40 类 train episode × 3 seed × 120 帧的配对评估，结果在 `outputs/diffusion_geometry_8gpu_b1024_20260927/root_train/`。train/val episode 不重叠；train A000 类固定包含单样本过拟合演示的 `G001T000A000R000`。起点 step 32500 的固定 train root ADE 为 24.49 cm，该单条 episode 为 14.19 cm；固定 val root ADE 为 22.70 cm。train watcher PID `2428440` 在同样的 step 35000、38000、41000、42750 评估，随后可对照各组相对自身起点的变化。正式八卡训练仍从全量窗口随机采样，监控只读权重，不改变优化过程。

## 早期 step 趋势：120 帧纯自回归生成

使用各 split 固定的 40 条 episode、每条 seed 0/1/2；每个 checkpoint 都使用同一批 episode 和随机种子。step 30500→32500 为先前四卡 batch 512 阶段；step 32500→33000 是当前八卡 batch 1024 续训的前 500 步。表中数值为 episode 先对三个 seed 求平均、再对 40 个 episode 求平均，单位 cm。

| step | train root ADE | val root ADE | train root FDE | val root FDE | train root 对齐 MPJPE | val root 对齐 MPJPE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30500 | 25.70 | 23.13 | 52.26 | 42.39 | 11.35 | 11.35 |
| **32500，续训起点** | **24.49** | **22.70** | **50.32** | **41.47** | **10.90** | **11.05** |
| **33000** | **24.04** | **22.45** | **49.50** | **39.89** | **10.76** | **11.02** |

对同一 split、同一 episode 配对计算，32500→33000 的 root ADE：train 下降 **0.45 cm**（40 条 episode bootstrap 95% CI 为下降 0.07–0.80 cm；29/40 条改善），val 下降 **0.26 cm**（CI 为下降 0.57 cm 至上升 0.02 cm；23/40 条改善）。单条过拟合演示 episode 在相同三 seed 上为 16.36→14.19→14.04 cm。当前只能说明 train 在前 500 步有小幅改善，val 的 ADE 变化尚不确定；不能据此认定全量训练已过拟合或已显著改善泛化。完整逐 episode 数据及各指标配对区间在 `outputs/diffusion_geometry_8gpu_b1024_20260927/root_trend_early.json`，后续 step 35000/38000/41000/42750 的固定评估已在等待 checkpoint。

该趋势已绘成 [PNG](../../../outputs/diffusion_geometry_8gpu_b1024_20260927/root_error_vs_step.png) 和 [SVG](../../../outputs/diffusion_geometry_8gpu_b1024_20260927/root_error_vs_step.svg)，绘图输入与数值见同名前缀的 JSON。新 checkpoint 的固定评估完成后，可用以下命令更新图；脚本只纳入 train/val 两边都完成的 step：

```bash
/data/users/autovla/.envs/remogen-motion-only/bin/python scripts/plot_root_rollout_trend.py \
  --root outputs/diffusion_geometry_8gpu_b1024_20260927 \
  --source-val docs/reports/20260927_eight_gpu_full_train/evidence/source_val_per_episode.jsonl \
  --output-prefix outputs/diffusion_geometry_8gpu_b1024_20260927/root_error_vs_step
```

### 补齐全部已保存的当前配方谱系 checkpoint

按同一固定 train/val 选择和三个 seed，补测 step 27000 的源权重、step 27500–32500 每 500 步保存的四卡权重，并纳入八卡续训的 step 33000、33500、34000，共 **15 个完整配对点**。其中 step 30500/32500 复用上表的原始逐 episode 结果；step 32500 为八卡续训起点。重绘的同名 PNG/SVG 已覆盖早期三点图，数值与原始文件路径归档于 [root_error_vs_step.json](evidence/root_error_vs_step.json)。以下为 root ADE，单位 cm：

| step | train | val | step | train | val | step | train | val |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27000 | 27.06 | 22.29 | 27500 | 26.28 | 22.41 | 28000 | 25.64 | 22.39 |
| 28500 | 25.66 | 22.28 | 29000 | 25.89 | 22.65 | 29500 | 25.99 | 22.58 |
| 30000 | 25.86 | 22.28 | 30500 | 25.70 | 23.13 | 31000 | 25.68 | 23.32 |
| 31500 | 25.41 | 22.72 | 32000 | 25.02 | 22.57 | 32500 | 24.49 | 22.70 |
| 33000 | 24.04 | 22.45 | 33500 | 23.72 | 22.37 | 34000 | **23.53** | **22.26** |

同一 episode 配对后，几何配方从 step 27500 到 34000 的 train root ADE 下降 **2.75 cm**（40 条 episode bootstrap 95% CI：下降 1.28–4.27 cm；28/40 条改善），val 下降 **0.15 cm**（CI：下降 1.57 cm 至上升 1.06 cm；18/40 条改善）。仅看八卡续训 32500→34000，train 下降 **0.96 cm**（CI：下降 0.26–1.62 cm），val 下降 **0.44 cm**（CI：下降 1.29 cm 至上升 0.27 cm）。配对细节见 [root_trend_full_stats.json](evidence/root_trend_full_stats.json)。因此固定训练序列的 root 轨迹持续变准，但验证序列的 root ADE 在 22–23 cm 左右波动，显示出训练样本专属收益的迹象；目前不能说长期自回归 root 泛化已有稳定改善。同期 val root 对齐 MPJPE 从 step 27000 的 12.20 cm 降到 34000 的 10.93 cm，局部姿态误差与 root 轨迹误差应分开看。train/val 使用不同 episode，绝对误差大小不宜直接比较，只比较各自相对本组起点的变化。

为了释放八卡，精确停止了原 GPU 0–3 四卡长训和 GPU 1/3 两条旧单卡任务；没有删除其 checkpoint。旧 CVAE 最后日志/已保存为 step 147500/140000，旧单卡 diffusion 为 159000/150000，四卡 A 长训为 32500/32500。两条单卡任务自上次保存后的更新无法恢复，详细 PID、命令与路径见 [preemption.json](evidence/preemption.json)。CVAE 的后续自动评估 watcher 已记录 `training_stopped_before_completion`，没有误以为训练完成。

当前结论是八卡训练已稳定进入反馈阶段，至 step 34000 的固定 train root ADE 继续下降；固定 val root ADE 尚未出现跨整个几何配方阶段的稳定下降。最终 val 与独立 test 精度仍待后续 checkpoint 验证，不能从吞吐预检或单样本过拟合推断。
已保存固定起点评估的[逐 episode 结果](evidence/source_val_per_episode.jsonl)和[八卡显存快照](evidence/gpu_start.txt)。预检的四份临时 checkpoint 和其他无依赖临时文件在检查进程与引用后精确清理；删除清单见 [cleanup.json](evidence/cleanup.json)。
