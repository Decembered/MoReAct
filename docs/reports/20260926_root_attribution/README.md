# Root 位置误差归因（2026-09-26）

结论：本次对照中，主要差距来自 diffusion 自回归历史反馈后的漂移。
VAE 的 GT-history 重建误差约 3.3 mm，不能解释完整生成的约 25 cm 偏差。
这不是严格可加的因果误差分解，也不意味着 VAE 对生成历史完全鲁棒。

## 设置与资产

- 将 `configs/diffusion_geometry.yaml` 的 `root_position` 权重改为 30；
  这是后续实验配方，没有修改正在运行的训练或旧 checkpoint 的目标。
- 固定读取当时的 `diffusion_stage2_b512_20260926_084433/best.pt`，step **9750**。
- 正式保留快照：`outputs/root_attribution_20260926/checkpoint.pt`，约 309 MiB，
  用于精确复现，避免之后 best.pt 更新。此 checkpoint 尚未训练 root_position 项。
- SHA256：`ad52bc8f2a89258822e3233b01cc0809c4434f43f3657021acae2d61210dc65c`。
- 所有组使用该 checkpoint 内嵌的同一个冻结 VAE、EMA denoiser、归一化和 FK repair。
- 8 条 val 序列按 manifest 顺序等间隔选取，另加已有握手与拥抱 test 样例；
  两个集合分别汇总。每条最多 192 个预测帧、30 FPS、H=2/F=8。
- 随机组 seeds=0/1/2，确定性 VAE 均值组只算一次；guidance=1。
  等权平均 episode/seed，未按帧数加权。运行设备 cuda:6。
- 执行日期 2026-09-26；开始约 15:10 UTC。执行成功，完整逐帧结果见
  [results.json](evidence/results.json)，运行与测试输出在同目录。

## 对照结果

下表为 pelvis 的 root ADE，单位 cm。所有组比较同一 GT 帧；teacher 每个
primitive 重置为 GT reactor history，feedback 使用各自生成的历史。

| 分支 | 8 条 val | 握手 test | 拥抱 test |
| --- | ---: | ---: | ---: |
| VAE posterior mean + GT history | 0.331 | 0.204 | 0.358 |
| VAE posterior sample + GT history | 0.331 | 0.204 | 0.358 |
| VAE posterior mean + 自身重建历史 | 1.455 | 0.472 | 1.495 |
| Diffusion + GT history | 1.059 | 0.617 | 1.067 |
| Diffusion + 自身生成历史 | 25.358 | 30.872 | 42.227 |

VAE 三组 encoder 均读取 GT future，属于有答案的重建/诊断；第三组也不是可部署
生成方法。Diffusion 不读取 GT future。不能把 VAE 重建当成公平生成基线，或把
两行相减解释为某个模块独占的因果贡献。

val 的 diffusion feedback 水平 ADE 为 25.10 cm，高度 MAE 为 2.06 cm；末帧
root FDE 为 56.23 cm。第一段 ADE 为 0.80 cm，最后一段平均为 55.62 cm。
每段强制 GT history 后误差约 1 cm，表明长时历史分布偏移和误差累积是重点。
VAE 自身重建历史组也从 0.33 增至 1.46 cm，说明 VAE 并非完全没有反馈敏感性。

握手 diffusion feedback 的三个 seed ADE 为 28.98/29.80/33.83 cm；拥抱为
31.50/70.28/24.91 cm。拥抱存在明显随机种子波动，不能只根据一个 GIF 下结论。

这些是相对单条记录的轨迹误差，小样本结果不等于全测试集生成质量，也不能证明
root_position=30 一定有效。当前结果支持优先训练 diffusion 的 root 监督与
generated-history 稳定性，而非先重训 VAE。还未分别消融历史编码、采样器和 decoder
在 diffusion 实际生成历史上的分布外敏感性。

## 验证与复现

- `tests/test_losses.py tests/test_training_generation.py`：10 passed；验证权重 30
  的物理尺度、冻结 VAE 梯度路径、加权总损失与旧配置兼容。
- teacher/feedback 使用配对随机流；所有样本第一 primitive 的 root 误差完全相同。
- FK repair 的 translation 改变量为零。
- 用正式 `rollout_arrays` 重跑握手 seed 0，逐帧 root 误差最大差
  `1.49e-08 m`，验证诊断与正式生成路径一致。见 [checks.json](evidence/checks.json)。

```bash
/data/users/autovla/.envs/remogen-motion-only/bin/python scripts/diagnose_root_error.py \
  --checkpoint outputs/root_attribution_20260926/checkpoint.pt \
  --output outputs/root_attribution_20260926/reproduction.json \
  --device cuda:6 --limit 8 --frames 192 --seeds 0 1 2 \
  --episodes G001T000A001R005 G001T000A000R004
```

临时工作目录为 `tmp/root_attribution_20260926`。完成后把 checkpoint 转存正式
outputs，把 JSON/日志和脚本快照归档本报告；检查诊断与测试进程结束、无活跃
进程持有该目录后，精确删除此临时目录（含 pytest 合成资产与重复 checkpoint）。
不修改其他 runs/data/outputs。
