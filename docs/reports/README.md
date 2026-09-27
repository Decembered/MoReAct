# 实验与维护报告

手工报告：

- [2026-09-27 八卡完整 diffusion 续训启动与 batch 对照](20260927_eight_gpu_full_train/README.md)：四档全局 batch 实测、固定 root 验证选择起点、八卡正式长训及周期性 root 评估。

- [2026-09-27 固定训练序列 root 过拟合探针](20260927_root_overfit_probe/README.md)：现有损失配方下 300 次集中更新将单条训练序列 120 帧反馈 root ADE 从 22.85 cm 降至 0.78 cm，并记录验证样本退化。

- [2026-09-27 当前几何配方四卡长训](20260927_geometry_long_start/README.md)：暂停原 GPU 0–3 DDP，并从 A 配方 step 27500 启动独立的完整长训。

- [2026-09-27 120 帧 rollout root 漂移诊断与短训消融](20260927_rollout_root_reduction/README.md)：冻结 checkpoint 配对诊断、历史状态干预与 A/B/C 短训记录。

- [2026-09-27 漂移监督与历史反馈因果诊断](20260927_drift_causality/README.md)：冻结模型漂移注入、梯度分析、latent 优化与真实历史干预，区分追赶目标风险和实际纠偏不足。

- [2026-09-27 root loss 与 Root MAE / Global MPJPE 的对齐诊断](20260927_root_loss_misalignment/README.md)：root_position 的坐标系与物理尺度、40ep×120帧漂移分解（锚点 vs 局部）、梯度份额审计，以及受控漂移注入验证（损失对平移/朝向的定价、模型对 δ 零补偿、训练 δ 仅约 10 cm）与 metric-aligned loss 建议。

- [2026-09-27 三模型训练集/测试集抽样评测](20260927_three_model_sample/README.md)：GPU4/6/7 step8500、GPU0-3 step21000 与官方 ReMoGen，20 类、40 条配对 rollout 及统计限制。

- [2026-09-26 从零训练 3 卡 diffusion（batch 768 + root_position=30）](20260926_scratch_b768_root30/README.md)：GPU4/6/7 从零启动、阶段步数与学习率按样本曝光缩放、预检显存、W&B 全损失组上传及网络故障说明。

- [2026-09-26 全类别分层随机验证](20260926_stratified_validation/README.md)：40 类覆盖、train/val 隔离检查、按步数换样本及主训练迁移。

- [2026-09-26 Batch512 diffusion 损失审计](20260926_diffusion_loss_audit/README.md)：旧几何训练停止记录、11 项损失梯度、固定窗口趋势与 VAE 参考。

- [2026-09-26 Root 位置误差归因](20260926_root_attribution/README.md)：同一 checkpoint 的 VAE / diffusion / 历史反馈对照，root_position 配方权重 30。

- [2026-09-26 历史临时产物归档与清理](20260926_artifact_cleanup/README.md)
- [2026-09-26 共享 CVAE / RVQ 理解分支验证](20260926_shared_rvq/README.md)
- [2026-09-26 残差语义训练与 bridge 对比](20260926_semantics_training/README.md)

自动 smoke 报告使用 `smoke_<UTC日期时间>_<唯一ID>/README.md`，包含命令、退出码、证据和清理状态。
失败记录同样保留，不将未通过或中断标为通过。

报告目录仅保留总结和必要的小型证据；模型、数据和大量可视化按 [维护规则](../MAINTENANCE.md) 处理。

- [2026-09-26 MoReAct / ReMoGen 生成对照](20260926_remogen_generation_comparison/README.md)：两个测试样例、GIF、指标及 FPS 标注差异。

- [2026-09-27 续训 best 权重对比](20260927_best_checkpoint_comparison/README.md)：8750/13250/19750、固定20条测试动作及GIF。

- [2026-09-27 历史平移增强与 root 特征加权](20260927_history_augmentation/README.md)：小幅历史扰动、root 特征 5 倍、FK 权重 20、DM 1.5 m 及 16 项测试。

- [2026-09-27 Diffusion 审查与修复](20260927_diffusion_review/README.md)：噪声概率 25%、生成历史边界标签一致性、单卡 rollout 验证与学习率调度，30 项回归测试。
