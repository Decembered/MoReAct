# 当前几何配方的四卡长训启动记录

2026-09-27，按用户要求暂停旧的 GPU 0–3 四卡 diffusion 训练，并直接启动
更改权重后的当前配方长训。旧进程 PID 2252047 及其四个 rank 已停止；
`runs/diffusion_stage2_b512_stratified_20260926/step_028000.pt` 保留，可独立恢复。
GPU 1 的单卡 CVAE 和 GPU 3 的单卡 diffusion 是其他任务，没有停止。

新运行：`runs/diffusion_geometry_long_b512_20260927/`。从 A 配方已完成的
`runs/root_drift_short_20260927/A/step_027500.pt` 恢复模型、EMA、冻结 CVAE、
优化器和学习率进度，目标为 step 53000；四卡 GPU 0–3，全球 batch 512。
使用独立输出目录，配置见 `launch.yaml`，来源 checkpoint SHA256 见
`provenance_start.json`。当前配方含 25% 历史平移增强、未来首帧增量重算、
`smpl_joints_rec=20`、24 个 root 特征通道 ×5、1.5 m distance map，
不含绝对 `root_position` 或 B 分支的相对 root 附加损失。

启动 PID 2614014。起点验证与第一个更新 step 27501 已通过并写入
`metrics.jsonl`；后续 checkpoint、loss 和日志保存在该运行目录。
此长训是直接启动的 A 配方运行，不预先声称其 120 帧 root ADE 已优于
B/C；短训 A/B/C 的固定 val/test 筛选仍独立继续。
