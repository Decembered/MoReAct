# 120 帧 rollout root 漂移：冻结诊断与短训消融

2026-09-27。目标、筛选阈值和 A/B/C 设计按本轮预设计划执行。训练中的 GPU 0–3 未迁移或中断。所有训练分支从不可变的 `step_027000.pt` 出发；其 SHA256 为
`dd4a77b5e2e4443c93fd3cd5081834beae6d0cd97d833db8212b5068657b687d`。

## 冻结权重诊断

固定 test 集 40 类各一条长度至少 122 帧的 episode，每条 seeds 0、1、2，
生成 120 帧。不同 checkpoint 使用相同起点、文本、actor 历史和逐段噪声种子；
教师历史基线每段重置为 GT。选择与逐样本数据在
`outputs/rollout_root_study_20260927/`，汇总见其中 `summary.json`。
置信区间按 episode 聚类，对同一 episode 的三个 seed 先求均值再 bootstrap。

| checkpoint | 完整反馈 root ADE | GT 历史 root ADE | root 对齐 MPJPE | 边界速度跳变 |
| --- | ---: | ---: | ---: | ---: |
| step 11000 | 30.85 cm | 1.08 cm | 10.47 cm | 0.127 m/s |
| step 19000 | 31.59 cm | 1.29 cm | 12.82 cm | 4.603 m/s |
| step 27000 | 30.07 cm | 1.51 cm | 13.60 cm | 7.403 m/s |

step 27000 相对 step 11000 的 root ADE 差值为 −0.79 cm，episode 配对
95% CI [−3.99, +2.44] cm；不能判定持续训练改善了 root 漂移。
同期 root 对齐 MPJPE 增加 3.13 cm，95% CI [+2.44, +3.79] cm；
边界速度跳变增加 7.28 m/s，95% CI [+6.33, +8.24] m/s。
第 1 段到第 15 段，step 27000 的平均段内 root ADE 从 1.18 cm
增长到 50.86 cm，髋部朝向误差从 2.3° 增长到 34.7°。
方向误差按髋部连线计算，和用于局部参考系的方向一致。

在 step 27000 的同一已生成轨迹上，于第 42 和 82 帧历史切点分别只读修改下一段的
历史输入：只校正终点位置，使下一段 root ADE 平均降低 22.57 cm，episode 配对
95% CI [17.20, 28.26] cm；只校正朝向平均变化 +0.18 cm，区间跨零；
保持漂移终点位姿而替换局部历史状态，平均降低 0.67 cm，区间跨零。
该干预仅做误差归因；正式生成没有 GT 历史输入。位置校正的收益是即时
世界坐标重对齐，不代表模型自身能识别或纠正累积偏移。

## 短训设计与状态

配置与 SHA 记录在 `runs/root_drift_short_20260927/`。每支从 step 27000
保留冻结 CVAE、归一化、数据划分和优化器状态，全球 batch 512，目标 step 28500，
每 500 步保存 checkpoint。A 用 25% 历史平移增强和当前新配方；B 在 A 上增加
相对各自历史终点的 pelvis 位移 Huber ×30 与 root 相对旋转矩阵 MSE ×10；
C 保持 A 的损失但把 primitives 从 4 扩为 8。

B 首次启动额外重复了一次 SMPL-X FK，GPU 4 在首步 OOM；该空运行保存在
`B_failed_oom_initial/`。复用既有 FK 后，B 已重新从原始 step 27000 启动，
首步通过。后续 val 选择、固定 test 评估和筛选结论待训练完成后填入，
不能以这里的冻结诊断代替短训结论。
自动续跑脚本 `scripts/complete_root_drift_study.py` 在 A/B 到达 28500 后训练 C、
逐个评估固定 val checkpoint、选择每支模型，并只对选中者运行固定 test。
完成后生成本目录的 `final_results.md`；执行状态在
`runs/root_drift_short_20260927/pipeline_status.jsonl`。
