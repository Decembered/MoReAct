# 全类别分层随机验证

2026-09-26。按用户要求核对 train/val 类别覆盖与序列隔离，并启用每次重新抽样。

## 数据核对

数据仍使用 `data/interx_h2_f8`。按完整 episode 的 `Axxx` 编码分层，保留原划分。
train 有 9110 条缓存序列，其中 9106 条满足当前 34 帧训练窗口；val 有 570 条，
全部满足窗口长度。两者都包含 A000–A039 共 **40 类**，val 每类 **8–22 条**。
完整 train/val episode ID 交集为零；检查不宣称 subject 身份互斥。
类别计数见 [split_audit.json](evidence/split_audit.json)。

## 采样协议

- `train.val_sampling: stratified_action`。每 250 step 验证一次，预算维持 512 窗口。
- 40 类均衡分配，每类 12 或 13 个；余下名额随机分配给 32 类。
- 每类先随机轮换序列，再在序列中随机选窗口；一轮用尽可选序列再重复轮换。
  同次验证不重复窗口，避免只按窗口抽样导致长序列权重过大。
- seed 为训练 seed+9001+step。不同验证步数换样本，同一步数可复现。
- 全局列表打乱后切给四个 rank，每卡 128 个，全局无重复。
  teacher-history 和 generated-history 验证使用同一组窗口。
- 每次保存 `validation_samples/step_*.json`，包含类别数、episode、窗口位置和 seed。
- 启动时检查 train/val episode 重叠及可用类别是否一致，失败时直接拒绝运行。
- 原 sequential 配置保持兼容。更改验证口径时，恢复模型/EMA/optimizer/RNG/步数，
  但清空旧 best 标量；新旧损失不混比。随机验证存在采样波动，单次 best 不等于
  全量验证集最优，需要后续更广评估。

## 检查与生效

11 项测试通过，包括全类别覆盖、名额均衡、无重复、rank 分片、跨步变化与同种子
复现、类别缺失/序列交集拒绝、既有训练恢复与分布式接触梯度回归。

真实 checkpoint step10000、原 batch512 配置在 GPU6 上以单 rank NCCL 执行两轮
实际 teacher/rollout 验证（未训练）：

| 验证 seed 对应 step | 类别数 | 每类窗口 | 独立窗口 | 不同序列 |
| --- | ---: | ---: | ---: | ---: |
| 10000 | 40 | 12–13 | 512 | 474 |
| 10250 | 40 | 12–13 | 512 | 477 |

两次抽样不同，全部有限结果在 [preflight_results.json](evidence/preflight_results.json)。
第一次 teacher/rollout loss 为 0.13827/0.42838；第二次为 0.13299/0.43365。
两次是同一模型不同样本，不表示训练改善。

主训练迁移到 `runs/diffusion_stage2_b512_stratified_20260926`，仍使用 GPU0–3。
从旧 run 的最近保存 checkpoint step10250 恢复；旧日志最后为10350，因此约100个
已记录更新会重跑。旧 run 与 best 资产保留。只改变验证采样配置；
`root_position=30` 没有加入当前训练，原 loss/学习率/课程/优化器均恢复。

首次 launcher 因 launch YAML 使用 JSON 科学计数法、被 YAML 解析为字符串而退出，
未执行训练更新。已使用 `yaml.safe_dump` 重写并检查配置 roundtrip 后重新启动。
迁移状态和生效进度见 [activation.json](evidence/activation.json)。

## 复现与清理

```bash
/data/users/autovla/.envs/remogen-motion-only/bin/python -m pytest \
  tests/test_validation_sampling.py tests/test_training_generation.py tests/test_distributed.py \
  -q -p no:cacheprovider --basetemp tmp/stratified_validation_recheck/pytest
```

真实预检调用 `DistributedTrainer(..., resume=step_010000.pt)`，配置仅增加
val_sampling 并将 device 设为 cuda:6，依次设置 step=10000/10250 调用 validate。
实际配置、测试日志、预检指标和抽样清单保留 evidence。
临时目录 `tmp/stratified_validation_20260926` 在进程完成与归档后精确删除，
包括 pytest 合成 checkpoint；正式训练目录和恢复 checkpoint 保留。
