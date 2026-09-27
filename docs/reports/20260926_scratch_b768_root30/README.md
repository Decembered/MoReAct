# 从零训练 3 卡 diffusion（batch 768 + root_position=30）

2026-09-26。按用户要求：在 GPU 4–7 上开启**新的 diffusion 训练**，从零随机初始化，
扩大 batch，并把全部 train / val 损失上传 W&B。

一句话结论：新 run 已在 **GPU 4/6/7（GPU5 让给并行的语义 T5 训练）** 以
**全局 batch 768（256/rank）从零训练**，`root_position` 权重 30，val 为
40 类分层随机采样 768 窗口；W&B run
[6ec03ivy](https://wandb.ai/tangy2462-xiamen-university/moreact/runs/6ec03ivy)
已在线入队全部 train/val/val_rollout 项。

## 1. 运行的配方与依据

| 项 | 旧配方（b512 系列） | 新 run |
| --- | --- | --- |
| 数据/模型 | H=2 F=8，4 个 primitive 连续 | 同 |
| 初始化 | — | 从零（无 `--resume`），冻结 CVAE 用 `frozen_vae.pt` |
| 卡数 / 全局 batch | 2 或 4 卡 / 512 | **3 卡（4,6,7）/ 768**（local 256） |
| 阶段步数 | 3000 / 25000 / 25000 = 53000 | **2000 / 17000 / 17000 = 36000** |
| 样本曝光量 | 27.1M | **27.6M（+1.9%）** |
| 学习率 | 1e-4 起线性退火 | 1e-4 起线性退火到 step 36000 |
| `root_position` | 无（本次为新增项） | **30**（其余几何项沿用 `configs/diffusion_geometry.yaml`） |
| 验证 | 顺序或分层 | `stratified_action`，每 250 step，768 窗口 |
| 记录 | — | log 每 10 step，checkpoint 每 500 step，`--validate-at-start` |

- 阶段步数与 batch 一起缩放，使每个阶段的**剩余样本曝光量**不变；
  学习率按同一"每样本 LR 曲线"退火，不因 batch 变大而跳变。
- `batch_size` 属于 resume 时的受保护键，本次是从零训练所以不受该限制；
  若日后要续训本 run，需用 `--stage2` 才能再改 batch/阶段/学习率。
- 冻结 CVAE 取 `runs/diffusion_h2_f8_equal_roles_20260925_120415/frozen_vae.pt`
  （step 50000，sha256 `6093da86…591f`，已硬链接到新 run 目录）。它与现有
  checkpoint 内嵌的 VAE 在 **148/148 张量上逐位相同**，唯一差异是 `latent_scale`
  标量缓冲（由 diffusion run 起始估计，本 run 会重新估计）。

## 2. 预检（tmp/scratch_b768_preflight_20260926，已归档后删除）

真实 3 卡、local 256，`--rollout-override 1.0` 强制走最重的生成历史回填路径，2 个
优化步后执行一次完整验证：

| 检查 | 结果 |
| --- | --- |
| 训练损失 | step1 4.374，step2 3.527，全部项有限 |
| 验证 | val_loss 1.8651，val_rollout_loss 4.2802，val_root_position 0.00201，均有限 |
| 显存峰值 | 3 卡合计 **52 158 MiB ≈ 17.4 GB/卡**（24.5 GB 卡，余量约 7 GB） |
| 三 rank 一致性 | `latent_scale` 三卡同为 1.2877082824707031（`seed_all` 在 `estimate_scale` 之前，抽到同一批窗口） |
| 分层验证 | 40 类 × 19–20 = 768 窗口，无重复窗口，565 条序列 |
| 日志 | 0 error / 0 traceback，逐项无 NaN |

预检说明随机初始化下「生成历史验证」不会产生非有限值，因此正式 run 的
`--validate-at-start` 与周期性验证都可安全执行。

## 3. 正式 run 当前数值（截至 2026-09-26 17:4x UTC）

- 位置：`runs/diffusion_scratch_b768_root30_20260926`，launcher pid 2279616，
  3 个 rank 进程，约 **3.8 s/step**（512 batch 的 4 卡旧 run 为 3.3 s/step，
  折算每样本吞吐更高）。
- train：step 350，`rollout_probability` 0.000（step 2000 前为纯真实历史），
  lr 9.99e-5，loss 1.897（初始化）→ 0.227。
- val：step 0 → 250，val_loss 1.8618 → **0.8727**，val_rollout_loss
  4.2262 → **1.7723**，val_root_position 0.00205 → **0.000384**。
- 数字仅代表训练早期，且新目标含 `root_position` 项，**不能**与旧 run 的
  val 曲线直接比大小。

## 4. W&B 上传

- run：`6ec03ivy`（entity `tangy2462-xiamen-university`，project `moreact`），
  上传进程 `scripts/sync_diffusion_wandb.py`，`status.json` 为 `online`，
  `retries` 0，`last_enqueued_step` 跟随训练步数。
- 覆盖范围：`train` / `val` / `val_rollout` 三组，每组 12 项 raw + 12 项 weighted
  + total，共 25 个标量；`trainer/step` 为共同横轴。
- 服务端验证：step 0 的 `val`、`val_rollout` 两组各 25 个指标已确认落到云端
  （`wandb_verified.json`）。**本次执行期间 `api.wandb.ai` 出现网络故障**
  （curl 也失败，SDK 报 `Network error (ReadTimeout), entering retry loop`），
  训练侧的 `train` 组当时仍在 SDK 重试队列中。
- 数据不会丢失：记录已写入本地队列文件
  `wandb_sync/wandb/run-20260926_165824-6ec03ivy/`，网络恢复后由 SDK 续传；
  必要时可在 run 结束后手工补传：
  `wandb sync runs/diffusion_scratch_b768_root30_20260926/wandb_sync/wandb/run-20260926_165824-6ec03ivy`。
  本地 `metrics.jsonl` 始终是可重放的权威来源。

## 5. 同时段其他任务的处理

- **GPU6/7 的原 root30 run 已停**（`runs/diffusion_root30_b512_20260926`）：
  停止于 step 10865（最后一次验证 step 10800，val_loss 0.13994），
  checkpoint 与 best 资产全部保留，其 W&B run `uktz1gac` 已正常 `finished`
  （10865 步全部入队）。它被本 run 取代，原因见 `superseded.json`。
- **GPU0–3 的 baseline 未动**（`diffusion_stage2_b512_stratified_20260926`，
  无 `root_position`、batch 512），继续作为未加 root 监督的对照。
- **GPU5 的语义 T5 captioner 未动**（按用户选择与 diffusion 并行）。

## 6. 限制与注意

- 本次是 3 卡而不是 4 卡：GPU5 被语义训练占用 19.4 GB，只剩 4.8 GB，
  放不下一个 local-256 的 rank（实测约 17.4 GB/rank）。若要 4 卡，需要该训练让卡。
- 从零训练意味着与旧 run 不共享权重进度；`root_position=30` 同时进入目标，
  所以新旧曲线不可直接比较，后续评估应使用同一 `rollout` 协议。
- 分层验证窗口数为 768（`val_batches: 1` × local 256 × 3 rank），不是 root30 run 的
  1024；类内仍为 19–20 窗口，跨步换样本、同 step 可复现。
- `threads: 1` 下 GPU 利用率呈 0%/100% 交替（数据加载与计算串行），
  这是既有设置的已知现象，未改动。

## 复现

```bash
cd /data/autovla/projects/MoReAct
/data/users/autovla/.envs/remogen-motion-only/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 -m moreact.train_distributed \
  --config runs/diffusion_scratch_b768_root30_20260926/launch.yaml \
  --output runs/diffusion_scratch_b768_root30_20260926 \
  --vae runs/diffusion_scratch_b768_root30_20260926/frozen_vae.pt \
  --devices 4,6,7 --validate-at-start
nohup /data/users/autovla/.envs/remogen-motion-only/bin/python -u scripts/sync_diffusion_wandb.py \
  --run runs/diffusion_scratch_b768_root30_20260926 > runs/diffusion_scratch_b768_root30_20260926/wandb_sync.log 2>&1 &
```

证据见 [evidence/](evidence/)：`launch.yaml`（原样配置）、`launch.json`（启动记录与
hash）、`run_summary.json`（预检与运行数值）、`wandb_sync_status.json`、
`superseded_run_wandb_status.json`、`wandb_sync_tests.log`（16 项上传相关测试通过）。
