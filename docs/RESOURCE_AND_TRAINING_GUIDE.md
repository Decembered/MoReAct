# MoReAct 资源资产、配套关系与训练使用总览

核对日期：2026-09-27 UTC。主项目绝对路径：`/data/autovla/projects/MoReAct`。
本文针对本机已有资产和实际实现，汇总接手项目、推理、训练、恢复及存储交接所需信息。
运行状态是核对时快照；后续步数、best 和进程状态以运行目录为准。

文档源文件：`/data/autovla/projects/MoReAct/docs/RESOURCE_AND_TRAINING_GUIDE.md`。
42_store 文档副本：`/mnt/42_store/autovla/MoReAct/documentation/RESOURCE_AND_TRAINING_GUIDE.md`。
**store 中存放的是文档和路径清单，当前代码、数据和模型仍位于下表原路径，未迁移。**

## 1. 项目能做什么

| 功能 | 当前状态 | 主要入口 |
| --- | --- | --- |
| 双人动作条件 CVAE 训练、真实未来重建 | 已实现 | `train_vae`、`evaluate` |
| Actor 历史条件下生成 Reactor 未来 | 已实现，连续 latent diffusion | `train_diffusion`、`rollout` |
| 多 GPU 训练、恢复、EMA、历史回填 | 已实现 | `moreact.train_distributed` |
| 完整双人动作 → 交互描述 | 已实现；本机一轮正式语义训练已完成 | `semantics describe/evaluate` |
| 在线生成 API | 已实现分段接口 | `ReactionGenerator.initialize/step` |
| 在线语义调度、自动将理解结果反馈给生成 | 尚未实现 | 没有可运行的 `react` / `ReactionSession` |

生成：双方历史 + 可选 CLIP 文本 → 去噪 latent → 冻结 CVAE decoder → Reactor 未来。
理解：完整双方动作 → 冻结 CVAE encoder 的确定性 mu → K=2 RVQ → T5 → 描述。
生成不经过 RVQ；当前理解输出也不作为生成条件。

## 2. 资源资产绝对路径

下列路径核对时存在。`RESOURCE_ASSETS.json` 是配套机器可读清单，记录绝对路径、解析后路径、用途及存在状态。

### 2.1 必需输入与运行环境

| 资源 | 绝对路径 | 用法 / 配对约束 |
| --- | --- | --- |
| 项目源码 | `/data/autovla/projects/MoReAct` | 从此目录执行本文命令 |
| 已有 Python 环境 | `/data/users/autovla/.envs/remogen-motion-only/bin/python` | 生成、语义模块均使用过的解释器 |
| Inter-X 原始数据根 | `/data/autovla/projects/interx_ardy_mvp/full_data` | 数据准备输入；不是 bridge 的派生特征 |
| 原始动作 | `/data/autovla/projects/interx_ardy_mvp/full_data/motions` | 每个 episode 下 `P1.npz`、`P2.npz` |
| 整体交互文本 | `/data/autovla/projects/interx_ardy_mvp/full_data/texts.zip` | 生成的可选文本条件、理解的监督文本 |
| 官方划分 | `/data/autovla/projects/interx_ardy_mvp/full_data/splits` | train/val/test 按 episode 隔离 |
| Actor/Reactor 顺序 | `/data/autovla/projects/interx_ardy_mvp/full_data/annots/interaction_order.pkl` | order=0：P1→P2；order=1：P2→P1 |
| 已审核角色修正 | `/data/autovla/projects/MoReAct/assets/role_overrides.json` | 高于原始 order；保留审查证据 |
| SMPL-X 模型根 | `/data/autovla/projects/remogen_official_release/smpl_models` | 内有 `smplx/SMPLX_MALE.npz`、`SMPLX_FEMALE.npz`、`SMPLX_NEUTRAL.npz` |
| CLIP ViT-B/32 | `/home/autovla/.cache/clip/ViT-B-32.pt` | 冻结文本编码，512 维条件 |
| 原始 FLAN-T5-large | `/data/autovla/projects/models/flan-t5-large` | 理解训练的语言初始化及本地 tokenizer 资产 |

`single_person_texts*` 并非本项目默认交互描述监督入口。不要把其中 first/second person 直接当作 P1/P2。
SMPL-X 根是身体模型目录，不是 ReMoGen 的动作 checkpoint；本项目运行不需要导入 ReMoGen 源码。

### 2.2 动作缓存及基础权重

| 资源 | 绝对路径 | 用途 |
| --- | --- | --- |
| 当前动作缓存 | `/data/autovla/projects/MoReAct/data/interx_h2_f8` | 30 FPS、H=2、F=8、276D |
| 样本文件 | `/data/autovla/projects/MoReAct/data/interx_h2_f8/episodes` | 单 episode NPZ，供训练和语义 describe 使用 |
| 数据身份 / 角色 / 划分 | `/data/autovla/projects/MoReAct/data/interx_h2_f8/manifest.json` | 与 checkpoint 的 digest 配套 |
| 归一化统计 | `/data/autovla/projects/MoReAct/data/interx_h2_f8/stats.npz` | 只由训练集统计，双方等权 |
| 准备报告 | `/data/autovla/projects/MoReAct/data/interx_h2_f8/preparation_report.json` | 样本数量、跳过原因、统计版本 |
| 训练窗口核验 | `/data/autovla/projects/MoReAct/data/interx_h2_f8/input_validation.json` | 可用窗口数、双方统计权重 |
| 早期缓存 | `/data/autovla/projects/MoReAct/data/interx` | 历史资产，不替换当前 H=2/F=8 缓存 |
| CVAE 原训练目录 | `/data/autovla/projects/MoReAct/runs/cvae_h2_f8_equal_roles_20260924_163259` | 原始训练配置、日志、last；不要把可变 last 当固定基线 |
| 固定 CVAE 文件 | `/data/autovla/projects/MoReAct/runs/diffusion_geometry_20260925_122644/frozen_vae.pt` | 现有生成实验使用的固定 VAE 资产之一；以各运行 provenance 为准 |

准备报告 counts：train=9110、val=570、test=1707，总计11387段。
`input_validation.json` 中满足训练连续窗口要求的 train episodes=9106，train windows=1325879；
因此“准备成功段数”和“可抽训练窗口段数”不是同一统计口径。
数据 digest：`32893a39105b183275d36b61d72f2e4d5b7fe1a999c382906e0277b4557ca048`。

### 2.3 当前生成训练与检查点

当前主运行目录：
`/data/autovla/projects/MoReAct/runs/diffusion_geometry_8gpu_b1024_20260927`。

| 相对于主运行目录的文件 | 用法 |
| --- | --- |
| `launch.yaml`、`config.json` | 实际运行配方；优先于默认示例 |
| `provenance_start.json` | 原始启动命令、来源 checkpoint、hash、batch、目标步数 |
| `distributed.json` | world size、local/global batch 和恢复来源 |
| `last.pt` | 最新可恢复训练状态，会更新 |
| `step_034000.pt` 等 `step_NNNNNN.pt` | 固定步骤文件，用于稳定复现和对比 |
| `best.pt` + `best.json` | EMA 真值历史验证 loss 最优，不能等同长 rollout 质量最优 |
| `best_rollout.pt` + `best_rollout.json` | EMA 生成历史验证 loss 最优，仍是 loss 代理 |
| `metrics.jsonl` | 持久训练、验证、生成历史验证指标 |
| `validation_samples/step_*.json` | 对应步骤的分层验证样本清单 |

当前运行从下面文件恢复：
`/data/autovla/projects/MoReAct/runs/diffusion_geometry_long_b512_20260927/step_032500.pt`。
来源 SHA256：`6b31f4f0ca8d72834a19715eb8bd98961dec5346d27baa47192a9f0615ccd208`。

本次核对时主进程 PID=2426542，world_size=8，物理 GPU=0–7；global batch=1024，local batch=128。
起点32500，目标42750；每500步保存/验证，启用 stratified_action 和 generated-history validation。
学习率基值约3.93605e-5，从 step32500 起按新进度退火。
特殊阶段长度32500/1/10249用于此次恢复后的进度转换，**不是从零训练的通用配方**。

选定步骤的额外 root rollout 评估：

- 验证集：`/data/autovla/projects/MoReAct/outputs/diffusion_geometry_8gpu_b1024_20260927/root_val`。
- 训练集：`/data/autovla/projects/MoReAct/outputs/diffusion_geometry_8gpu_b1024_20260927/root_train`。
- watcher 等待 step35000、38000、41000、42750；存在 watcher 不代表这些结果已全部完成。
- 固定验证选择：`/data/autovla/projects/MoReAct/outputs/root_drift_short_20260927/val_selection.json`。

### 2.4 必须整套使用的语义资产

| 环节 | 绝对路径 |
| --- | --- |
| 冻结源 diffusion | `/data/autovla/projects/MoReAct/runs/diffusion_stage2_b512_20260926_084433/step_007000.pt` |
| 独立冻结 CVAE 快照 | `/data/autovla/projects/MoReAct/data/semantics_rvq_20260926/latents/vae_snapshot.pt` |
| 双方向 mu 缓存 | `/data/autovla/projects/MoReAct/data/semantics_rvq_20260926/latents` |
| RVQ 权重 | `/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926/rvq/best.pt` |
| token 缓存 | `/data/autovla/projects/MoReAct/data/semantics_rvq_20260926/tokens` |
| T5 描述权重 | `/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926/captioner/best.pt` |
| T5 恢复状态 | `/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926/captioner/last.pt` |
| T5 身份契约 | `/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926/captioner/contract.json` |
| 流水线状态 | `/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926/status.json` |
| 最终开发集比较 | `/data/autovla/projects/MoReAct/outputs/semantics_rvq_20260926/comparison_final/comparison.json` |

latents manifest、tokens manifest、captioner contract 的 CVAE 身份、统计身份与数据 digest 一致：

- CVAE identity：`a67fb8295e257b05a9dc6fcdce4c27cc92b8f7f92a22433d08d0e64bc57f7c4a`。
- stats identity：`bffa89b764b726615b0dc1e89554c2ca7278e07cfc554dc5de60c3736b08d99e`。
- RVQ identity：`a934bdf1aa4bca53f212f9536b35d6ed271cf2ad867d678bc121139ba8388801`。

这些是项目定义的资产身份，不全部等于某个 `.pt` 文件的 SHA256。
换 CVAE/统计必须重新提取 latent；换 RVQ 必须重新编码 token 并训练匹配的 T5。
理解使用 mu，不使用 posterior sample，也不使用 diffusion 的 latent_scale 做量化。

## 3. 环境与启动前检查

```bash
cd /data/autovla/projects/MoReAct
export MOREACT_PYTHON=/data/users/autovla/.envs/remogen-motion-only/bin/python
"$MOREACT_PYTHON" -m moreact --help
"$MOREACT_PYTHON" -m moreact semantics --help
nvidia-smi
command -v ffmpeg
```

核对环境：Python3.8.20、torch2.1.0、numpy1.21.5、smplx0.1.28、transformers4.38.2、tokenizers0.15.2、wandb0.16.6。
`clip` 模块存在；其安装元数据不叫 `openai-clip`。本环境未找到 sentencepiece，但已有本地 tokenizer 资产支持此前完成的语义训练；
新建环境仍按 `pyproject.toml` 的 semantics extra 安装完整依赖，不把这台机器的缺省状态当安装规范。

新环境安装入口（在新环境中执行，不升级正在训练的环境）：

```bash
python -m pip install -e '.[text,test,semantics,tracking]'
```

`configs/default.yaml` 为底层默认；`--config` 指定的 YAML 深度覆盖默认值。
数据字段相对路径按项目根解释；命令仍统一在项目根执行，避免输出和覆盖配置路径含义不清。
`configs/diffusion_geometry.yaml` 只是覆盖片段；某次实验的 `launch.yaml` 才是完整实际参数。
GPU号只是命令示例，先确认可用设备；本文编写时8卡主训练正在运行，不要再启动重复作业。

## 4. 数据准备与表示约定

```bash
"$MOREACT_PYTHON" -m moreact prepare --device cuda:0
```

此命令使用默认完整缓存。要改 H/F、FPS、角色修正或特征配置，应使用新的 cache 目录与覆盖配置，不能覆盖已有训练缓存。
`prepare --limit N` 是每个 split 的上限，仅用于独立临时缓存；重用 episode 特征可用 `--reuse-cache data/interx`，
但仍会核验源摘要并重新计算新配置的统计。

数据契约：120 FPS原始动作按4倍下采样为30 FPS；Y-up转换为Z-up；保留双方gender/betas和共同地面。
每人276维：transl3 + root/身体poses6d132 + transl_delta3 + global_orient_delta6d6 + joints66 + joints_delta66。
增量用当前减前帧，禁止混入前向差分或未来平滑。共同坐标系由目标人物最后历史帧的位置/朝向确定。
H=2、F=8：模型每次观察2帧历史、生成8帧，每约0.267秒更新一次actor历史，不是每帧重新规划。
单样本训练连续4个primitive；actor未来不作为模型条件。训练集统计中的双方等权贡献不等于使用actor未来作条件。

## 5. 从零训练生成模型

以下是**新实验模板**，不是当前8卡训练的启动指令；使用独立输出目录。

### 5.1 训练 CVAE

```bash
"$MOREACT_PYTHON" -m moreact train_vae \
  --output runs/my_generation_v1/vae --device cuda:0
```

默认batch128、AdamW lr1e-4、weight decay0、grad clip1、EMA0.999。
默认阶段100k/100k/100k：真实历史 → 逐渐增加生成历史 → 全生成历史。
VAE：5层、hidden256、latent1×128；loss含特征/关节/FK/增量一致性、骨长和脚接触，KL权重1e-6。

### 5.2 冻结 CVAE，训练 diffusion

```bash
"$MOREACT_PYTHON" -m moreact train_diffusion \
  --config configs/diffusion_geometry.yaml \
  --vae runs/my_generation_v1/vae/last.pt \
  --output runs/my_generation_v1/diffusion --device cuda:0
```

当前几何配方：latent MSE，加解码特征、SMPL-X关节、FK一致性、速度、骨长、脚接触、root朝向及角速度、双人距离与接触。
`smpl_joints_rec=20`；24个root特征通道内部×5；distance map使用GT距离<1.5m；接触阈值0.10m。
没有独立绝对 `root_position` 项。训练历史以0.25概率加入标准差2cm、上限5cm的水平平移，
按实际历史重算未来首帧增量；验证/推理不加噪声。完整权重以配置为准。

去噪器8层、hidden512、4heads；10步cosine DDPM，预测z0；完整CVAE冻结。
估计latent标准差进行缩放，不减均值。文本dropout0.1，CFG只移除文本、不移除动作历史。

### 5.3 新建多卡训练

复制配方到自己的launch.yaml并设置global batch；它必须能被卡数整除。例如global512 / 4卡 = 每卡128。

```bash
"$MOREACT_PYTHON" - <<'PYCFG'
from pathlib import Path
import yaml
from moreact.config import load_config
cfg = load_config("configs/diffusion_geometry.yaml")
cfg["train"]["batch_size"] = 512
out = Path("runs/my_generation_v1/launch.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PYCFG
"$MOREACT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m moreact.train_distributed \
  --config runs/my_generation_v1/launch.yaml \
  --kind diffusion --vae runs/my_generation_v1/vae/last.pt \
  --output runs/my_generation_v1/diffusion_ddp --devices 0,1,2,3
```

上例先合并默认与几何覆盖配置生成 `runs/my_generation_v1/launch.yaml`；若文件已存在会拒绝覆盖。
`--devices` 是脚本使用的可见GPU编号，不要再叠加不一致的 `CUDA_VISIBLE_DEVICES` 重映射。
DDP仅同步可训练网络；冻结CVAE、SMPL-X每卡各一份。改变global batch和optimizer步数会改变训练轨迹。

## 6. 恢复已有训练与阶段切换

同目标、同配置恢复，使用原run的launch.yaml和last.pt；`--steps` 表示绝对目标步骤。
下面是当前主run未来中断后的恢复命令，**仅在确认原训练已停止后执行**：

```bash
RUN=/data/autovla/projects/MoReAct/runs/diffusion_geometry_8gpu_b1024_20260927
"$MOREACT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m moreact.train_distributed --kind diffusion \
  --config "$RUN/launch.yaml" --output "$RUN" --resume "$RUN/last.pt" \
  --devices 0,1,2,3,4,5,6,7 --steps 42750
```

普通恢复**不加 `--stage2`**。该参数仅用于显式迁移到新run；要求 stage1_steps 和 lr_schedule_start_step 等于来源checkpoint.step，
允许指定batch/课程变更，并重置不再可比的best。新损失目标用新run和 `--new-objective`；不要借普通resume悄悄改变损失。
`--validate-at-start` 可保存起点验证，尚未进行训练更新也可能出现best文件。

checkpoint包含模型、EMA、optimizer、RNG、step、配置、mean/std与数据digest；diffusion内嵌冻结CVAE。
只有权重不等于完整训练恢复；不要删来源checkpoint或正在等待评估的step文件。

观察训练：

```bash
tail -n 3 "$RUN/metrics.jsonl"
cat "$RUN/best.json" "$RUN/best_rollout.json"
```

W&B可选，由 `scripts/sync_diffusion_wandb.py --run RUN` 独立读取metrics.jsonl和wandb_spec.json上传；
需要既有登录及spec，不在配置内写密钥。上传成功与训练完成是两个状态，JSONL是本地事实来源。

## 7. 动作生成与评估

下面选现存的固定step034000作为可复现示例，**不宣称它是最佳生成模型**：

```bash
GEN=/data/autovla/projects/MoReAct/runs/diffusion_geometry_8gpu_b1024_20260927/step_034000.pt
CACHE=/data/autovla/projects/MoReAct/data/interx_h2_f8
"$MOREACT_PYTHON" -m moreact rollout \
  --checkpoint "$GEN" --cache "$CACHE" \
  --split test --episode G001T000A000R004 \
  --output outputs/my_rollout_v1 --frames 120 --seed 0 --no-text --device cuda:0
```

`--no-text` 用零embedding；`--text '...'` 使用指定文本；不传两者则读取数据集参考caption，不能称为在线动作理解。
`--guidance 1` 为默认；`--start-frame` 是历史起点，生成从起点+H开始。`--no-video` 可跳过视频。
输出motion.npz、actor/reactor/target_smplx.npz、metadata.json和preview.mp4；SMPL-X导出明确为Z-up。
独立SMPL-X文件不能直接当成语义describe的prepared NPZ。

```bash
"$MOREACT_PYTHON" -m moreact evaluate \
  --checkpoint "$GEN" --cache "$CACHE" --split val --limit 8 \
  --output outputs/my_eval_v1 --frames 120 --device cuda:0
```

`--limit 0` 为全split；默认limit8仅是小样本。默认有/无文本 × actor正常/历史逆序/移除六组。
VAE evaluate使用真实未来posterior mean做重建；diffusion evaluate是真实前缀后自回归采样，不能混比成同一种指标。
报告包括root ADE/FDE、世界/根对齐关节误差、双人距离、脚滑、边界速度跳变等。
脚滑接触来自高度代理；最近关节距离不等于网格碰撞率。推理计时不含加载/文本编码/渲染/I/O，不能直接证明端到端实时。

在线集成：`ReactionGenerator.initialize()`输入reactor历史 `[B,H,276]` 及body参数；
`step(actor_history,state,text_embedding)`输出 `[B,F,276]` 世界坐标未来与新state。
body顺序actor/reactor，betas `[B,2,10]`、genders `[B,2]`、offsets `[B,2,3]`，文本 `[B,512]`。
接口没有actor未来参数；完整约定见源码和项目README。

## 8. 理解模块：使用现成资产

```bash
SEM_DATA=/data/autovla/projects/MoReAct/data/semantics_rvq_20260926
SEM_RUN=/data/autovla/projects/MoReAct/runs/semantics_rvq_20260926
"$MOREACT_PYTHON" -m moreact semantics describe \
  --motion-checkpoint "$SEM_DATA/latents/vae_snapshot.pt" \
  --rvq "$SEM_RUN/rvq/best.pt" --checkpoint "$SEM_RUN/captioner/best.pt" \
  --input data/interx_h2_f8/episodes/G001T000A000R004.npz \
  --output outputs/my_description_v1.json --motion-device cpu --device cuda:0
```

describe读取features/betas/genders/offsets，不读取caption作为动作输入；输出路径必须是新的。
完整动作输入最高4096tokens、目标最高512，不静默截断；空文本或未完整结束会记录valid=false。

```bash
"$MOREACT_PYTHON" -m moreact semantics evaluate \
  --cache "$SEM_DATA/tokens" --rvq "$SEM_RUN/rvq/best.pt" \
  --checkpoint "$SEM_RUN/captioner/best.pt" \
  --output outputs/my_semantics_test_v1 --split test --device cuda:0
```

不传limit为全量；可先 `--limit 8` 检查流程。若增加重建诊断，必须同时提供
`--motion-checkpoint "$SEM_DATA/latents/vae_snapshot.pt" --motion-cache data/interx_h2_f8`。
重建诊断使用真实历史，不代表长时动作生成效果。

## 9. 理解模块：重新训练完整链路

新实验统一使用新目录，固定不可变的生成checkpoint作为CVAE来源：

```bash
"$MOREACT_PYTHON" -m moreact semantics cache-latents \
  --checkpoint runs/diffusion_stage2_b512_20260926_084433/step_007000.pt \
  --cache data/interx_h2_f8 --output data/my_semantics_v1/latents --device cpu
"$MOREACT_PYTHON" -m moreact semantics fit-rvq \
  --cache data/my_semantics_v1/latents --output runs/my_semantics_v1/rvq \
  --epochs 20 --levels 2 --size 256 --batch-size 1024 --seed 42 --device cpu
"$MOREACT_PYTHON" -m moreact semantics cache-tokens \
  --cache data/my_semantics_v1/latents --rvq runs/my_semantics_v1/rvq/best.pt \
  --output data/my_semantics_v1/tokens --device cpu
"$MOREACT_PYTHON" -m moreact semantics train \
  --cache data/my_semantics_v1/tokens --output runs/my_semantics_v1/captioner \
  --base-model /data/autovla/projects/models/flan-t5-large \
  --steps 5000 --batch-size 1 --gradient-accumulation 16 --gradient-checkpointing \
  --learning-rate 0.0001 --validate-every 100 --seed 42 --device cuda:0
```

RVQ仅train更新，双方等权，K2×256codes，128维，EMA0.99；按val平均量化MSE选best。
T5全部参数含新embedding可训练，但不会反向更新CVAE/RVQ；有效batch16，按val NLL选best。
恢复RVQ追加 `--resume runs/my_semantics_v1/rvq/last.pt --epochs 30`；恢复T5使用同目录、同优化/累积参数，
追加 `--resume runs/my_semantics_v1/captioner/last.pt`，steps仍为绝对目标。

已有 `scripts/run_semantics_training.py` 包含历史cache_pid、GPU等待和bridge对比流程，
不要盲目重复运行原配方覆盖已完成实验；优先使用上述单阶段命令。

## 10. 已有结果、证据与解释边界

语义流水线status为complete，captioner最后日志step5000；末次val_nll约1.04872。
这是last步骤的值，不代表best的NLL。旧架构/训练记录中的“正式训练待完成”属于先前时点；实现边界仍以架构文档为准。

固定64条共同val样本的最终比较：

| 系统 / 条件 | word-F1 | ROUGE-L-F1 | valid fraction |
| --- | ---: | ---: | ---: |
| Bridge原生系统 | 0.50117 | 0.43064 | 1.0 |
| MoReAct正常 | 0.50817 | 0.43776 | 1.0 |
| MoReAct仅第一层 | 0.47992 | 0.40278 | 1.0 |
| MoReAct打乱时间 | 0.51047 | 0.43617 | 1.0 |
| MoReAct移除actor | 0.46400 | 0.38621 | 1.0 |
| MoReAct移除reactor | 0.47110 | 0.38845 | 1.0 |

来源：`outputs/semantics_rvq_20260926/comparison_final/comparison.json`及selection/逐样本输出。
这是开发集小规模原生系统对比，训练任务与预算不同；不是全test结果，不足以宣称显著提升。
打乱时间后词面指标接近，不能由正常分数推断时序理解可靠。actor方向为共享CVAE的迁移编码，仍需人工核验角色/动作/顺序。

生成短训A/B/C报告：`docs/reports/20260927_rollout_root_reduction/final_results.md`。
B相对A test root ADE改善约0.5%，95% CI跨零；C也未达到门槛。此结果不是当前8卡长训的最终质量结论。

## 11. 42_store 文档交接与未来资产迁移

选用目录：`/mnt/42_store/autovla/MoReAct/documentation/`。
内容：本汇总Markdown、机器可读资产路径清单和校验和文件；大型数据/权重未复制。
43_store可用作后续替代位置，例如 `/mnt/43_store/autovla/MoReAct/`，本次未写入该路径。

核对时42_store约1.3T可用、43_store约658G可用；空间会变化，复制资产前重新 `df -h`。
不要仅复制checkpoint就认为项目可复现，也不要移动正在写入的run。

如未来需要完整离线资产包，按任务选择：

| 目标 | 必须一起保留 |
| --- | --- |
| 现成动作生成 | 源码/环境说明 + 固定diffusion checkpoint + SMPL-X + 相匹配prepared cache；自定义文本还需CLIP |
| 从原始数据重建 | 上述外部数据根中的motions/texts.zip/splits/interaction_order + role_overrides + CLIP + 原配置 |
| 恢复生成训练 | 完整last checkpoint + launch/config/provenance + 原data统计/digest + 日志；保留来源和评估step |
| 现成动作理解 | vae_snapshot + RVQ + captioner + contract/tokenizer/base-model资产 + 匹配的prepared动作与身体模型 |
| 恢复理解训练 | latent/token manifests与缓存 + RVQ/T5 last + 优化/词表/身份契约 + 配置日志 |

复制固定step后记录SHA256；active last只能取一致快照，不在写入时直接拷贝。
迁移后需核对配置与checkpoint内嵌路径。rollout可用 `--cache`、`--body-models`覆盖部分路径，
但CLIP和语义/评估其他内嵌路径不一定都暴露CLI覆盖；不能只改一份default.yaml就认为所有旧checkpoint已迁移。
保留兼容路径或显式验证每个加载入口。文件缺失时先定位真实资产，不用随机模型/旧统计替代。

## 12. 常见问题与维护

| 现象 | 排查顺序 |
| --- | --- |
| 数据digest/mean/std不匹配 | 核对H/F、统计版本、role_overrides、缓存与checkpoint来源；不用早期cache替换 |
| RVQ或T5身份错误 | 核对固定CVAE snapshot、RVQ ID、token manifest、captioner contract是否同套 |
| resume目标或配置错误 | 确认同目标普通恢复；改变损失/课程须新run，使用明确切换参数 |
| 显存不足 | 查看活跃GPU；global/local batch分别确认；T5启用梯度检查点/累积，不能静默截短动作 |
| describe输入缺字段 | 使用prepared episode NPZ，不使用独立SMPL-X导出 |
| 角色颠倒/地面异常 | 核对原order与审核修正、共同地面、Y-up→Z-up；勿分别落地 |
| 训练loss好但长生成漂移 | 查看generated-history验证与固定120帧root rollout，不只看teacher-history best |
| 旧脚本找不到路径 | 工作区整理保留兼容链接；依赖仍以绝对路径清单为准 |

临时smoke用 `bash scripts/smoke.sh`：会使用GPU，每次独立tmp，先保存证据再清理；文档检查无需执行。
正式runs/data、未提交代码、恢复来源、watcher等待的checkpoint不能按名称/年龄删除。
数据和模型许可证分别遵循原资产约束，项目代码来源见NOTICE.md。

进一步阅读（项目内路径）：
`README.md`、`docs/ARCHITECTURE.md`、`docs/SHARED_RVQ.md`、`docs/DISTRIBUTED_TRAINING.md`、
`docs/DIFFUSION_LOSSES.md`、`docs/MAINTENANCE.md`、`docs/reports/README.md`。
