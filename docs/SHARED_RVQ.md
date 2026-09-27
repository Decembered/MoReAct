# 共享连续 CVAE + K=2 RVQ 整段理解

状态：接口、缓存、量化、T5 训练/恢复和评估链路已实现；正式理解质量待训练验证。
第一版冻结既有 CVAE，仅理解分支量化；不重训 diffusion，不把生成改成量化解码。
架构见 [ARCHITECTURE.md](ARCHITECTURE.md)，验证范围见 [报告](reports/20260926_shared_rvq/README.md)。

## 环境与入口

使用具备兼容 PyTorch 的项目环境，理解可选依赖为 `transformers>=4.38,<4.43`、sentencepiece 和 tokenizers：

```bash
pip install -e '.[semantics]'
python -m moreact semantics --help
```

语言资产默认 `/data/autovla/projects/models/flan-t5-large`，只从本地加载。
基础动作生成不加载 transformers，不需要 bridge 的源码路径。下面命令均从项目根目录执行。
正式长训练预算和设备由运行者显式指定；示例命令不是已经执行过的正式训练。

## 1. 固定生成 checkpoint，提取整段 latent

```bash
python -m moreact semantics cache-latents \
  --checkpoint runs/diffusion_restart_b512_20260926_013059/step_001000.pt \
  --cache data/interx_h2_f8 --output data/shared_rvq_v1/latents --device cpu
```

`--checkpoint` 必须为内嵌 CVAE 的 diffusion checkpoint，或本命令导出的语义快照。
示例中的 step 文件是不可变基线，不代表最佳动作质量。加载时核对源文件身份并导出只包含冻结 CVAE、
配置、统计及身份的 `vae_snapshot.pt`。不读取不断更新的训练权重作为动态依赖。

每段使用 `[2,T,276]` 世界坐标因果特征，角色顺序 actor/reactor；betas `[2,10]`、genders `[2]`、offsets `[2,3]`。
默认 30 FPS、H=2、F=8、128D latent，实际参数必须来自同一个 checkpoint。
双方各作为一次目标，交换时同步调整条件槽及身体偏移，并用目标人物历史建立参考系。

`encode_mean` 在 eval 模式直接返回 mu，不消耗随机数。未使用 posterior sample、latent scale 或额外单位归一化。
前 H 帧作为种子历史；目标每 F 帧推进。尾部不足 F 时取覆盖末尾的完整目标窗，保留重叠及新增有效帧数。
不足 H+F 帧报错。缓存 mu 形状 `[窗口,人物,latent_dim]`，spans 为 `[start,end,new_frames]`，end 不含在窗口内。

输出 manifest 保留 episode/split、角色、所有交互 caption、源动作哈希、文件哈希和身份。
caption 只在缓存监督元数据中出现，不传给 motion tokenizer。`--limit N` 限制每个 split，正式默认 0 为全量。

## 2. 只在 train latent 上拟合 RVQ

```bash
python -m moreact semantics fit-rvq --cache data/shared_rvq_v1/latents \
  --output runs/shared_rvq_v1/rvq --epochs 20 --device cpu
python -m moreact semantics cache-tokens --cache data/shared_rvq_v1/latents \
  --rvq runs/shared_rvq_v1/rvq/best.pt --output data/shared_rvq_v1/tokens
```

默认 K=2、每层 256 个码、128 维、EMA decay=0.99，每次更新 1024 个双人窗口。
每个窗口同时贡献双方，等权拟合；校验/测试不更新码本。第一层量化 mu，第二层量化残差，码向量相加。
码本使用训练 batch 初始化/替换低使用率条目，使用独立随机发生器打乱窗口，不训练 encoder。

每轮记录双方向验证 MSE、码本使用数和 perplexity；按双方平均验证量化误差保存 best，last 保存 EMA 与 RNG。
`--resume runs/shared_rvq_v1/rvq/last.pt --epochs 30` 表示同配置/同缓存续至第 30 轮。
`--levels 1` 可训练独立 K=1 对照；所有产物使用新目录。

离散缓存形状为 `[窗口,2,K]`，原始索引各为 `0..255`；语言边界使用独立符号 `<rvq0_i>`、`<rvq1_j>`。
每时间组顺序为 `<step><valid_n><actor>...<reactor>...`，不拆开两个残差层。
CVAE、统计、数据、码本、缓存文件身份不匹配即拒绝加载。

## 3. T5 整段描述训练

```bash
python -m moreact semantics train --cache data/shared_rvq_v1/tokens \
  --output runs/shared_rvq_v1/captioner --steps 1000 --validate-every 100 \
  --base-model /data/autovla/projects/models/flan-t5-large --device cuda:0
```

设备示例不表示 GPU 空闲，应按实际资源安排。`--steps` 必填，是绝对更新目标。
默认 batch=1、AdamW lr=1e-4、weight decay=0、梯度裁剪=1。全部 T5 参数含新增 embedding 可训练；
不会加载旧 bridge 的 motion embedding，也不加载 CVAE/RVQ 来进行反向传播。

显存受限时可用 `--gradient-checkpointing --gradient-accumulation 16`。
有效 batch 为 `batch-size × gradient-accumulation`，每个 micro-batch 的目标 token 平均 loss
等权累积；batch-size=1 时为逐样本平均。恢复必须保持这些设置一致。
正式运行及 bridge 对比见 [2026-09-26 训练记录](reports/20260926_semantics_training/README.md)。

训练随机抽取 train episode 和一条原始 caption；验证使用全部 val episode 的固定第一条 caption，
按有效目标 token 统计 NLL。每次验证保存 last 和更优的 best。NLL 最优不是语义质量最优的证明。
恢复使用同一 output 和 `--resume .../last.pt`，验证缓存、优化参数及词表，恢复 optimizer 和 RNG。

第一次更新前审计所有 split 的完整动作输入及每条参考 caption，写 `length_audit.json`。
输入最多 4096 tokens、目标最多 512，禁止静默截断；超限写错误记录后停止，不进行参数更新。
推理禁止生成动作/角色标记，记录 EOS 完整性；不完整或空输出标记 `valid=false`，不伪造替代文本。

## 4. 描述与评估

```bash
python -m moreact semantics describe \
  --motion-checkpoint data/shared_rvq_v1/latents/vae_snapshot.pt \
  --rvq runs/shared_rvq_v1/rvq/best.pt --checkpoint runs/shared_rvq_v1/captioner/best.pt \
  --input data/interx_h2_f8/episodes/G001T000A000R000.npz --output outputs/description.json

python -m moreact semantics evaluate --cache data/shared_rvq_v1/tokens \
  --rvq runs/shared_rvq_v1/rvq/best.pt --checkpoint runs/shared_rvq_v1/captioner/best.pt \
  --output outputs/shared_rvq_v1/test --split test \
  --motion-checkpoint data/shared_rvq_v1/latents/vae_snapshot.pt --motion-cache data/interx_h2_f8
```

describe 输入为 MoReAct prepared NPZ，必须包含 features/betas/genders/offsets；不会读取 caption 字段。
不直接接受仅含 poses/trans 的导出文件，需要先转换为同一特征约定。
输出包含文本、完整性、有效性、帧数、时间窗和模型身份。

evaluate 记录正常、打乱时间组、移除 actor、移除 reactor、只保留第一层的描述及多参考 word-F1/ROUGE-L-F1。
这些是词面代理指标；报告为人工填写动作、角色、先后关系错误预留 review 字段。
只保留第一层是 K=2 模型的推理消融；公平 K=1 对照应另训码本及 captioner，并在相同 split 上比较。

同时提供 motion-checkpoint/motion-cache 时额外输出 `reconstruction.json`，比较双方 `decode(mu)` 和
`decode(q1+q2)` 的 FK、root ADE/FDE、脚滑及段间连续性；尾窗仅拼接新增帧。
它采用真实历史，是重建诊断，不是 diffusion 生成质量。量化后的解码从不进入正式生成。

## 共享实例的 Python API

```python
from moreact.generate import ReactionGenerator
from moreact.semantics import SharedMotionTokenizer, InteractionCaptioner

generator = ReactionGenerator("diffusion.pt", device="cuda:0", seed=0)
motion = SharedMotionTokenizer.from_generator(generator)
assert motion.vae is generator.vae
captioner = InteractionCaptioner(motion, "rvq/best.pt", "captioner/best.pt", device="cuda:1")
result = captioner.describe(features, betas, genders, offsets)
```

理解和生成必须绑定完全相同的 VAE/统计；接口只读取冻结实例，不改变生成采样器、latent scale 或 RNG。
改变共享参数或将 VAE 切到 train 模式会使语义编码报错。量化器与文本模型也检查资产身份。

## 验证与限制

CPU 测试覆盖旧 encode 数值和 RNG 兼容、确定性均值、双方向与尾窗、RVQ 残差与恢复、split 隔离、
缓存/资产错误、T5 训练恢复及真实小型模型推理。真实资产验证范围见 [报告](reports/20260926_shared_rvq/README.md)。
当前 encoder 未按双目标训练，actor token 是迁移表示；冻结 latent 可能缺失完整语义信息。
本版没有通过微调 CVAE 或额外读取原始动作补偿该缺陷，也不把工程接通宣称为理解质量达标。

新实验遵守 [维护规则](MAINTENANCE.md)：临时内容在独立 tmp 目录，先归档轻量证据再清理；
正式缓存和训练目录不覆盖，码本冻结后修改必须重新生成 token 并对齐语言模型。
