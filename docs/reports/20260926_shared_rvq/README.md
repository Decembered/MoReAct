# 共享 CVAE / RVQ 第一版实现与验证

日期：2026-09-26。结论：冻结 CVAE 的连续生成与离散理解分支已接通，六个语义命令可训练、恢复和评估。
本次没有正式训练理解模型，没有证明语义质量达标，也没有修改或停止已有训练任务。

实现规范见 [SHARED_RVQ.md](../../SHARED_RVQ.md)，当前架构见 [ARCHITECTURE.md](../../ARCHITECTURE.md)。

## 实现范围

- `ReactionVAE.encode_mean` 使用原 encoder 的确定性 mu，要求 eval，不采样；旧参数名和 encode/decode 路径兼容。
- `moreact/semantics/` 提供共享实例、双方向分段、重叠尾窗、资产绑定、训练集 EMA RVQ、T5 与重建诊断。
- CLI 提供 cache-latents、fit-rvq、cache-tokens、train、describe、evaluate；默认 CPU，长训练步数显式指定。
- CVAE/统计/RVQ/词表身份不匹配报错；描述及评估还返回所加载 caption checkpoint 的 SHA-256。
- 动作生成仍使用连续 latent；整段理解结果不进入生成条件。普通生成不导入 T5 或 bridge 源码。
- 主架构已更新；早期在线闭环内容移到历史文档。运行时会话与在线理解不属于这次交付。

## 验证环境和资产

全部新验证在 CPU 执行，4 threads；Python 3.8、PyTorch 2.1.0、transformers 4.38.2。
解释器为 `/data/users/autovla/.envs/remogen-motion-only/bin/python`。

真实生成 checkpoint 为 `runs/diffusion_restart_b512_20260926_013059/step_001000.pt`，
使用其内嵌 CVAE、配置和统计。真实动作来自 `data/interx_h2_f8`，仅取 train/val/test 各 1 段。
完整源文件 SHA-256 见 [source.json](source.json)，CVAE 和统计身份见
[缓存记录](evidence/real_cache.log)。这些正式源资产未删除。

真实语言权重为本地 `/data/autovla/projects/models/flan-t5-large`。只做一次无梯度前向，
没有更新该模型。训练、恢复、describe 和评估流程另外使用随机小型 T5，必须与真实语言验证分开解读。

## 检查结果

| 检查 | 结果与证据 |
| --- | --- |
| 完整 CPU 测试 | 32 passed，见 [full_tests.log](evidence/full_tests.log) |
| 缓存源身份及恢复扩展测试 | 8 passed，见 [final_semantics_tests.log](evidence/final_semantics_tests.log) |
| Caption 身份及完整推理 RNG 检查 | 8 passed，见 [asset_identity_tests.log](evidence/asset_identity_tests.log) |
| 真实 CVAE 缓存 | 3 段，30 FPS / H=2 / F=8 / 276D / 128D |
| 真实 RVQ | K=2，每层 256，EMA decay=0.99，2 epoch，batch=32；另做同配置 K=1 |
| 原始 FLAN-T5-large 前向 | loss=5.911961，有限；输入 227、目标 63 tokens；无参数更新，见 [记录](evidence/real_t5_forward.json) |
| 随机小 T5 流程 | d_model=16、1 层、原 FLAN 词表，2 updates；描述、五组消融和真实 SMPL-X 诊断运行完成 |
| 长度审计 | 3 段全部通过，无截断；只代表这次样本，正式训练必须审计全量缓存，见 [审计](evidence/real_length_audit.json) |

CPU 自动测试包括旧 encode 逐值/RNG 兼容、均值重复性、共享实例前后生成一致、角色和参考系交换、
窗口与短输入、两级残差、恢复一致、train/val 隔离、冻结资产字节一致、词表篡改和超长输入拒绝。
完整描述初始化与推理也验证不改变 CPU torch RNG。未宣称跨设备或不同计算环境逐位一致。

最初的小型 tokenizer 测试暴露 `token_type_ids` 被传给 T5 的问题，已显式关闭该字段并通过回归。
恢复加载会冻结推理模型，训练恢复路径已显式恢复 T5 的可训练状态；连续两步与中断恢复两步参数逐值一致。

## 指标的适用边界

RVQ 第 1 轮的最低双向平均验证量化 MSE 为 **0.9825328271**，actor=0.8313406834、
reactor=1.1337249709；按此保存 best。第 2 轮末层使用数为 17/4，完整每轮指标见
[K=2](evidence/real_rvq/metrics.jsonl) 和 [K=1](evidence/real_rvq_k1/metrics.jsonl)。
这次训练数据极少，K=1 与 K=2 误差几乎相同，不能据此证明第二层的收益或充分码本覆盖。

真实 val 样本、真实 SMPL-X、真实历史条件下的重建结果：

| 目标人物 | decode(mu) MPJPE | decode(q1+q2) MPJPE |
| --- | --- | --- |
| actor | 3.43 mm | 21.18 mm |
| reactor | 3.33 mm | 37.18 mm |

完整根轨迹、脚滑、连续性和特征误差见 [real_reconstruction.json](evidence/real_reconstruction.json)。
这是 teacher-history 重建；actor 是目标槽迁移使用。量化重建明显变差，不能宣称量化无损或双方向能力已验证。
正式 diffusion 生成没有使用这些量化 latent，因此上述误差不是生成质量回归。

随机小型 T5 的 val NLL=10.73349。测试样本五种输入的有效描述比例和词面指标均为 0；
它尚未学会有效描述。这是接口与错误标记的检查结果，不是完整训练后的语义结论。
见 [描述](evidence/real_description.json)、[消融评估](evidence/real_evaluation/report.json) 和
[测试集重建](evidence/real_evaluation/reconstruction.json)。
动作、角色、先后关系错误字段留给人工审核；当前没有人工错误率。公平 K=1 语言基线还需单独训练。

## 复现

在项目根目录使用上述解释器；创建新的独立临时目录，禁止覆盖正式数据或历史实验。
以下是本次小规模流程（`python` 指上述解释器）：

```bash
python -m pytest -q
python -m moreact semantics cache-latents --checkpoint runs/diffusion_restart_b512_20260926_013059/step_001000.pt --cache data/interx_h2_f8 --output tmp/shared_rvq_validation/real_latents --limit 1
python -m moreact semantics fit-rvq --cache tmp/shared_rvq_validation/real_latents --output tmp/shared_rvq_validation/real_rvq --epochs 2 --batch-size 32
python -m moreact semantics fit-rvq --cache tmp/shared_rvq_validation/real_latents --output tmp/shared_rvq_validation/real_rvq_k1 --epochs 2 --batch-size 32 --levels 1
python -m moreact semantics cache-tokens --cache tmp/shared_rvq_validation/real_latents --rvq tmp/shared_rvq_validation/real_rvq/best.pt --output tmp/shared_rvq_validation/real_tokens
python docs/reports/20260926_shared_rvq/evidence/real_checks.py
python -m moreact semantics train --cache tmp/shared_rvq_validation/real_tokens --output tmp/shared_rvq_validation/tiny_captioner --base-model tmp/shared_rvq_validation/tiny_language_base --steps 2 --validate-every 1
python -m moreact semantics describe --motion-checkpoint tmp/shared_rvq_validation/real_latents/vae_snapshot.pt --rvq tmp/shared_rvq_validation/real_rvq/best.pt --checkpoint tmp/shared_rvq_validation/tiny_captioner/best.pt --input data/interx_h2_f8/episodes/G001T000A000R000.npz --output tmp/shared_rvq_validation/real_description.json
python -m moreact semantics evaluate --cache tmp/shared_rvq_validation/real_tokens --rvq tmp/shared_rvq_validation/real_rvq/best.pt --checkpoint tmp/shared_rvq_validation/tiny_captioner/best.pt --output tmp/shared_rvq_validation/real_evaluation --limit 1 --motion-checkpoint tmp/shared_rvq_validation/real_latents/vae_snapshot.pt --motion-cache data/interx_h2_f8
```

`real_checks.py` 为归档的一次性复现证据，包含真实前向及随机小模型的配置，不是工程公共入口。
真实 FLAN 前向的新增 embedding 初始化没有固定种子，复现 loss 可以变化；检查有限性而非数值完全相等。
正式完整训练与评估流程见主规范，本次未启动。

## 归档和清理

本次唯一待清理目录为 `tmp/shared_rvq_validation`。日志、manifest、配置、指标和一次性复现脚本
已存入本报告的 evidence；[evidence_manifest.json](evidence_manifest.json) 记录每项大小和 SHA-256。
没有将临时权重或 latent/token 数组重复复制到文档目录。

删除前核对报告证据、活动进程命令/工作目录/打开文件以及正式配置引用。最终清理状态和字节数见
[cleanup.json](cleanup.json)。历史证据里的临时路径用于溯源，不能作为可恢复训练资产；复现需重新运行命令。

清理完成：删除本次临时目录中的 547 个文件，合计 95,948,894 bytes（约 91.5 MiB）；25 项归档证据哈希核对通过。
未发现可访问进程或正式配置对该目录的引用；其他用户/内核进程的部分 cwd/fd 不可读，检查范围记录于 cleanup.json。
静态语法、可选依赖隔离和文档检查见 [static_checks.json](static_checks.json)。
