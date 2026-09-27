# MoReAct 残差语义训练与 bridge 对比

2026-09-26，用户确认在 MoReAct 正式训练共享 CVAE + 两级 RVQ 语义模块，并与 bridge 对比。
当前为已启动实验，质量结论以正式输出为准，不把启动或预检当作收敛。

## 配方与资产

- 冻结源：`runs/diffusion_stage2_b512_20260926_084433/step_007000.pt` 内嵌 CVAE；提取时保存独立 snapshot 与 SHA256。
- 数据：`data/interx_h2_f8`，train=9110 / val=570 / test=1707；不限制样本数。
- latent：`data/semantics_rvq_20260926/latents`，双角色确定性 mu，H=2 / F=8 / 30 FPS。
- RVQ：K=2，每层256码，EMA decay=.99，batch=1024，seed=42，20 epochs；仅 train 更新，按 val 平均 MSE 选 best。
- Captioner：本地原始 FLAN-T5-large，FP32 AdamW lr=1e-4，无 warmup，micro-batch=1，accumulation=16，gradient checkpointing，seed=42。
- 首轮100个 optimizer updates（1600个有放回抽样训练样本），运行共同验证集对比；继承 optimizer/RNG 继续到5000 updates（共80000样本）。每100步全量 val NLL 验证，按 NLL 保存 best。
- 更新前审计所有 split 的长度，无静默截断；最大输入与最大目标组合做两次 GPU optimizer capacity check，丢弃该预检模型。预检目标包含所有 split 仅用于显存检查，其权重不进入训练。
- 设备：缓存、RVQ 在 CPU 4 threads；文本模型用 GPU5，主训练等空闲显存至少18000 MiB。不会停止已有作业。共享设备吞吐会变化，预检失败写 status 并停止后续流程。

正式配方：[semantics_rvq_20260926.json](../../../configs/semantics_rvq_20260926.json)。
编排入口 `scripts/run_semantics_training.py`，状态 `runs/semantics_rvq_20260926/status.json`。
每阶段保存独立日志；失败不会自动用较小数据、随机模型或其他权重替代。
编排不支持整条流程盲目重复运行；失败后根据状态使用单阶段 CLI，RVQ/文本恢复使用原目录 last.pt。

## 比较协议

`outputs/semantics_rvq_20260926/comparison/selection.json` 在训练前固定64条共同 val 样本。
按类别轮询、类内 SHA256(seed+episode) 排序；只保留两边角色顺序一致的样本。
这是一轮开发集比较，不是最终 test 指标。全量 test 尚未安排为本轮终止条件。

Bridge 基线为 `ttr_remogen_bridge/artifacts/interaction_rvq_v5_flan_large/train_4gpu/best_text.pt`，
复用其已编码完整动作、原 pair_m2t prompt 和原 loader，记录 checkpoint SHA256。
两边使用相同 MoReAct 多参考文本、word-F1/ROUGE-L-F1 和 valid fraction；greedy、max_new_tokens=512。
保存逐样本描述，便于人工检查动作、角色及先后关系；词面指标不等于语义正确率。
MoReAct 另外评估 base_only、shuffle、remove_actor、remove_reactor；base_only 是推理消融，不是独立K=1训练。

Bridge 是八任务训练，MoReAct 是完整交互描述单任务，训练预算也不同，因此比较的是当前原生系统，
不能将差异单独归因于残差表示。原始 bridge 缓存标注10 FPS、MoReAct标注30 FPS，已抽查同一episode帧数相同，
但名义时长不同；没有完成原始采样时间审计，禁止据此比较真实时间分辨率和时序理解优劣。

输出：baseline `bridge.json`；100步选优模型 `moreact.json` 和 `comparison.json`；
5000步选优模型在相邻 `comparison_final/`。两次评估都明确保存被加载 checkpoint 的哈希。
Bridge baseline 独立运行，与 CPU cache/RVQ 并行；其失败日志曾出现 import 路径错误，已增加 bridge/src 后重启，保留原日志。

## 验证

语义模块9项测试通过（2.48秒）；新增梯度累积/检查点恢复路径复用连续训练与中断恢复逐值一致性测试。
命令：`python -m pytest -q tests/test_semantics.py`。正式文本训练尚未产出指标前不宣称质量提升。

## 已完成的 bridge 对照

64条固定共同验证样本：word-F1=0.5011735，ROUGE-L-F1=0.4306416，valid_fraction=1.0。
仅代表当前 bridge 基线，不是 MoReAct 对比结论。权重及选择清单哈希见 [baseline摘要](bridge_baseline_summary.json)，逐样本输出保留于上述正式 outputs 目录。

## 已完成 RVQ 拟合

20轮完成；最优第19轮，验证双人均值 MSE=0.51965581，actor=0.50270895，reactor=0.53660267。
最优码本在验证集使用数=[215, 149]，perplexity=[159.31636197509684, 72.73609042244559]。仅为量化指标，不能跨 latent 表示与 bridge 直接比较 MSE。
详见 [RVQ指标](rvq_metrics.json)。

## 文本训练预检

11387条完整样本全量长度审计通过，错误0；最大输入3491 tokens、最大目标208 tokens。
最长输入/目标组合的2次FP32 AdamW更新通过，峰值allocated显存17,760,094,208 bytes（约16.54 GiB）。
预检权重已随进程退出丢弃。正式FLAN-T5-large训练进程已启动，详见运行状态及captioner/metrics.jsonl。
