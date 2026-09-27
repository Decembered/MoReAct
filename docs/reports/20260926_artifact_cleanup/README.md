# 2026-09-26 历史临时产物归档与清理

状态：已先归档并校验轻量证据，再按精确清单完成删除。删除前检查了同用户进程的命令、工作目录、可读取的打开文件及保留实验元信息引用；未发现目标仍被占用或引用。

## 范围与依据

本轮整理目录规范；清理已完成的 smoke、预检、固定样本过拟合及 DDP benchmark。正式训练、正式数据、模型对比结果和曲线继续保留。
归档原配置、指标、日志和数据准备元信息；不复制大型 checkpoint、数据片段或视频。文件大小、SHA256 和删除状态见 [manifest.json](manifest.json)。

## 保留的结果

- 原 200-step smoke 和 fixed-batch overfit 的历史结论见 [验证记录](../../VALIDATION.md)，不代表收敛或泛化质量。
- 两卡启动/恢复及 best checkpoint 测试保留原日志和配置；本轮没有重新执行 GPU smoke。
- DDP benchmark 的单卡/四卡 step 中位数为 1.8605/0.7032 秒，约 2.65 倍；保留 [原始结果](evidence/runs/ddp_benchmark/result.json)。
- Stage 2 checkpoint 比较保留 [原始指标](evidence/runs/stage2_preflight/checkpoint_comparison.json)。

## 清理清单

| 原始路径 | 文件数 | 字节数 |
| --- | ---: | ---: |
| `runs/ddp_smoke` | 19 | 647577124 |
| `runs/smoke_vae` | 4 | 5043509 |
| `runs/smoke_diffusion` | 4 | 3893361 |
| `runs/stage2_preflight` | 9 | 323815250 |
| `runs/full_size_check` | 4 | 140640373 |
| `runs/restart_best_smoke` | 15 | 1295161254 |
| `runs/full_size_diffusion_check` | 4 | 323760606 |
| `runs/diffusion_geometry_preflight` | 2 | 1946 |
| `runs/real_overfit_vae` | 4 | 5043391 |
| `runs/ddp_benchmark` | 23 | 827158729 |
| `data/interx_smoke` | 28 | 9011892 |
| `data/interx_smoke_h2_f8` | 28 | 9053565 |
| `outputs/smoke_preview` | 7 | 459808 |
| `outputs/smoke_vae_eval` | 1 | 3467 |
| `outputs/smoke_eval` | 61 | 1774592 |
| `outputs/tests_stage2.log` | 1 | 100 |
| `outputs/tests_equal_roles.log` | 1 | 99 |
| `outputs/tests_distributed.log` | 1 | 100 |
| `outputs/tests_best_checkpoint.log` | 1 | 100 |
| `outputs/tests_diffusion_geometry.log` | 1 | 100 |
| `outputs/tests_ddp_mask.log` | 1 | 98 |
| `.pytest_cache` | 5 | 2254 |
| `moreact/__pycache__` | 28 | 151734 |
| `scripts/__pycache__` | 1 | 3256 |
| `tests/__pycache__` | 14 | 64070 |

合计候选大小：3.346 GiB。归档证据保存在 `evidence/`，使用原项目相对目录结构。

## 复现与限制

当前可维护的工程检查入口为项目根目录的 `bash scripts/smoke.sh`。历史 H=16 smoke 使用过旧 loss 配方，当前 H=2 配方不等于历史实验的逐位复现。
旧 config/metadata 中的运行路径仅供溯源；清理后不能直接使用对应 checkpoint 或视频。DDP、预检和 overfit 报告是过去的验证证据，本轮没有重新运行这些实验。

## 本轮维护检查

已删除 25 项临时目录/文件、267 个文件，原文件合计 3,592,620,778 字节（约 3.35 GiB）；保留证据不到 1 MiB。实际磁盘释放量受文件系统分配影响。

smoke 生命周期模拟测试：3 passed，覆盖成功归档后清理、失败保留、归档失败不删除；未启动 GPU smoke。

活跃正式训练未被终止或移动，正式数据缓存、训练权重及正式评估产物保留。进程检查见 [process_audit.json](process_audit.json)。

最终检查：11 份 Markdown 文档及 36 个本地链接有效；新增脚本/测试符合 Python 3.8 语法；
`bash -n scripts/smoke.sh` 通过。删除清单中的路径均已不存在，指定正式运行目录仍存在。
训练持续写入新 checkpoint，因此清理前后项目总大小差值不等于本次删除的文件字节总量。
