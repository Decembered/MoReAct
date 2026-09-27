# 文档索引

| 文档 | 用途 |
| --- | --- |
| [项目入口](../README.md) | 当前可运行能力、安装、训练与生成命令 |
| [资源、资产配套与训练总览](RESOURCE_AND_TRAINING_GUIDE.md) | 本机绝对路径、checkpoint 配对、训练恢复、推理、语义理解与 store 交接 |
| [机器可读资产清单](RESOURCE_ASSETS.json) | 已核实的资产绝对路径与存在状态 |
| [架构](ARCHITECTURE.md) | 当前模块与目标语义闭环、接口、时序及待实现能力 |
| [共享 CVAE/RVQ 规范](SHARED_RVQ.md) | 已实现的连续生成、冻结均值编码、K=2 理解和完整命令流程 |
| [维护规则](MAINTENANCE.md) | 文件归属、临时实验、归档和删除流程 |
| [验证记录](VALIDATION.md) | 历史工程检查和质量边界 |
| [扩散损失](DIFFUSION_LOSSES.md) | 可选几何损失的定义与权重 |
| [分布式训练](DISTRIBUTED_TRAINING.md) | DDP、恢复、checkpoint 和性能口径 |
| [实验报告索引](reports/README.md) | 已归档实验与清理记录 |
| [早期在线闭环设想](history/ONLINE_SEMANTICS_DESIGN.md) | 历史设计，不代表当前在线能力 |

文档中的命令默认在项目根目录执行。架构行为改变时更新主文档；一次性结果写入
`docs/reports/`，不在根目录继续新增实验笔记。已删除产物的路径仅用于溯源，不表示文件仍可用。
