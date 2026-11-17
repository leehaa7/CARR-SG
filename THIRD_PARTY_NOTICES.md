# 来源与许可

本预览从 CARR-SG 的 PEMS Routed 实现提取。历史代码仓库为 [SDGL-SeqDe](https://github.com/songbaizhu/SDGL-SeqDe)。本预览使用的源文件及对应关系如下。

| 源文件 | SHA-256 | 预览文件 |
|---|---|---|
| `graph_distribution.py` | `d7b4ee6534af4c41eda3ab4c9c08f0d28a902140ba58ead02f7748233a90ce14` | `core/graph_learning.py` |
| `model_distribution.py` | `85998b7aff9cf602f22a91f17c9ae516627da5dbf158b32b8e87765e0c107dc2` | `core/graph_learning.py`、`core/dgcsd.py` |

预览对入口和参数封装作了适配，只提取这两个模块使用的计算。两张说明图来自 CARR-SG 稿件的 Fig. 2 和 Fig. 3。[PyTorch](https://github.com/pytorch/pytorch) 通过依赖安装。

代码沿用历史仓库的 [Apache-2.0](https://github.com/songbaizhu/SDGL-SeqDe)，仓库中的 [LICENSE](LICENSE) 复制自历史实现。该许可证允许使用、修改和再分发代码。再分发时须保留许可证与适用的署名和通知，并标明对文件的修改。完整权利和义务见 [Apache Software Foundation 许可证原文](https://www.apache.org/licenses/LICENSE-2.0)。

