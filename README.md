# CARR-SG

**Complementarity-Aware Representation Routing with Shared Graph Guidance for Multivariate Time Series Forecasting**

论文目前正在审稿。本仓库提供共享图与上下文图的构建，以及双图卷积与空间分解（DGCSD）的 PyTorch 实现。

## 共享图与上下文图

传感器之间既有跨时间窗相对稳定的联系，也有随当前交通状态变化的联系。共享图提供跨窗口通用的关系参照，上下文图补充当前窗口的局部关系。

1. **共享图来自节点嵌入。** 模型在训练中为每个传感器学习一份节点嵌入，用嵌入之间的相似度构图并做行归一化。同一组模型参数下，各输入窗口使用同一张共享图。
2. **上下文图来自当前历史窗口。** 先用共享图传播预处理后的历史数据，输入减去传播值后得到互补偏差。上下文图以这部分偏差为主要依据，用一阶时间差分调节共享节点信息的贡献，再根据所得节点表示计算边权。

[Fig. 2 · 上下文图构建（PDF）](assets/context_graph.pdf) · [构图代码](core/graph_learning.py)

## DGCSD 如何使用两种图

构图后，DGCSD 接收两种图和待传播的隐藏表示。它用共享图将隐藏表示分成**共享图传播分量**与**互补分量**。前者沿共享图传播，后者沿上下文图传播。最后汇合各阶传播结果，并与原始隐藏表示一起投影，作为后续层的输入。

[Fig. 3 · 双图卷积与空间分解（PDF）](assets/DGCSD.pdf) · [传播代码](core/dgcsd.py)

| 模块 | 输入 | 输出 |
|---|---|---|
| `DualGraphLearner` | 历史张量 `[B, C, N, T]` | 共享图 `[N, N]`，每个样本的上下文图 `[B, N, N]` |
| `DGCSD` | 隐藏张量 `[B, C_in, N, T]` 和两种图 | 更新后的隐藏特征 `[B, C_out, N, T]` |

`B` 为样本数，`C` 为通道数，`N` 为节点数，`T` 为时间步数。

## 运行演示

在仓库根目录运行。默认使用小规模合成输入，可直接在 CPU 上检查两个模块的调用。

```bash
python -m pip install -r requirements.txt
python demo.py
```

也可用 [PEMS04.npz](https://github.com/guoshnBJTU/ASTGNN/blob/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS04/PEMS04.npz) 演示。下载数据后指定文件路径。

```bash
python demo.py --pems04 path/to/PEMS04.npz
```

[示例](demo.py)读取开头两个不重叠的 12 步流量窗口，对每个节点在各自窗口内标准化，再构建一张共享图和每窗一张上下文图。为展示传播接口，它直接用标准化后的流量作为单通道节点特征。程序展示两张上下文图的差异、两个分支的一阶传播及 DGCSD 的输出形状。示例使用真实流量和随机初始化的参数（固定种子），演示到中间特征为止。

## 代码可用性

完整实现，包括训练与评估脚本、实验配置和运行说明，将在论文录用后于本仓库公开。代码采用 [Apache-2.0](LICENSE) 许可。
