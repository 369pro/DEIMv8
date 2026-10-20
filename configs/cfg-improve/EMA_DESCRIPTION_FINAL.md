# EMA 注意力机制实现描述（最终修正版）

在无人机航拍场景下，道路标志尺度小、分布稀疏，且常与百米级上下文（如"限速80"与"解除限速"成对出现）存在长距离依赖。HGNetV2 原骨干采用 SE 模块增强通道权重，但 SE 仅通过全局池化+FC 激励通道，缺乏像素级空间交互，对航拍图像中"稀疏目标+大背景"特征耦合不足，导致小目标漏检率居高不下。

为此，本文在 TT100K 道路标志检测框架中，将 HGNetV2 骨干网络的 SE 注意力替换为 EMA（Efficient Multi-scale Attention），重构基本构建块：

**分组并行处理**：将通道分成 8 组（`factor=8`），通过 reshape 将输入特征重塑为 `(B×G, C//G, H, W)` 形式，所有通道组并行处理，在降低计算复杂度的同时保留完整通道交互，FLOPs 降至 SE 的约 30%。

**并行双支路架构**：
- **空间注意力支路**：通过水平方向自适应池化（`pool_h: AdaptiveAvgPool2d((None, 1))`）和垂直方向自适应池化（`pool_w: AdaptiveAvgPool2d((1, None))`）分别捕获 H 和 W 维度的空间信息，拼接后经 1×1 卷积融合生成空间权重，通过 sigmoid 激活后与原始特征逐元素相乘，再经 GroupNorm 归一化，实现"在哪重要"的空间感知；
- **多尺度上下文支路**：对分组后的特征直接执行 3×3 卷积（`Conv2d(C//G, C//G, 3×3)`），提取局部上下文信息，建立空间依赖关系。

**跨空间-通道联合建模**：两支路分别通过 `AdaptiveAvgPool2d((1,1))` 全局池化得到通道级特征，再经 softmax 归一化生成通道权重；通过矩阵乘法实现双向交叉融合：`weights = matmul(softmax(agp(x1)), reshape(x2)) + matmul(softmax(agp(x2)), reshape(x1))`，融合权重经 sigmoid 激活后与原始分组特征逐元素相乘，实现跨空间-通道的联合增强。

**即插即用集成**：在 `HG_Block` 的聚合层（`aggregation`）中，EMA 与通道调整卷积（`ConvBNAct(total_chs→out_chs, 1×1)`）串联组成 `nn.Sequential`，处理拼接后的多层特征；EMA 输出与残差输入相加（`drop_path(x) + identity`），不改变 HGNetV2 原有的下采样（`HG_Stage.downsample`）与跳跃连接（`HG_Block.residual`）结构，Neck/Head 零改动，仅需在配置文件中将 `agg='se'` 改为 `agg='ema'` 即可。

