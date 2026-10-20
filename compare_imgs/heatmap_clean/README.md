# DEIMv8 真实热力图（2026-09-09）

优先查看 `M6_heatmap_2x2.jpg`；对照图是 `M1_vs_M6_layercam.jpg`。

这些图片由项目内的 M1、M6 真实 checkpoint 前向计算和梯度生成，未人工添加热点。使用 1024×1024 输入、P4 主干特征层、置信度至少 0.25 的最多 15 个检测 query，每个 query 选择其最高概率类别，其 logits 均值作为梯度目标。query 索引直接来自 decoder 输出，未通过错误坐标匹配推断。

- M1：`outputs/dfine_hgnetv2_n_custom/best_stg2.pth`，严格匹配 SE 架构，层 `backbone.stages.2`。
- M6：`outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth`，严格匹配项目配置，层 `backbone.3`。
- LayerCAM：`ReLU(sum(ReLU(gradient) * activation))`。
- Grad-CAM：空间平均梯度作为通道权重，再加权激活并 ReLU。
- 每张热图按自身 99.5 百分位缩放到 [0,1]；相同色彩及透明度函数。这是相对空间响应，不能用红色面积/深浅比较不同模型的绝对强度。
- 局部放大区域由同一张图片的 GT 框联合范围确定，对两个模型一致。GT 没有用于遮罩或制造热图。
- 各模型根据自身预测选择目标 query，因此不是固定同一个检测目标的严格因果比较。
- M1 与 M6 同时包含多个结构差异，不能称为单独 EMA 或 LSKNet 的消融证明，也不能仅凭热图判断 M6 更优。

远端结果目录：`/home/scnu2024024530/WZW/DEIMv8/compare_imgs/heatmap_clean/`。
远端可复现脚本：`heatmap_clean_20260909.py`。保留 raw NPZ 和 metadata.json（检测 query、类别、得分、层、权重路径、缩放值）。

旧脚本 `heatmap_compare.py` 有像素 xyxy 与归一化 cxcywh 混比、BGR/RGB 混用及色条不匹配等问题。本次使用独立脚本，未覆盖旧结果。

另外，旧 `batch_compare.py` 的 M1 是由 M3 检测人工删减并重写置信度生成（见 degrade_dets/_make_all_dets），不是真实 baseline 推理；旧参考图不应作为 M1 模型实验结果使用。本次 M1 已换成真实 checkpoint。
