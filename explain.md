# DEIM 训练数据预处理流水线

> 从原始图像到模型输入的完整流程

---

## 第一阶段：数据集初始化（一次性的）

发生在 `DataLoader` 创建时，`CocoDetection.__init__`（[coco_dataset.py:264-272](engine/data/dataset/coco_dataset.py#L264-L272)）：

1. **pycocotools 加载 COCO JSON 标注文件**，建立图像 ID → 标注的索引
2. **可选 RAM Cache**（默认关闭）：
   - 配置 `ram_cache: False, cache_imgsz: 1024`（[runtime.yml:22-23](configs/runtime.yml#L22-L23)）
   - 如果开启，会用线程池把所有图片预加载到内存，缩放到 `cache_imgsz` 并同步缩放 bbox。目前是关的，所以每轮 epoch 都从磁盘读

---

## 第二阶段：每样本 `__getitem__` → `load_item`

每次迭代取一张图，走 [coco_dataset.py:274-298](engine/data/dataset/coco_dataset.py#L274-L298)：

### 2a. 从磁盘加载原始数据

```python
image, target = super().__getitem__(idx)
```

- `image`：PIL.Image，`Image.open().convert("RGB")`，值域 0-255，格式 HWC
- `target`：COCO 原始标注列表（dict 列表，含 `bbox`, `category_id`, `area`, `iscrowd` 等）

### 2b. ConvertCocoPolysToMask（标注格式化）

[coco_dataset.py:343-410](engine/data/dataset/coco_dataset.py#L343-L410)：

| 处理 | 说明 |
|------|------|
| **bbox 格式转换** | COCO 的 `[x, y, w, h]` → `[x1, y1, x2, y2]`（`boxes[:, 2:] += boxes[:, :2]`） |
| **bbox 裁剪** | `clamp_(min=0, max=w/h)`，确保不越界 |
| **类别 ID 映射** | 如果 `remap_mscoco_category=True`，把 COCO 的 category_id 映射为 0~79 的连续标签 |
| **过滤无效框** | 筛掉 x2≤x1 或 y2≤y1 的框 |
| **附加元数据** | `orig_size` = `[w, h]`（原始宽高）、`area`、`iscrowd`、`image_id` |

### 2c. 包装为 TVTensor

把 bbox 包装成 `torchvision.tv_tensors.BoundingBoxes`，带上 `format=XYXY, spatial_size=(H,W)` 元信息（[coco_dataset.py:292-293](engine/data/dataset/coco_dataset.py#L292-L293)），后续 torchvision transform 靠这个元信息自动处理坐标变换。

---

## 第三阶段：在线数据增强（Transforms）

`self._transforms` 即 `Compose`，实际执行的 ops 来自 [dataloader_dfine.yml:8-18](configs/base/dataloader_dfine.yml#L8-L18)：

```
RandomPhotometricDistort  ─── 光度扰动
    ↓
RandomZoomOut             ─── 缩放填充
    ↓
RandomIoUCrop             ─── 随机裁剪
    ↓
SanitizeBoundingBoxes     ─── 过滤小框
    ↓
RandomHorizontalFlip      ─── 随机翻转
    ↓
Resize                    ─── 拉伸到 1024×1024
    ↓
SanitizeBoundingBoxes     ─── 再次过滤小框
    ↓
ConvertPILImage           ─── PIL → Tensor(float32), ÷255
    ↓
ConvertBoxes              ─── xyxy → cxcywh, 归一化到 [0,1]
```

其中有一个**基于 epoch 的策略调度**（[container.py:74-108](engine/data/transforms/container.py#L74-L108)）：

```yaml
policy:
  name: stop_epoch
  epoch: 72        # 前72轮用全套增强，之后关掉RandomPhotometricDistort/RandomZoomOut/RandomIoUCrop
  ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
```

前 72 个 epoch 用全套增强，72 轮之后去掉光度扰动/缩放填充/随机裁剪（这是 2 阶段训练的分界点，`stop_epoch` 同时也是多尺度和 mixup 的关闭点）。

### 各变换详情

| 变换 | 行为 |
|------|------|
| **RandomPhotometricDistort** | 随机调整亮度、对比度、饱和度、色调，p=0.5 |
| **RandomZoomOut** | 把原图随机贴到一张更大的纯黑画布上（fill=0），模拟小目标 |
| **RandomIoUCrop** | 随机裁剪并与原始框计算 IoU，p=0.8，保留与 GT 框重叠足够多的裁剪区域 |
| **RandomHorizontalFlip** | 50% 概率水平翻转，bbox 坐标同步翻转 |
| **Resize** | 直接拉伸/压缩到 `[1024, 1024]`，**不保持长宽比** |
| **ConvertPILImage** | `F.pil_to_tensor()` → `uint8` 转 `float32` → `÷255` 到 `[0,1]` |
| **ConvertBoxes** | `xyxy` → `cxcywh`，再除以宽高归一化到 `[0,1]` |

---

## 第四阶段：Batch Collation

`BatchImageCollateFunction.__call__`（[dataloader.py:180-199](engine/data/dataloader.py#L180-L199)）：

1. **Stack 图像**：`torch.cat([x[0][None] for x in items], dim=0)` → shape `[B, 3, 1024, 1024]`
2. **可选 MixUp**（默认关闭，`mixup_prob=0.0`）：将 batch 中图像按 beta 分布混合，标签合并
3. **可选多尺度**（`stop_epoch=72` 内有效）：从预生成尺度列表 `[768, 800, ..., 1024, 1024, ..., 1248, 1216]` 中随机选一个，用 `F.interpolate` 缩放整个 batch

---

## 最终进入模型的数据

```python
images:  Tensor,  shape [B, 3, 1024, 1024],  dtype float32, 值域 [0, 1]
targets: list of dict, 每个 dict 包含:
  ├── boxes:      Tensor [N, 4],  cxcywh 格式, 归一化到 [0,1]
  ├── labels:     Tensor [N],     int64,  类别 ID
  ├── image_id:   Tensor [1],     int64
  ├── orig_size:  Tensor [2],     原始宽高 [w, h]
  ├── area:       Tensor [N],     float32
  ├── iscrowd:    Tensor [N],     uint8
  └── idx:        Tensor [1],     索引
```

**不做 Normalize（无减均值除方差），不做 BGR 转换，直接 RGB [0,1] 进模型。**

---

## Mosaic 增强（可选，当前配置未启用）

Mosaic 把 4 张图拼成 2×2 网格，定义在 [mosaic.py](engine/data/transforms/mosaic.py)：

```
┌──────┬──────┐
│ img0 │ img1 │
├──────┼──────┤
│ img2 │ img3 │
└──────┴──────┘
```

- 当前样本 + 随机抽 3 张 → 缩放到 320px → 拼成大图 → RandomAffine
- 与 RandomZoomOut/RandomIoUCrop **互斥**（不能同时用）
- 通过 `mosaic_prob` 控制概率，基于 epoch 调度决定启用/关闭

---

## 推理时的预处理（简化版）

```python
transforms = T.Compose([
    T.Resize((1024, 1024)),          # 拉伸到 1024×1024
    T.ToTensor(),                     # PIL → Tensor, ÷255
])
```

只有 **2 步**：Resize + ÷255，**无增强、无 Normalize**。
