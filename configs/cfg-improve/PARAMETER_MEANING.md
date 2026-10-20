# dfine-n-C2f-IDWB-MFM-LSK-lite.yaml 参数详解

## 📋 配置文件结构

该配置文件定义了DFINE目标检测网络的架构，包含三个主要部分：
- **Backbone**: 特征提取网络
- **Encoder**: 特征编码和增强网络  
- **Decoder**: 多尺度特征融合和解码网络

---

## 1️⃣ Backbone (特征提取网络)

### 基本格式
```yaml
[from, module, args]
```
- **from**: `-1`表示使用上一层输出，数字表示使用backbone的特定层
- **module**: 模块名称
- **args**: 模块参数列表

### 模块详解

#### StemBlock (初始特征提取层)
```yaml
[-1, StemBlock, [16, 16, True]]
```
- **参数**: `[输入通道, 输出通道, use_conv]`
- **含义**: 初始16通道输入，输出16通道特征

#### HG_Stage (HGNet阶段)
```yaml
[-1, HG_Stage, [输入通道, 输出通道, stride, kernel, use_SE, use_CBAM, blocks, use_DCM, attention]]
```

**参数说明**:
- **输入通道**: 输入特征图的通道数
- **输出通道**: 输出特征图的通道数
- **stride**: 步长 (1=保持尺寸，2=下采样)
- **kernel**: 卷积核大小
- **use_SE**: 是否使用SE注意力 (True/False)
- **use_CBAM**: 是否使用CBAM注意力 (True/False)
- **blocks**: 重复块的数量
- **use_DCM**: 是否使用DCM (True/False)
- **attention**: 注意力类型 ('se', 'cbam' 等)

**示例**:
```yaml
# P3/8阶段：32→256通道，使用SE注意力
[-1, HG_Stage, [32, 256, 1, 3, True, False, 3, True, 'se']]
```

---

## 2️⃣ Encoder (编码器)

### PSConv (极化自注意力卷积)
```yaml
[2, PSConv, [112, 1, 1]]
# from=2, 模块=PSConv, args=[输出通道, groups, kernel_size]
```

**参数说明**:
- **from=2**: 使用backbone第2层输出
- **输出通道**: 112 (特征维度，从原始160降低到112)
- **groups**: 1 (分组卷积的组数)
- **kernel_size**: 1 (卷积核大小)

### TransformerEncoderBlock (Transformer编码器)
```yaml
[-1, TransformerEncoderBlock, [8, 448, 0.1, "relu", 10000, False]]
```

**参数说明**:
- **num_heads**: 8 (注意力头数)
- **hidden_dim**: 448 (隐藏层维度，从640降低)
- **dropout**: 0.1 (dropout率)
- **activation**: "relu" (激活函数)
- **max_seq_len**: 10000 (最大序列长度)
- **use_bias**: False (是否使用偏置)

### MFM (多尺度特征调制)
```yaml
[[-1, 6], MFM, [112]]
```

**参数说明**:
- **from**: `[-1, 6]` - 融合当前层(-1)和第6层的特征
- **输出通道**: 112

### LSKBlock (大尺度可分离卷积块)
```yaml
[-1, LSKBlock, []]
```

**参数说明**:
- **from**: `-1` (使用上一层输出)
- **无参数**: 使用默认配置
- **作用**: 增强空间特征提取能力

### C2f_Block (C2f模块块)
```yaml
[-1, C2f_Block, [112, {'module':InceptionDWBlock, 'param':{'mlp_ratio': 3.0}}, 1]]
```

**参数说明**:
- **输出通道**: 112
- **block_config**: 
  - `module`: InceptionDWBlock (使用Inception深度可分离卷积)
  - `param`: `{'mlp_ratio': 3.0}` (MLP扩展比例，从4.0降低到3.0)
- **repeats**: 1 (重复次数，从原始2次降低到1次)

### SCDown (自校准下采样)
```yaml
[-1, SCDown, [112, 3, 2, 'silu']]
```

**参数说明**:
- **输出通道**: 112
- **kernel_size**: 3 (卷积核大小)
- **stride**: 2 (步长，实现2倍下采样)
- **activation**: 'silu' (激活函数)

### nn.Upsample (上采样)
```yaml
[-1, nn.Upsample, [None, 2, "nearest"]]
```

**参数说明**:
- **None**: scale_factor参数位置
- **2**: 上采样倍数 (2倍)
- **"nearest"**: 插值方式 (最近邻)

---

## 3️⃣ Decoder (解码器)

### DFINETransformer
```yaml
[[18, 22, 26], DFINETransformer, {...}]
```

**from参数**: `[18, 22, 26]`
- 使用encoder第18、22、26层的输出
- 分别对应P3/8、P4/16、P5/32三个尺度

**字典参数**:
- **feat_strides**: `[8, 16, 32]` - 特征步长，对应下采样倍数
- **hidden_dim**: `112` - 隐藏层维度 (从原始160降低)
- **num_levels**: `3` - 多尺度层数
- **num_layers**: `3` - Transformer层数
- **num_points**: `[2, 5, 2]` - 各尺度的参考点数 (P3有2个，P4有5个，P5有2个)
- **dim_feedforward**: `448` - FeedForward网络维度 (从640降低)

---

## 📊 关键优化参数对比

| 参数 | 原始值 | Lite版本 | 变化 |
|------|--------|----------|------|
| 特征维度 | 160 | 112 | -30% |
| Transformer hidden | 640 | 448 | -30% |
| dim_feedforward | 640 | 448 | -30% |
| C2f repeats | 2 | 1 | -50% |
| mlp_ratio | 4.0 | 3.0 | -25% |
| num_points | [3,6,3] | [2,5,2] | -17% |

---

## 🎯 网络流程

1. **Backbone**: 提取多尺度特征 (P2/4, P3/8, P4/16, P5/32)
2. **Encoder**: 
   - 通过PSConv调整特征维度
   - 使用Transformer增强P5特征
   - 通过上采样和MFM融合不同尺度特征
   - 使用LSKBlock和C2f_Block增强特征
3. **Decoder**: 使用DFINETransformer进行多尺度融合和检测


