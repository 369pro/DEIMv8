"""
EMA 注意力参数传递流程示例代码
这个文件展示了从 YAML 配置到实际创建 EMA 模块的完整过程
"""

# ============================================
# 1. YAML 配置文件中的定义
# ============================================
yaml_config = {
    "backbone": [
        [-1, "HG_Stage", [16, 64, 1, 3, False, False, 3, True, 'ema']]
    ]
}

# 模拟参数列表
yaml_args = [16, 64, 1, 3, False, False, 3, True, 'ema']
print("=" * 60)
print("步骤 1: YAML 配置参数")
print("=" * 60)
print(f"参数列表: {yaml_args}")
print(f"最后一个参数 (agg): {yaml_args[-1]}")
print()


# ============================================
# 2. tasks.py 中的 parse_module 函数处理
# ============================================
def simulate_parse_module_for_hgstage(args, ch_f):
    """
    模拟 tasks.py 第 212-214 行的处理逻辑
    """
    print("=" * 60)
    print("步骤 2: tasks.py - parse_module 处理")
    print("=" * 60)
    print(f"原始 args: {args}")
    
    # 模拟: c1, cmid, c2 = ch[f], args[0], args[1]
    c1 = ch_f  # 从前一层获取的通道数（假设是 16）
    cmid = args[0]  # 16
    c2 = args[1]    # 64
    
    # 模拟: args = [c1, cmid, c2, *args[2:]]
    new_args = [c1, cmid, c2, *args[2:]]
    
    print(f"处理后的 args: {new_args}")
    print(f"参数含义:")
    print(f"  - args[0] (in_chs): {new_args[0]}")
    print(f"  - args[1] (mid_chs): {new_args[1]}")
    print(f"  - args[2] (out_chs): {new_args[2]}")
    print(f"  - args[3] (block_num): {new_args[3]}")
    print(f"  - args[4] (layer_num): {new_args[4]}")
    print(f"  - args[5] (downsample): {new_args[5]}")
    print(f"  - args[6] (light_block): {new_args[6]}")
    print(f"  - args[7] (kernel_size): {new_args[7]}")
    print(f"  - args[8] (use_lab): {new_args[8]}")
    print(f"  - args[9] (agg): {new_args[9]} ⭐")
    print()
    
    return new_args

processed_args = simulate_parse_module_for_hgstage(yaml_args, ch_f=16)


# ============================================
# 3. HG_Stage.__init__ 接收参数
# ============================================
def simulate_hg_stage_init(args):
    """
    模拟 hgnetv2.py 中 HG_Stage.__init__ 的参数接收
    """
    print("=" * 60)
    print("步骤 3: HG_Stage.__init__ 接收参数")
    print("=" * 60)
    
    # 解包参数（对应 hgnetv2.py 第 417-430 行）
    in_chs = args[0]
    mid_chs = args[1]
    out_chs = args[2]
    block_num = args[3]
    layer_num = args[4]
    downsample = args[5]
    light_block = args[6]
    kernel_size = args[7]
    use_lab = args[8]
    agg = args[9]  # ⭐ 'ema' 字符串
    
    print(f"接收到的参数:")
    print(f"  in_chs={in_chs}, mid_chs={mid_chs}, out_chs={out_chs}")
    print(f"  block_num={block_num}, layer_num={layer_num}")
    print(f"  downsample={downsample}, light_block={light_block}")
    print(f"  kernel_size={kernel_size}, use_lab={use_lab}")
    print(f"  agg={agg} ⭐ (这是注意力类型)")
    print()
    
    # 模拟创建 HG_Block（对应 hgnetv2.py 第 448-464 行）
    print(f"创建 {block_num} 个 HG_Block，每个传递 agg='{agg}'")
    print()
    
    return {
        'in_chs': in_chs,
        'mid_chs': mid_chs,
        'out_chs': out_chs,
        'block_num': block_num,
        'layer_num': layer_num,
        'agg': agg
    }

stage_params = simulate_hg_stage_init(processed_args)


# ============================================
# 4. HG_Block.__init__ 根据 agg 创建注意力模块
# ============================================
def simulate_hg_block_init(agg, out_chs, total_chs):
    """
    模拟 hgnetv2.py 中 HG_Block.__init__ 根据 agg 创建注意力模块
    """
    print("=" * 60)
    print("步骤 4: HG_Block.__init__ 创建注意力模块")
    print("=" * 60)
    
    print(f"agg 参数值: '{agg}'")
    print(f"out_chs: {out_chs}")
    print(f"total_chs: {total_chs}")
    print()
    
    # 模拟 hgnetv2.py 第 342-350 行的条件判断
    if agg == 'se':
        print("❌ 不匹配: agg != 'se'")
    elif agg == 'ese':
        print("❌ 不匹配: agg != 'ese'")
    elif agg == 'ema':  # ⭐ 匹配！
        print("✅ 匹配! agg == 'ema'")
        print()
        print("执行代码:")
        print("  aggregation_conv = ConvBNAct(")
        print(f"    total_chs={total_chs}, out_chs={out_chs}, kernel_size=1, ...")
        print("  )")
        print(f"  att = EMA(out_chs={out_chs})  # ⭐ 创建 EMA 模块")
        print("  self.aggregation = nn.Sequential(")
        print("    aggregation_conv,")
        print("    att,")
        print("  )")
        print()
        print("✅ EMA 注意力模块已成功创建！")
    elif agg == 'simam':
        print("❌ 不匹配: agg != 'simam'")
    else:
        print(f"❌ 错误: 未知的 agg 值 '{agg}'")
    
    print()

simulate_hg_block_init(
    agg=stage_params['agg'],
    out_chs=stage_params['out_chs'],
    total_chs=stage_params['in_chs'] + stage_params['layer_num'] * stage_params['mid_chs']
)


# ============================================
# 5. EMA 模块的实际结构
# ============================================
print("=" * 60)
print("步骤 5: EMA 模块结构")
print("=" * 60)
print("""
EMA 模块 (engine/extre_module/custom_nn/attention/ema.py):

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        # channels = out_chs (从 HG_Block 传入)
        # 创建以下组件:
        #   - AdaptiveAvgPool2d (用于空间池化)
        #   - GroupNorm (分组归一化)
        #   - Conv2d 1x1 和 3x3 (卷积层)
        
    def forward(self, x):
        # 实现 EMA 注意力机制:
        #   1. 分组处理特征
        #   2. 水平和垂直方向的空间注意力
        #   3. 特征交互和权重计算
        #   4. 应用注意力权重
        return attended_features
""")

print("=" * 60)
print("完整流程总结")
print("=" * 60)
print("""
YAML: ['ema'] 
  ↓
parse_module: args = [..., 'ema'] (字符串保持)
  ↓
HG_Stage(..., agg='ema') (参数传递)
  ↓
HG_Block(..., agg='ema') (继续传递)
  ↓
if agg == 'ema': att = EMA(out_chs) (条件判断并创建)
  ↓
self.aggregation = Sequential(conv, EMA(...)) (组装模块)
""")

