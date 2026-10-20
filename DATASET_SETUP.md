# TT100K 数据集路径

训练、验证、蒸馏和对比可视化共用环境变量 `TT100K_ROOT`。
未设置时默认使用当前用户的 `~/tt100k`，保持原服务器的运行方式。
迁移后在启动任务的终端设置：

```bash
export TT100K_ROOT="/path/to/tt100k"
bash train.sh
# 或 bash validate.sh / bash distill.sh / bash batch_infer.sh
```

目录结构：

```text
$TT100K_ROOT/
├── TT100k-train/
│   ├── images/
│   └── train.json
└── TT100k-val/
    ├── images/
    └── val.json
```

YAML 配置中的 `${TT100K_ROOT}` 由配置加载器解析，不需要手动替换。
其他数据集的示例配置仍需按对应数据集设置；模型权重路径不属于此变量。
训练权重仍遵循项目原有的 Git 忽略规则。
