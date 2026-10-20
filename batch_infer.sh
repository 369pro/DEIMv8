#!/bin/bash
# Batch inference for ablation comparison: 4 models on same test images
# Usage: bash batch_infer.sh

MODE=${1:-all}  # "all" or "single"

export TT100K_ROOT="${TT100K_ROOT:-$HOME/tt100k}"
VAL_DIR="$TT100K_ROOT/TT100k-val/images"
OUT_DIR=./compare_imgs
mkdir -p $OUT_DIR

# Pick 4 diverse test images
# Dense, sparse, medium, challenging
IMAGES=(
  # Will be populated below
)

# --- Step 1: Pick best test images ---
echo "=== Picking test images ==="
python3 -c "
import json, os, sys
from collections import defaultdict

root = os.path.expanduser(os.environ['TT100K_ROOT'])
val_json = os.path.join(root, 'TT100k-val/val.json')
val_dir = os.path.join(root, 'TT100k-val/images')

with open(val_json) as f:
    data = json.load(f)

img_map = {im['id']: im for im in data['images']}
ann_counts = defaultdict(int)
for a in data['annotations']:
    ann_counts[a['image_id']] += 1

sorted_imgs = sorted(ann_counts.items(), key=lambda x: -x[1])

picked = []
# 1) Densest
for img_id, cnt in sorted_imgs:
    if img_id in img_map:
        fn = img_map[img_id]['file_name']
        p = os.path.join(val_dir, fn)
        if os.path.exists(p) and not any(fn in x for x in picked):
            picked.append(('dense', fn, p))
            break

# 2) Sparse (2 objects)
for img_id, cnt in reversed(sorted_imgs):
    if img_id in img_map and cnt == 2:
        fn = img_map[img_id]['file_name']
        p = os.path.join(val_dir, fn)
        if os.path.exists(p) and not any(fn in x for x in picked):
            picked.append(('sparse', fn, p))
            break

# 3) Medium (5-7 objects)
for img_id, cnt in sorted_imgs:
    if 5 <= cnt <= 7 and img_id in img_map:
        fn = img_map[img_id]['file_name']
        p = os.path.join(val_dir, fn)
        if os.path.exists(p) and not any(fn in x for x in picked):
            picked.append(('medium', fn, p))
            break

# 4) Another diverse
for img_id, cnt in sorted_imgs:
    if img_id in img_map and cnt >= 3:
        fn = img_map[img_id]['file_name']
        p = os.path.join(val_dir, fn)
        if os.path.exists(p) and not any(fn in x for x in picked):
            picked.append(('normal', fn, p))
            break

for tag, fn, p in picked:
    print(f'{tag}|{fn}|{p}')
" > /tmp/test_images.txt

cat /tmp/test_images.txt

# --- Step 2: Run inference for each model ---
declare -A MODEL_CONFIGS
MODEL_CONFIGS=(
  ["base"]="configs/dfine/dfine_hgnetv2_n_custom.yml|outputs/dfine_hgnetv2_n_custom/best_stg2.pth"
  ["m2_pconv_ema"]="configs/test/dfine_hgnetv2_test.yml|outputs/dfine-n-pconv-ema/best_stg2.pth"
  ["m3_c2f"]="configs/test/dfine_hgnetv2_test.yml|outputs/dfine-n-c2f/best_stg2.pth"
  ["m6_all"]="configs/test/dfine_hgnetv2_test.yml|outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth"
)

declare -A MODEL_YAML
MODEL_YAML=(
  ["base"]=""
  ["m2_pconv_ema"]="configs/cfg-improve/dfine-n-ema-pconv.yaml"
  ["m3_c2f"]="configs/cfg-improve/dfine-n-C2f-IDWB.yaml"
  ["m6_all"]="configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK-lite-EMA.yaml"
)

echo "=== Running inference ==="
while IFS='|' read -r tag fn path; do
  for model in base m2_pconv_ema m3_c2f m6_all; do
    cfg="${MODEL_CONFIGS[$model]%%|*}"
    weight="${MODEL_CONFIGS[$model]##*|}"
    yaml_override="${MODEL_YAML[$model]}"

    model_out="$OUT_DIR/$model"
    mkdir -p "$model_out"

    echo "--- $model on $fn ($tag) ---"

    if [ -z "$yaml_override" ]; then
      python3 tools/inference/torch_inf.py \
        -c "$cfg" -r "$weight" -i "$path" -o "$model_out" -t 0.3
    else
      # For DEIM_MG models, modify config on-the-fly
      python3 -c "
import sys, os
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from engine.core import YAMLConfig

# Load
cfg = YAMLConfig('$cfg', **{'DEIM_MG': {'yaml_path': '$yaml_override'}})
if 'HGNetv2' in cfg.yaml_cfg:
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False

ckpt = torch.load('$weight', map_location='cpu')
class_names = ckpt.get('name', None)
state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
cfg.model.load_state_dict(state)

class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
    def forward(self, images, orig_sizes):
        return self.postprocessor(self.model(images), orig_sizes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wrapper().to(device).eval()

# Infer
im = Image.open('$path').convert('RGB')
w, h = im.size
sz = torch.tensor([[w, h]]).to(device)
tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
x = tf(im).unsqueeze(0).to(device)

with torch.no_grad():
    labels, boxes, scores = model(x, sz)

# Save raw tensors for later comparison figure
os.makedirs('$model_out', exist_ok=True)
base = os.path.splitext(os.path.basename('$path'))[0]
torch.save({'labels': labels, 'boxes': boxes, 'scores': scores, 'class_names': class_names},
           os.path.join('$model_out', base + '_result.pt'))
print(f'Saved: $model_out/{base}_result.pt')
" 2>&1 | tail -5
    fi
  done
done < /tmp/test_images.txt

echo "=== All inference done ==="
