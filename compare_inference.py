"""
Ablation comparison: 4 models on same test images, side-by-side figure.
Base → +PConv+EMA → +PConv+EMA+C2F → +All Modules
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import YAMLConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRH = 0.3
OUTPUT_DIR = Path('./compare_imgs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each model entry: (display_name, config_path, weight_path, yaml_override or None)
MODELS = [
    ('Base',           'configs/dfine/dfine_hgnetv2_n_custom.yml',
                       'outputs/dfine_hgnetv2_n_custom/best_stg2.pth', None),
    ('+PConv&EMA',     'configs/test/dfine_hgnetv2_test.yml',
                       'outputs/dfine-n-pconv-ema/best_stg2.pth',
                       'configs/cfg-improve/dfine-n-ema-pconv.yaml'),
    ('+C2f',           'configs/test/dfine_hgnetv2_test.yml',
                       'outputs/dfine-n-c2f/best_stg2.pth',
                       'configs/cfg-improve/dfine-n-C2f-IDWB.yaml'),
    ('+MFM&LSK (All)',  'configs/test/dfine_hgnetv2_test.yml',
                       'outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth',
                       'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK-lite-EMA.yaml'),
]

COLORS = [
    (220, 30, 30), (30, 180, 30), (30, 60, 220), (255, 140, 0),
    (220, 200, 0), (0, 200, 200), (200, 0, 200), (160, 0, 0),
    (0, 140, 0), (120, 0, 140), (255, 100, 150), (0, 120, 120),
]


def get_color(cid):
    return COLORS[cid % len(COLORS)]


def get_font(size):
    for fp in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
               '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
               'arial.ttf']:
        try:
            return ImageFont.truetype(fp, size)
        except IOError:
            pass
    return ImageFont.load_default()


def draw_boxes(im, labels, boxes, scores, class_names, thrh=THRH):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    mask = scores[0] > thrh
    for b, lid, s in zip(boxes[0][mask], labels[0][mask], scores[0][mask]):
        lid = int(lid.item())
        color = get_color(lid)
        x1, y1, x2, y2 = b.tolist()
        lw = max(int(min(w, h) * 0.002), 2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = str(class_names[lid]) if class_names and lid < len(class_names) else str(lid)
        fs = max(int(min(x2 - x1, y2 - y1) * 0.08), 10)
        font = get_font(fs)
        tbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        tx, ty = x1, y1 - th - 2
        if ty < 0:
            ty = y1 + 2
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return im


def load_one(config_path, weight_path, yaml_override):
    if yaml_override:
        cfg = YAMLConfig(config_path, **{'DEIM_MG': {'yaml_path': yaml_override}})
    else:
        cfg = YAMLConfig(config_path)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    ckpt = torch.load(weight_path, map_location='cpu')
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

    model = Wrapper().to(DEVICE).eval()
    return model, class_names


def infer(model, img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    t = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
    x = t(im).unsqueeze(0).to(DEVICE)
    sz = torch.tensor([[w, h]]).to(DEVICE)
    with torch.no_grad():
        labels, boxes, scores = model(x, sz)
    return im, labels, boxes, scores


def pick_test_images(n=4):
    """Select n diverse test images from validation set."""
    root = os.path.expanduser(os.environ.get('TT100K_ROOT') or '~/tt100k')
    val_json = os.path.join(root, 'TT100k-val/val.json')
    val_dir = os.path.join(root, 'TT100k-val/images')

    with open(val_json) as f:
        data = json.load(f)
    img_map = {im['id']: im for im in data['images']}
    img_counts = {}
    for a in data['annotations']:
        img_counts[a['image_id']] = img_counts.get(a['image_id'], 0) + 1

    picked = []
    sorted_ids = sorted(img_counts.items(), key=lambda x: -x[1])

    # 1) densest
    for img_id, cnt in sorted_ids:
        if img_id in img_map:
            fn = img_map[img_id]['file_name']
            p = os.path.join(val_dir, fn)
            if os.path.exists(p):
                picked.append((f'{cnt} objects (dense)', p))
                break

    # 2) sparse
    for img_id, cnt in reversed(sorted_ids):
        if img_id in img_map and 1 <= cnt <= 2:
            fn = img_map[img_id]['file_name']
            p = os.path.join(val_dir, fn)
            if os.path.exists(p):
                picked.append((f'{cnt} objects (sparse)', p))
                break

    # 3) medium
    for img_id, cnt in sorted_ids:
        if 4 <= cnt <= 6 and img_id in img_map:
            fn = img_map[img_id]['file_name']
            p = os.path.join(val_dir, fn)
            if os.path.exists(p):
                picked.append((f'{cnt} objects (medium)', p))
                break

    # 4) another distinct one
    for img_id, cnt in sorted_ids:
        if img_id in img_map and img_map[img_id]['file_name'] not in [t[1] for t in picked]:
            fn = img_map[img_id]['file_name']
            p = os.path.join(val_dir, fn)
            if os.path.exists(p):
                picked.append((f'{cnt} objects', p))
                break

    return picked[:n]


def build_comparison(test_images, model_results, class_names, out_path, col_w=750, row_h=750):
    """Render a grid: rows=4 models, cols=N test_images."""
    n_models = len(model_results)
    n_imgs = len(test_images)
    gap = 6
    left_w = 120
    top_h = 40
    cw, rh = col_w, row_h
    cw_total = left_w + n_imgs * cw + (n_imgs - 1) * gap + 30
    rh_total = top_h + n_models * rh + (n_models - 1) * gap + 20

    canvas = Image.new('RGB', (cw_total, rh_total), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(18)

    # Column headers
    for j, (desc, _) in enumerate(test_images):
        cx = left_w + j * (cw + gap) + cw // 2
        tw = draw.textlength(desc, font=title_font)
        draw.text((cx - tw // 2, 10), desc, fill=(40, 40, 40), font=title_font)

    # Rows
    for i, (name, res_imgs) in enumerate(model_results.items()):
        y0 = top_h + i * (rh + gap)
        # Row label with vertical text
        label_font = get_font(16)
        draw.text((5, y0 + rh // 2 - 10), name, fill=(60, 60, 60), font=label_font)
        for j, (desc, img_path) in enumerate(test_images):
            x0 = left_w + j * (cw + gap)
            result = draw_boxes(*res_imgs[j], class_names)
            result = result.resize((cw, rh), Image.LANCZOS)
            canvas.paste(result, (x0, y0))
            draw.rectangle([x0, y0, x0 + cw, y0 + rh], outline=(200, 200, 200), width=1)

    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def main():
    print(f"Device: {DEVICE}")
    test_images = pick_test_images()
    print(f"Test images: {[t[0] for t in test_images]}")

    # Load all models
    models = {}
    global_cn = None
    for name, cfg_path, wt_path, yaml_ov in MODELS:
        print(f"Loading {name}...")
        model, cn = load_one(cfg_path, wt_path, yaml_ov)
        models[name] = model
        if cn is not None and len(cn) > 0:
            global_cn = cn
        print(f"  {name} OK. classes={len(cn) if cn else 'N/A'}")

    # Run all models on all test images
    results = {name: [] for name in models}
    for name, model in models.items():
        print(f"Inferring with {name}...")
        for desc, img_path in test_images:
            results[name].append(infer(model, img_path))

    # Generate comparison figure
    build_comparison(test_images, results, global_cn, OUTPUT_DIR / 'ablation_comparison.png')

    # Also generate individual detailed figures per test image
    for j, (desc, img_path) in enumerate(test_images):
        im_name = Path(img_path).stem
        # Stack vertically: each model's result
        for i, (name, res_imgs) in enumerate(results.items()):
            result = draw_boxes(*res_imgs[j], global_cn)
            result.save(OUTPUT_DIR / f'{im_name}_{name.replace(" ", "_")}.jpg', quality=95)
        print(f"  Saved individual results for {im_name}")

    print("\nDone! Check compare_imgs/")


if __name__ == '__main__':
    main()
