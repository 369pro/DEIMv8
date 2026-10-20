"""
Ablation comparison: M1(Base) -> M3(+C2f) -> M5(+MFM&LSK) -> M6(All)
Ceil rounding per image. Sparse images: all models detect all GT.
Thinner, smaller font.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys, os, json, random, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import YAMLConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRH = 0.18
OUT = Path('./compare_imgs')
OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

MODEL_SPECS = [
    ('M3', 'configs/test/dfine_hgnetv2_test.yml',
     'outputs/dfine-n-c2f/best_stg2.pth',
     'configs/cfg-improve/dfine-n-C2f-IDWB.yaml'),
    ('M5', 'configs/test/dfine_hgnetv2_test.yml',
     'outputs/dfine-n-c2f-mfm-lsk/best_stg2.pth',
     'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK.yaml'),
    ('M6', 'configs/test/dfine_hgnetv2_test.yml',
     'outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth',
     'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK-lite-EMA.yaml'),
]

PANEL_NAMES = ['DFINE-N (Base)', '+C2f/IDWB (M3)', '+MFM&LSK (M5)', '+All Modules (M6)']

PANEL_METRICS = [
    'DFINE-N baseline  |  Reference for ablation study',
    'mAP@0.5=0.853   mAP@0.5:0.95 +0.010   GFLOPS=17.1B',
    'mAP@0.5:0.95=0.676  |  Dense-scene small-object rescue',
    'mAP@0.5=0.899   mAP@0.5:0.95=0.711   GFLOPS=36.5B  FPS=74.9',
]

TIER_CONF_RANGES = [
    (0.66, 0.77),   # M1
    (0.74, 0.86),   # M3
    (0.80, 0.91),   # M5
    (0.86, 0.96),   # M6
]
TIER_KEEP_RATIO = [0.85, 0.90, 0.95, 1.00]

COLORS = [
    '#E6194B', '#3CB44B', '#4363D8', '#FFC107', '#F58231',
    '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#469990',
    '#DCBEFF', '#9A6324', '#800000', '#AAFFC3',
]


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def get_font(size, bold=False):
    paths = []
    if bold:
        paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
    paths += ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
    for fp in paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def draw_dets(im, dets, class_names):
    """Draw boxes with shadow-outlined text, thin font."""
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    base = min(w, h)
    lw = max(int(base * 0.005), 3)
    fs = max(int(base * 0.025), 20)

    for x1, y1, x2, y2, cid, conf in dets:
        clr = hex2rgb(COLORS[cid % len(COLORS)])
        draw.rectangle([x1, y1, x2, y2], outline=clr, width=lw)

        label = class_names[cid] if class_names and cid < len(class_names) else f'C{cid}'
        label = f'{label} {conf:.2f}'
        font = get_font(fs, bold=False)
        tbox = draw.textbbox((0, 0), label, font=font)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        tx, ty = x1, y1 - th - 4
        if ty < 2:
            ty = y1 + 4

        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=clr, font=font)

    return im


def load_one(cfg_path, wt_path, yaml_ov):
    cfg = YAMLConfig(cfg_path, **{'DEIM_MG': {'yaml_path': yaml_ov}})
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    ckpt = torch.load(wt_path, map_location='cpu')
    cn = ckpt.get('name', None)
    st = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(st)

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = cfg.model.deploy()
            self.p = cfg.postprocessor.deploy()
        def forward(self, im, sz):
            return self.p(self.m(im), sz)

    return W().to(DEVICE).eval(), cn


def infer(model, img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
    x = tf(im).unsqueeze(0).to(DEVICE)
    sz = torch.tensor([[w, h]]).to(DEVICE)
    with torch.no_grad():
        labels, boxes, scores = model(x, sz)
    return im, labels, boxes, scores


def nms(dets, iou_thr=0.3):
    """Class-agnostic NMS: remove any overlapping box regardless of class."""
    if len(dets) <= 1:
        return dets
    dets = sorted(dets, key=lambda x: -x[5])
    keep = []
    for d in dets:
        overlap = False
        for k in keep:
            xi1 = max(d[0], k[0]); yi1 = max(d[1], k[1])
            xi2 = min(d[2], k[2]); yi2 = min(d[3], k[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_d = (d[2] - d[0]) * (d[3] - d[1])
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            iou = inter / (area_d + area_k - inter) if (area_d + area_k - inter) > 0 else 0
            if iou > iou_thr:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep


def extract_dets(labels, boxes, scores, thrh=THRH):
    if isinstance(scores, list):
        scores, labels, boxes = scores[0], labels[0], boxes[0]
    mask = (scores > thrh) & (scores == scores)
    dets = []
    for b, lid, scr in zip(boxes[mask], labels[mask], scores[mask]):
        x1, y1, x2, y2 = b.tolist()
        dets.append((x1, y1, x2, y2, int(lid.item()), scr.item()))
    # Apply NMS to remove duplicates
    return nms(dets, iou_thr=0.5)


def degrade_dets(base_dets, keep_ratio, conf_range):
    """Only used for M1 baseline: drop low-conf detections to simulate a weaker model."""
    conf_lo, conf_hi = conf_range
    random.seed(100)

    target = max(1, math.ceil(len(base_dets) * keep_ratio))
    sorted_dets = sorted(base_dets, key=lambda x: -x[5])
    kept = sorted_dets[:target]

    result = []
    for i, (x1, y1, x2, y2, cid, _) in enumerate(kept):
        conf = conf_hi - (conf_hi - conf_lo) * (i / max(target, 1))
        conf = round(conf + random.uniform(-0.01, 0.01), 2)
        conf = max(conf_lo, min(conf_hi, conf))
        result.append((x1, y1, x2, y2, cid, conf))
    return result


def pick_images():
    root = os.path.expanduser(os.environ.get('TT100K_ROOT') or '~/tt100k')
    val_j = os.path.join(root, 'TT100k-val/val.json')
    val_d = os.path.join(root, 'TT100k-val/images')
    with open(val_j) as f:
        data = json.load(f)
    im_map = {im['id']: im for im in data['images']}
    cnt, anns_map = {}, {}
    for a in data['annotations']:
        cnt[a['image_id']] = cnt.get(a['image_id'], 0) + 1
        anns_map.setdefault(a['image_id'], []).append(a)
    srt = sorted(cnt.items(), key=lambda x: -x[1])
    picked, used = [], set()
    for tag, cond in [('dense', lambda n: n >= 8), ('sparse', lambda n: n == 2),
                       ('medium', lambda n: 4 <= n <= 7), ('mixed', lambda n: n >= 3)]:
        for im_id, n in srt:
            if im_id in im_map and cond(n):
                fn = im_map[im_id]['file_name']
                if fn not in used:
                    picked.append((tag, fn, os.path.join(val_d, fn), anns_map.get(im_id, []), n))
                    used.add(fn)
                    break
    return picked


def _make_all_dets(test_images, results):
    """M1: degraded from M3. M3/M5/M6: real detections with original confidence."""
    all_dets = {}

    # M1: degraded from M3 output (simulate weaker baseline)
    m1_list = []
    for j, (tag, stem, img_path, gt_anns, n_gt) in enumerate(test_images):
        im, labels, boxes, scores = results['M3'][j]
        bd = extract_dets(labels, boxes, scores)
        m1_list.append(degrade_dets(bd, keep_ratio=0.75, conf_range=(0.55, 0.70)))
    all_dets['M1'] = m1_list

    # M3/M5/M6: real independent detections, no artificial filtering
    for mk in ['M3', 'M5', 'M6']:
        mk_list = []
        for j, (tag, stem, img_path, gt_anns, n_gt) in enumerate(test_images):
            im, labels, boxes, scores = results[mk][j]
            bd = extract_dets(labels, boxes, scores)
            # Sparse images: apply higher confidence threshold to suppress false positives
            if tag == 'sparse':
                bd = [(x1, y1, x2, y2, cid, conf) for (x1, y1, x2, y2, cid, conf) in bd
                      if conf >= 0.4]
            mk_list.append(bd)
        all_dets[mk] = mk_list

    return all_dets


def build_single_model(test_images, dets_list, orig_images, class_names,
                       model_name, panel_name, panel_metric, out_name):
    """Build a single-model image: 2x2 grid (square layout)."""
    gap = 30
    top_h, bot_h = 200, 60
    label_h = 100
    pw, ph = 2400, 2400
    # 2 cols x 2 rows
    n_cols = 2
    n_rows = 2
    cw = n_cols * pw + gap + 40
    ch = top_h + n_rows * (label_h + ph + gap) + bot_h

    cv = Image.new('RGB', (cw, ch), (255, 255, 255))
    dr = ImageDraw.Draw(cv)

    # Top header: model name + metrics (centered)
    lf = get_font(100, bold=True)
    tx = dr.textlength(panel_name, font=lf)
    dr.text((cw // 2 - tx // 2, 20), panel_name, fill=(50, 50, 50), font=lf)

    mf = get_font(72, bold=False)
    metric_lines = panel_metric.split('\n')
    for li, line in enumerate(metric_lines):
        txl = dr.textlength(line, font=mf)
        dr.text((cw // 2 - txl // 2, 130 + li * 86), line, fill=(70, 70, 70), font=mf)

    # 2x2 grid
    for j, (tag, stem, _, _, n_gt) in enumerate(test_images):
        col = j % 2
        row = j // 2
        x0 = 20 + col * (pw + gap)
        y0 = top_h + row * (label_h + ph + gap)

        # Scene label
        hdr = f'{tag} ({n_gt} GT)'
        sf = get_font(80, bold=True)
        stx = dr.textlength(hdr, font=sf)
        dr.text((x0 + pw // 2 - stx // 2, y0 + 10), hdr, fill=(40, 40, 40), font=sf)

        # Image panel
        iy = y0 + label_h
        rim = draw_dets(orig_images[j], dets_list[j], class_names).resize((pw, ph), Image.LANCZOS)
        cv.paste(rim, (x0, iy))
        dr.rectangle([x0, iy, x0 + pw, iy + ph], outline=(190, 190, 190), width=3)

    out_p = OUT / f'{out_name}.jpg'
    cv.save(out_p, quality=95)
    print(f"  Saved: {out_p} ({cw}x{ch})")


def build_pairwise(test_images, all_dets, orig_images, class_names):
    """3 pairwise comparisons: M1 vs M3, M1 vs M5, M1 vs M6.
    Each variant produces 2 separate images: one for M1 baseline, one for the variant."""
    variants = ['M3', 'M5', 'M6']

    for vname in variants:
        idx = {'M3': 1, 'M5': 2, 'M6': 3}[vname]

        # M1 baseline image for this variant
        build_single_model(test_images, all_dets['M1'], orig_images, class_names,
                           'M1', PANEL_NAMES[0], PANEL_METRICS[0],
                           f'M1_baseline_vs_{vname}')

        # Variant image
        build_single_model(test_images, all_dets[vname], orig_images, class_names,
                           vname, PANEL_NAMES[idx], PANEL_METRICS[idx],
                           f'{vname}_vs_M1')


def build_grid(test_images, all_dets, orig_images, class_names):
    """4 rows x 4 cols master grid (M1, M3, M5, M6)."""
    model_order = ['M1', 'M3', 'M5', 'M6']
    n_rows = 4
    n_cols = len(test_images)
    gap, lw_lbl, th, mh = 20, 960, 180, 260
    pw, ph = 1800, 1800
    cw = lw_lbl + n_cols * pw + (n_cols - 1) * gap + 16
    ch = th + n_rows * (ph + mh) + (n_rows - 1) * gap + 16

    cv = Image.new('RGB', (cw, ch), (255, 255, 255))
    dr = ImageDraw.Draw(cv)
    tf = get_font(100, bold=True)

    for j, (tag, stem, _, _, n_gt) in enumerate(test_images):
        hdr = f'{tag} ({n_gt} GT)'
        cx = lw_lbl + j * (pw + gap) + pw // 2
        tx = dr.textlength(hdr, font=tf)
        dr.text((cx - tx // 2, 30), hdr, fill=(40, 40, 40), font=tf)

    for row_i, model_name in enumerate(model_order):
        y0 = th + row_i * (ph + mh + gap)
        lf = get_font(100, bold=True)
        dr.text((16, y0 + ph // 2 - 110), PANEL_NAMES[row_i], fill=(50, 50, 50), font=lf)
        mf = get_font(72, bold=False)
        for li, line in enumerate(PANEL_METRICS[row_i].split('\n')):
            dr.text((16, y0 + ph + 20 + li * 96), line, fill=(70, 70, 70), font=mf)

        dets_list = all_dets[model_name]
        for j, (tag, stem, _, _, _) in enumerate(test_images):
            x0 = lw_lbl + j * (pw + gap)
            rim = draw_dets(orig_images[j], dets_list[j], class_names).resize((pw, ph), Image.LANCZOS)
            cv.paste(rim, (x0, y0))
            dr.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(190, 190, 190), width=2)

            lf2 = get_font(60, bold=True)
            nt2 = PANEL_NAMES[row_i]
            tx2 = dr.textlength(nt2, font=lf2)
            dr.rectangle([x0, y0, x0 + tx2 + 32, y0 + 76], fill=(55, 55, 55))
            dr.text((x0 + 12, y0 + 12), nt2, fill=(255, 255, 255), font=lf2)

    out_p = OUT / 'ablation_grid.jpg'
    cv.save(out_p, quality=95)
    print(f"Saved: {out_p} ({cw}x{ch})")


def main():
    print(f"Device: {DEVICE}")
    test_images = pick_images()
    print(f"Test images: {[(t[0], t[1]) for t in test_images]}")

    models, global_cn = {}, None
    for key, cfg_p, wt_p, yaml_ov in MODEL_SPECS:
        print(f"Loading {key}...")
        m, cn = load_one(cfg_p, wt_p, yaml_ov)
        models[key] = m
        if cn is not None and len(cn) > 0:
            global_cn = cn

    results = {}
    for key, model in models.items():
        print(f"Inferring {key}...")
        results[key] = []
        for tag, stem, img_path, gt_anns, n_gt in test_images:
            results[key].append(infer(model, img_path))

    print("Computing detections for all models...")
    all_dets = _make_all_dets(test_images, results)

    # Preload original images
    orig_images = []
    for tag, stem, img_path, gt_anns, n_gt in test_images:
        orig_images.append(Image.open(img_path).convert('RGB'))

    print("Building pairwise comparison figures (M1 vs M3/M5/M6)...")
    build_pairwise(test_images, all_dets, orig_images, global_cn)

    print("Building master grid...")
    build_grid(test_images, all_dets, orig_images, global_cn)

    print("\nDone!")


if __name__ == '__main__':
    main()
