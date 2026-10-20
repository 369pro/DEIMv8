"""
Per-detection Grad-CAM with GT boxes.
Target = class logit of specific detection queries (not layer-average).
Green boxes = GT annotations. Shared per-image-pair normalization.
"""
import warnings
warnings.filterwarnings('ignore')
import torch, cv2, os, sys, json
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import YAMLConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('./compare_imgs')

FONT_PATHS = [
    '/tmp/NotoSansCJKsc-Regular.otf',
    '/tmp/wqy-extract/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
]

def load_font(size):
    for fp in FONT_PATHS:
        try: return ImageFont.truetype(fp, size)
        except Exception: continue
    return ImageFont.load_default()

# Decoder-input C2f_Block at P4/16 — same position across models
LAYERS = {'M3': 'encoder.6', 'M5': 'encoder.13', 'M6': 'encoder.13'}

MODELS = {
    'M3': ('configs/test/dfine_hgnetv2_test.yml',
           'outputs/dfine-n-c2f/best_stg2.pth',
           'configs/cfg-improve/dfine-n-C2f-IDWB.yaml'),
    'M5': ('configs/test/dfine_hgnetv2_test.yml',
           'outputs/dfine-n-c2f-mfm-lsk/best_stg2.pth',
           'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK.yaml'),
    'M6': ('configs/test/dfine_hgnetv2_test.yml',
           'outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth',
           'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK-lite-EMA.yaml'),
}

IMAGES = [
    ('dense',  '53947.jpg'),  # 14 GT
    ('sparse', '66913.jpg'),  # 2 GT
    ('medium', '1719.jpg'),   # 5 GT
]

TT100K_ROOT = os.path.expanduser(os.environ.get('TT100K_ROOT') or '~/tt100k')
VAL_JSON = os.path.join(TT100K_ROOT, 'TT100k-val/val.json')
IMG_DIR = os.path.join(TT100K_ROOT, 'TT100k-val/images')

# ---------- GT loading ----------
_gt_cache = None
def load_gt():
    global _gt_cache
    if _gt_cache is None:
        with open(VAL_JSON) as f:
            _gt_cache = json.load(f)
    return _gt_cache

def get_gt_boxes(filename):
    data = load_gt()
    im_map = {i['id']: i for i in data['images']}
    for im_id, info in im_map.items():
        if info['file_name'] == filename:
            anns = [a for a in data['annotations'] if a['image_id'] == im_id]
            return [(a['bbox'][0], a['bbox'][1],
                     a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3])
                    for a in anns]
    return []

# ---------- Model helpers ----------
def get_layer(model, name):
    for k in name.split('.'):
        model = model[int(k)] if k.isdigit() else getattr(model, k)
    return model

def load_model(cfg_p, wt_p, yaml_ov):
    cfg = YAMLConfig(cfg_p, **{'DEIM_MG': {'yaml_path': yaml_ov}})
    if 'HGNetv2' in cfg.yaml_cfg: cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    ck = torch.load(wt_p, map_location='cpu')
    st = ck['ema']['module'] if 'ema' in ck else ck['model']
    cfg.model.load_state_dict(st)
    return cfg.model.to(DEVICE).eval(), cfg.postprocessor.deploy()

def get_top_dets(model, post, img_path, conf_thr=0.25, top_k=5):
    im = Image.open(img_path).convert('RGB')
    w0, h0 = im.size
    tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
    xt = tf(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(xt)
    labels, boxes, scores = post(out, torch.tensor([[w0, h0]]).to(DEVICE))
    l, b, s = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
    logits = out['pred_logits'][0].detach().cpu()
    pboxes = out['pred_boxes'][0].detach().cpu()

    mask = s >= conf_thr
    l, b, s = l[mask], b[mask], s[mask]
    if not len(s): return [], im
    order = s.argsort(descending=True)[:min(top_k, len(s))]

    results = []
    for idx in order:
        box = b[idx]; cls = int(l[idx])
        best_q = 0; best_iou = 0
        for q in range(pboxes.size(0)):
            ix1 = max(box[0], pboxes[q,0]); iy1 = max(box[1], pboxes[q,1])
            ix2 = min(box[2], pboxes[q,2]); iy2 = min(box[3], pboxes[q,3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                a1 = (box[2]-box[0])*(box[3]-box[1])
                a2 = (pboxes[q,2]-pboxes[q,0])*(pboxes[q,3]-pboxes[q,1])
                iou = inter/(a1+a2-inter+1e-6)
                if iou > best_iou: best_iou = iou; best_q = q
        results.append({'qi': best_q, 'ci': cls, 'box': box.tolist()})
    return results, im

def grad_cam(model, layer, im_pil, queries, top_k=5):
    w0, h0 = im_pil.size
    tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
    xt = tf(im_pil).unsqueeze(0).to(DEVICE)
    act = {}
    h = layer.register_forward_hook(lambda m,i,o: act.update(feat=o))
    out = model(xt)
    logits = out['pred_logits']
    h.remove()
    if 'feat' not in act: return None
    feat = act['feat']
    cams = []
    for q in queries[:top_k]:
        model.zero_grad()
        score = logits[0, q['qi'], q['ci']]
        g = torch.autograd.grad(score, feat, retain_graph=True)[0]
        w = g[0].mean(dim=(1,2))
        cam = torch.relu((w.view(-1,1,1)*feat[0]).sum(0)).detach().cpu().numpy()
        cams.append(cam)
    if not cams: return None
    avg = np.mean(cams, 0)
    mn, mx = avg.min(), avg.max()
    avg = (avg-mn)/(mx-mn+1e-8)
    return cv2.resize(avg, (w0, h0))

def draw_gt_boxes(im_pil, boxes):
    """Draw green GT boxes on a copy of im_pil."""
    im = im_pil.copy()
    dr = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in boxes:
        dr.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
    return im

def overlay(cam, im_pil, vmin, vmax, alpha=0.5):
    clipped = np.clip((cam-vmin)/(vmax-vmin+1e-8), 0, 1)
    h8 = (clipped*255).astype(np.uint8)
    color = cv2.applyColorMap(h8, cv2.COLORMAP_TURBO)
    orig = np.array(im_pil)
    blended = cv2.addWeighted(orig, 1-alpha, color, alpha, 0)
    return Image.fromarray(blended)

# ---------- Grid ----------
def grid(pairs, title, la, lb, fn):
    gap, lw = 28, 900
    th, rh, bh = 300, 3300, 120
    pw, ph = 3300, 3300
    n = len(pairs)
    cw = lw + n*pw + (n-1)*gap + 40
    ch = th + 2*(rh+gap) + bh
    cv = Image.new('RGB', (cw, ch), (255,255,255))
    dr = ImageDraw.Draw(cv)
    f = load_font(90)
    tx = dr.textlength(title, font=f)
    dr.text((cw//2-tx//2, 30), title, fill=(30,30,30), font=f)
    cb_w = pw*3+gap*2; cb_x = lw; cb_y = th-80
    for i in range(256):
        v = i/255
        r,g,b = (0,int(255*v/0.33),255) if v<0.33 else ((int(255*(v-0.33)/0.34),255,int(255*(0.67-v)/0.34)) if v<0.67 else (255,int(255*(1-v)/0.33),0))
        x = cb_x + i*cb_w//256
        dr.rectangle([x,cb_y,x+cb_w//256+1,cb_y+36], fill=(r,g,b))
    sf = load_font(60)
    dr.text((cb_x, cb_y+42), '低', fill=(60,60,60), font=sf)
    hi = '高'; dr.text((cb_x+cb_w-dr.textlength(hi,font=sf), cb_y+42), hi, fill=(60,60,60), font=sf)
    hf = load_font(80)
    for j, sub in enumerate(['密集 (14 GT)', '稀疏 (2 GT)', '中等 (5 GT)']):
        cx = lw + j*(pw+gap) + pw//2
        stx = dr.textlength(sub, font=hf)
        dr.text((cx-stx//2, th-30), sub, fill=(50,50,50), font=hf)
    for ri, (lab, hms) in enumerate([(la, [p[0] for p in pairs]), (lb, [p[1] for p in pairs])]):
        y0 = th + ri*(rh+gap)
        lf = load_font(90)
        dr.text((20, y0+40), lab, fill=(50,50,50), font=lf)
        for j, hm in enumerate(hms):
            x0 = lw + j*(pw+gap); iy = y0+140
            if hm: cv.paste(hm.resize((pw,ph), Image.LANCZOS), (x0, iy))
            else: dr.rectangle([x0,iy,x0+pw,iy+ph], fill=(240,240,240))
            dr.rectangle([x0,iy,x0+pw,iy+ph], outline=(180,180,180), width=3)
    cv.save(OUT/fn, quality=95)
    print(f"  已保存: {fn}  ({cw}×{ch})")

# ---------- Main ----------
def run(title, la, lb, fn, ka, kb):
    print(f"\n{'='*60}\n  {title}")
    ma, pa = load_model(*MODELS[ka]); ta = get_layer(ma, LAYERS[ka])
    mb, pb = load_model(*MODELS[kb]); tb = get_layer(mb, LAYERS[kb])
    pairs = []
    for tag, fname in IMAGES:
        ip = f'{IMG_DIR}/{fname}'
        da, im = get_top_dets(ma, pa, ip); db, _ = get_top_dets(mb, pb, ip)
        n = min(5, len(da), len(db))
        if n < 1: pairs.append((None,None)); continue
        ca = grad_cam(ma, ta, im, da, n)
        cb = grad_cam(mb, tb, im, db, n)
        if ca is None or cb is None: pairs.append((None,None)); continue
        # Shared norm
        C = np.concatenate([ca.ravel(), cb.ravel()])
        lo, hi = np.percentile(C, [5, 98])
        # Draw GT boxes on base image, then overlay heatmap
        gt_boxes = get_gt_boxes(fname)
        im_gt = draw_gt_boxes(im, gt_boxes)
        ha = overlay(ca, im_gt, lo, hi)
        hb = overlay(cb, im_gt, lo, hi)
        pairs.append((ha, hb))
    grid(pairs, title, la, lb, fn)

def main():
    print(f"设备: {DEVICE}")
    run('EMA 消融：检测框 Grad-CAM + 绿色 GT 框（编码器 C2f，P4/16）',
        'M3: C2f_Block（SE，无 EMA）', 'M6: C2f_Block（EMA 交叉空间注意力）',
        'heatmap_ema_comparison.jpg', 'M3', 'M6')
    run('LSKNet + MFM 消融：检测框 Grad-CAM + 绿色 GT 框（编码器 C2f，P4/16）',
        'M3: C2f_Block（无 LSKNet / MFM）', 'M5: C2f_Block（LSKNet + MFM）',
        'heatmap_lsk_comparison.jpg', 'M3', 'M5')
    print("\n完成！")

if __name__ == '__main__':
    main()
