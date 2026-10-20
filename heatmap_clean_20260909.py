"""Real checkpoint LayerCAM/GradCAM. No synthetic detections or GT-shaped heatmaps."""
import os, json, gc, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from engine.core import YAMLConfig

torch.set_num_threads(4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('compare_imgs/heatmap_clean'); OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(os.environ.get('TT100K_ROOT') or '~/tt100k').expanduser() / 'TT100k-val'
FILES = ['53947.jpg', '66913.jpg', '1719.jpg', '9051.jpg']
SPECS = {
 'M1': ('configs/dfine/dfine_hgnetv2_n_custom.yml', 'outputs/dfine_hgnetv2_n_custom/best_stg2.pth', None, 'backbone.stages.2'),
 'M6': ('configs/test/dfine_hgnetv2_test.yml', 'outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth', 'configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK-lite-EMA.yaml', 'backbone.3'),
}
def load(key):
    conf, weight, override, layer = SPECS[key]
    ck = torch.load(weight, map_location='cpu', weights_only=False)
    state = ck['ema']['module'] if 'ema' in ck else ck['model']
    for agg in ['se', 'ema'] if key == 'M1' else ['ema']:
        cfg = YAMLConfig(conf, **({'DEIM_MG': {'yaml_path': override}} if override else {}))
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        cfg.yaml_cfg['HGNetv2']['agg'] = agg
        model = cfg.model
        try:
            model.load_state_dict(state, strict=True)
            print('LOADED',key,agg,layer,flush=True)
            return model.to(DEVICE).eval(), dict(model.named_modules())[layer], agg
        except RuntimeError as e:
            print('STRICT_LOAD_REJECTED',key,agg,str(e)[:500],flush=True)
    raise RuntimeError('No architecture strictly matches checkpoint '+key)

def cam(model, layer, im):
    a = {}
    hook = layer.register_forward_hook(lambda m,i,o: a.update(x=o))
    x = torch.from_numpy(np.asarray(im.resize((1024,1024))).copy()).permute(2,0,1).float().div(255)[None].to(DEVICE)
    x.requires_grad_(True)
    pred = model(x)
    hook.remove()
    logits = pred['pred_logits'][0]
    scores, cls = logits.sigmoid().max(-1)
    # Query indices are taken directly from decoder output: no coordinate matching.
    qs = scores.argsort(descending=True)
    qs = qs[scores[qs] >= .25][:15]
    if not len(qs): raise RuntimeError('No predictions above .25; no meaningful detection CAM')
    target = logits[qs, cls[qs]].mean()
    feat = a['x']
    grad, = torch.autograd.grad(target, feat)
    raw = {
      'layercam': (grad.relu()*feat).sum(1).relu()[0],
      'gradcam': (grad.mean((2,3),keepdim=True)*feat).sum(1).relu()[0],
    }
    maps = {k:v.detach().cpu().numpy() for k,v in raw.items()}
    boxes = pred['pred_boxes'][0,qs].detach().cpu().numpy()
    dets = [{'query':int(q), 'class':int(cls[q]), 'score':float(scores[q]), 'cxcywh_normalized':b.tolist()} for q,b in zip(qs,boxes)]
    return maps,dets

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def overlay(im, raw):
    # Within-map normalization for spatial pattern comparison, NOT absolute strength.
    scale = float(np.percentile(raw,99.5))
    normalized = np.clip(raw / max(scale,1e-12),0,1)
    normalized = cv2.resize(normalized,im.size,interpolation=cv2.INTER_LINEAR)
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(normalized*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    alpha = (.70*normalized**.65)[...,None]
    return Image.fromarray(np.uint8(np.asarray(im)*(1-alpha)+color*alpha)),scale

def render(images, maps, metadata, gt):
    for method in ['layercam','gradcam']:
        width, cell, margin, gap = 2400, 750, 36, 30
        canvas = Image.new('RGB',(width,2020),'white'); d=ImageDraw.Draw(canvas)
        d.text((margin,20), 'M1 vs M6 | '+method.upper()+' | Real checkpoint outputs',font=font(37),fill='#17212b')
        d.text((margin,72),'Same images / P4 backbone features / top-scoring detections (score >= 0.25)',font=font(24),fill='#536171')
        labels=['Original image','M1 baseline','M6 full model']
        for c,label in enumerate(labels): d.text((margin+c*(cell+gap),123),label,font=font(30),fill='#17212b')
        for row,im in enumerate(images[:3]):
            filename=FILES[row]; boxlist=gt[filename]
            # Same annotation-defined crop for both models; CAM itself is never masked by GT.
            x1=min(b[0] for b in boxlist); y1=min(b[1] for b in boxlist)
            x2=max(b[2] for b in boxlist); y2=max(b[3] for b in boxlist)
            pad=max(100,(x2-x1)*.12); crop=(max(0,int(x1-pad)),max(0,int(y1-pad)),min(im.width,int(x2+pad)),min(im.height,int(y2+pad)))
            panels=[im]
            for key in ['M1','M6']:
                hm,scale=overlay(im,maps[key][row][method]); panels.append(hm)
                metadata[key][row][method+'_scale']=scale
                hm.save(OUT/(key+'_'+Path(filename).stem+'_'+method+'.png'))
            y=170+row*580
            for c,panel in enumerate(panels):
                x=margin+c*(cell+gap)
                # Full scene keeps its aspect ratio, with a separate detail strip below.
                thumb=panel.copy(); thumb.thumbnail((cell,370))
                canvas.paste(thumb,(x+(cell-thumb.width)//2,y))
                detail=panel.crop(crop); detail.thumbnail((cell,160))
                canvas.paste(detail,(x+(cell-detail.width)//2,y+382))
                d.rectangle((x,y+380,x+cell,y+544),outline='#b7c4d1',width=2)
                if c==0: d.text((x,y+548),filename+' | identical detail crop',font=font(20),fill='#536171')
        ramp=np.tile(np.arange(256,dtype=np.uint8),(20,1))
        bar=Image.fromarray(cv2.cvtColor(cv2.applyColorMap(ramp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)).resize((420,20))
        canvas.paste(bar,(margin,1920))
        d.text((margin,1946),'0                 Relative response                 1',font=font(19),fill='#536171')
        d.text((540,1916),'Normalized separately per map (99.5th percentile). Compare locations, not absolute magnitude.',font=font(22),fill='#536171')
        d.text((540,1950),'No artificial hotspots, no GT masking, no fabricated baseline. Native input: 1024 x 1024.',font=font(22),fill='#536171')
        canvas.save(OUT/('M1_vs_M6_'+method+'.jpg'),quality=96)
        canvas.save(OUT/('M1_vs_M6_'+method+'.png'))
    for key in SPECS:
        canvas=Image.new('RGB',(2200,2470),'white'); d=ImageDraw.Draw(canvas)
        d.text((40,24),key+' | Detection LayerCAM',font=font(54),fill='#17212b')
        d.text((40,94),'Real checkpoint gradients | P4 features | relative response',font=font(29),fill='#536171')
        for j,im in enumerate(images):
            x=30+(j%2)*1090; y=155+(j//2)*1130
            d.text((x,y),FILES[j]+'  |  '+str(len(gt[FILES[j]]))+' GT',font=font(29),fill='#17212b')
            hm,_=overlay(im,maps[key][j]['layercam'])
            hm=hm.resize((1060,1060),Image.Resampling.LANCZOS)
            canvas.paste(hm,(x,y+44))
        d.text((30,2430),'Per-map 99.5th percentile normalization. Spatial response, not detection confidence.',font=font(24),fill='#536171')
        canvas.save(OUT/(key+'_heatmap_2x2.jpg'),quality=96)
        canvas.save(OUT/(key+'_heatmap_2x2.png'))

def main():
    print('DEVICE',DEVICE,flush=True)
    images=[Image.open(ROOT/'images'/f).convert('RGB') for f in FILES]
    data=json.load(open(ROOT/'val.json')); ids={i['id']:i['file_name'] for i in data['images']}
    gt={f:[] for f in FILES}
    for a in data['annotations']:
        f=ids[a['image_id']]
        if f in gt:
            x,y,w,h=a['bbox']; gt[f].append([x,y,x+w,y+h])
    maps={}; metadata={}
    previous=json.loads((OUT/'metadata.json').read_text()) if (OUT/'metadata.json').exists() else {}
    for key in SPECS:
        model,layer,agg=load(key); maps[key]=[]; metadata[key]=[]
        for filename,im in zip(FILES,images):
            cached=OUT/(key+'_'+Path(filename).stem+'_raw.npz')
            old=next((m for m in previous.get(key,[]) if m['file']==filename),None)
            if cached.exists() and old:
                with np.load(cached) as z: maps[key].append({k:z[k] for k in z.files})
                metadata[key].append(old)
                continue
            values,dets=cam(model,layer,im); maps[key].append(values)
            metadata[key].append({'file':filename,'architecture_agg':agg,'layer':SPECS[key][3],'checkpoint':SPECS[key][1],'detections':dets})
            np.savez_compressed(OUT/(key+'_'+Path(filename).stem+'_raw.npz'),**values)
            print('CAM_DONE',key,filename,'queries',len(dets),flush=True)
        del model,layer; gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    render(images,maps,metadata,gt)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('DONE',OUT,flush=True)
if __name__=='__main__':main()
