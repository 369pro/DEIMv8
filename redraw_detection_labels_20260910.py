"""Re-infer all four checkpoints; draw small collision-free crop labels."""
import json, gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps
import heatmap_clean_20260909 as h
from add_detection_zooms_20260910 import ROIS, SCENES
h.DEVICE='cpu';torch.set_num_threads(4)
h.SPECS['M3']=('configs/test/dfine_hgnetv2_test.yml','outputs/dfine-n-c2f/best_stg2.pth','configs/cfg-improve/dfine-n-C2f-IDWB.yaml','backbone.3')
h.SPECS['M5']=('configs/test/dfine_hgnetv2_test.yml','outputs/dfine-n-c2f-mfm-lsk/best_stg2.pth','configs/cfg-improve/dfine-n-C2f-IDWB-MFM-LSK.yaml','backbone.3')
OUT=Path('compare_imgs/paper_zoom_small_labels');OUT.mkdir(exist_ok=True)
def ft(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def intersects(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]
def draw_panel(im,dets,size,crop=None,fontsize=23):
    crop=crop or (0,0,im.width,im.height)
    base=im.crop(crop);thumb=ImageOps.contain(base,size,Image.Resampling.LANCZOS)
    result=Image.new('RGB',size,'#f7f9fb');ox=(size[0]-thumb.width)//2;oy=(size[1]-thumb.height)//2
    result.paste(thumb,(ox,oy));dr=ImageDraw.Draw(result);scale=thumb.width/(crop[2]-crop[0]);occupied=[]
    selected=[]
    for det in dets:
        x1,y1,x2,y2=det['box']
        if not (crop[0] <= (x1+x2)/2 < crop[2] and crop[1] <= (y1+y2)/2 < crop[3]):continue
        b=[ox+(x1-crop[0])*scale,oy+(y1-crop[1])*scale,ox+(x2-crop[0])*scale,oy+(y2-crop[1])*scale]
        b=[max(0,min(size[i%2]-1,t)) for i,t in enumerate(b)]
        selected.append((b,det))
    palette=['#bf382b','#126fb5','#11865c','#8b48a5','#b36600']
    for b,det in selected:dr.rectangle(b,outline=palette[det['class']%len(palette)],width=2)
    for b,det in sorted(selected,key=lambda x:x[0][0]):
        text=det['label']+' '+format(det['score'],'.2f');f=ft(fontsize);bb=dr.textbbox((0,0),text,font=f);w=bb[2]+10;hh=fontsize+10
        color=palette[det['class']%len(palette)];candidates=[]
        for dy in range(0,8):
            for yy in [b[1]-(dy+1)*(hh+5),b[3]+dy*(hh+5)+5]:
                for xx in [b[0],b[2]-w,(b[0]+b[2]-w)/2]:
                    xx=max(2,min(size[0]-w-2,xx));yy=max(2,min(size[1]-hh-2,yy))
                    candidates.append((xx,yy,xx+w,yy+hh))
        # Guaranteed non-overlap fallback: scan remaining label slots.
        candidates += [(x,y,x+w,y+hh) for y in range(2,size[1]-hh,hh+5) for x in range(2,size[0]-int(w),max(1,int(w)+5))]
        chosen=next((r for r in candidates if not any(intersects((r[0]-3,r[1]-3,r[2]+3,r[3]+3),q) for q in occupied)),None)
        if chosen is None:raise RuntimeError('No space for label '+text)
        occupied.append(chosen);x,y,_,_=chosen
        dr.line(((b[0]+b[2])/2,b[1],x+w/2,y+hh/2),fill=color,width=1)
        dr.rectangle(chosen,fill='white',outline=color,width=1)
        dr.text((x+5,y+3),text,font=f,fill=color,stroke_width=0)
    return result

def main():
    data=json.load(open(h.ROOT/'val.json'));cats={int(c['id']):c['name'] for c in data['categories']}
    print('CATEGORIES',cats,flush=True)
    images=[Image.open(h.ROOT/'images'/f).convert('RGB') for f in h.FILES]
    from engine.deim.postprocessor import PostProcessor
    post=PostProcessor(num_classes=45,use_focal_loss=True,num_top_queries=300).deploy()
    from torchvision.ops import nms
    all_dets={}
    for key in ['M1','M3','M5','M6']:
        model,_,_=h.load(key);all_dets[key]=[]
        for im,filename in zip(images,h.FILES):
            x=torch.from_numpy(np.asarray(im.resize((1024,1024))).copy()).permute(2,0,1).float()[None]/255
            with torch.no_grad(): lab,boxes,scores=post(model(x),torch.tensor([[im.width,im.height]]))
            lab,boxes,scores=lab[0],boxes[0],scores[0];keep=scores>=.25;lab,boxes,scores=lab[keep],boxes[keep],scores[keep]
            inds=nms(boxes,scores,.5)
            dets=[{'class':int(lab[i]),'label':cats[int(lab[i])],'score':float(scores[i]),'box':boxes[i].tolist()} for i in inds]
            all_dets[key].append(dets);print('INFERRED',key,filename,len(dets),flush=True)
        del model;gc.collect()
    (OUT/'detections.json').write_text(json.dumps(all_dets,indent=2))
    render(images,all_dets)

def render(images,all_dets):
    for key in ['M1','M3','M5','M6']:
        canvas=Image.new('RGB',(2860,4140),'white');d=ImageDraw.Draw(canvas)
        d.text((40,24),key+' | Detection details',font=ft(49),fill='#202b36')
        d.text((40,91),'Real checkpoint predictions | score >= 0.25 | identical crops across models',font=ft(29),fill='#596674')
        for j,(im,dets) in enumerate(zip(images,all_dets[key])):
            x=40+j%2*1420;y=150+j//2*1950
            d.text((x,y),f'({chr(97+j)}) {SCENES[j]}',font=ft(35),fill='#202b36')
            full=draw_panel(im,dets,(1360,1360),fontsize=15)
            canvas.paste(full,(x,y+50))
            for k,roi in enumerate(ROIS[j]):
                color=['#d34c24','#007eac'][k];letter=chr(65+k)
                rect=(x+roi[0]*1360,y+50+roi[1]*1360,x+roi[2]*1360,y+50+roi[3]*1360)
                d.rectangle(rect,outline=color,width=4)
                d.rectangle((rect[0],rect[1]-35,rect[0]+35,rect[1]),fill=color);d.text((rect[0]+5,rect[1]-34),letter,font=ft(27),fill='white')
                cw=1360 if len(ROIS[j])==1 else 665;cx=x+k*695;cy=y+1430
                crop=tuple(int(r*(im.width if i%2==0 else im.height)) for i,r in enumerate(roi))
                zoom=draw_panel(im,dets,(cw-12,420),crop=crop,fontsize=25)
                canvas.paste(zoom,(cx+6,cy+40));d.rectangle((cx,cy,cx+cw,cy+470),outline=color,width=3)
                d.text((cx+12,cy+6),'Detail '+letter,font=ft(25),fill=color)
        d.text((40,4100),'Labels redrawn independently in each view; confidence values are unchanged model outputs.',font=ft(25),fill='#596674')
        canvas.save(OUT/(key+'_zoom_small_labels.jpg'),quality=97,subsampling=0,dpi=(300,300))
        canvas.save(OUT/(key+'_zoom_small_labels.png'),dpi=(300,300))
        print('SAVED',key,flush=True)
if __name__=='__main__':main()
