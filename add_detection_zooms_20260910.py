"""Add identical, traceable crops to existing four-panel detection figures.
Does not change predictions or invent pixels. Requires Pillow.
"""
from pathlib import Path
import argparse, json
from PIL import Image, ImageDraw, ImageFont, ImageOps

SOURCES=['M1_baseline_vs_M3','M3_vs_M1','M5_vs_M1','M6_vs_M1']
SCENES=['Dense','Sparse','Medium','Mixed']
ROIS=[[(.38,.37,.72,.46)],[(.15,.485,.31,.55),(.75,.405,.90,.49)],
      [(.135,.375,.35,.455),(.535,.385,.73,.525)],[(.62,.345,.96,.445)]]
COLORS=['#D34C24','#007EAC']
def font(n):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,n)
    return ImageFont.load_default(size=n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).parent);p.add_argument('--output',type=Path)
    a=p.parse_args(); out=a.output or a.input/'paper_zoom';out.mkdir(parents=True,exist_ok=True)
    for name in SOURCES:
        src=Image.open(a.input/(name+'.jpg')).convert('RGB')
        if src.size!=(4870,5320):raise ValueError((name,src.size,'Unexpected source layout'))
        canvas=Image.new('RGB',(2860,3940),'white'); d=ImageDraw.Draw(canvas)
        title='M1 | Existing baseline illustration' if name.startswith('M1_') else name.split('_')[0]+' | Detection results with enlarged details'
        d.text((40,25),title,font=font(52),fill='#202B36')
        d.text((40,94),'Outlined regions correspond to the detail panels below each scene.',font=font(30),fill='#596674')
        for j in range(4):
            sx=20+(j%2)*2430; sy=300+(j//2)*2530
            panel=src.crop((sx,sy,sx+2400,sy+2400))
            x=40+(j%2)*1420; y=160+(j//2)*1835
            d.text((x,y),f'({chr(97+j)}) {SCENES[j]}',font=font(36),fill='#202B36')
            full=panel.resize((1360,1360),Image.Resampling.LANCZOS)
            canvas.paste(full,(x,y+52))
            for k,roi in enumerate(ROIS[j]):
                color=COLORS[k];label=chr(65+k)
                box=tuple(round(t*2400) for t in roi)
                rect=(x+round(roi[0]*1360),y+52+round(roi[1]*1360),x+round(roi[2]*1360),y+52+round(roi[3]*1360))
                d.rectangle(rect,outline=color,width=5)
                d.rectangle((rect[0],rect[1]-39,rect[0]+38,rect[1]),fill=color)
                d.text((rect[0]+6,rect[1]-38),label,font=font(30),fill='white')
                n=len(ROIS[j]); cw=1360 if n==1 else 665
                cx=x+k*695;cy=y+1432
                detail=ImageOps.contain(panel.crop(box),(cw-20,292),Image.Resampling.LANCZOS)
                d.rectangle((cx,cy,cx+cw,cy+350),outline=color,width=4)
                d.text((cx+13,cy+9),'Detail '+label,font=font(28),fill=color)
                canvas.paste(detail,(cx+(cw-detail.width)//2,cy+48+(292-detail.height)//2))
        note='Crops are identical across M1/M3/M5/M6; original detections and confidence labels are unchanged.'
        d.text((40,3856),note,font=font(25),fill='#596674')
        if name.startswith('M1_'):
            d.text((40,3895),'M1 source uses simulated baseline detections; this figure is NOT independent M1 inference.',font=font(25),fill='#A63B25')
        canvas.save(out/(name+'_zoom.jpg'),quality=97,subsampling=0,dpi=(300,300))
        canvas.save(out/(name+'_zoom.png'),dpi=(300,300))
        print(name,'done',flush=True)
    (out/'crop_regions.json').write_text(json.dumps({'source_size':[4870,5320],'normalized_panel_rois':dict(zip(SCENES,ROIS))},indent=2))
if __name__=='__main__':main()
