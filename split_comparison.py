"""Split existing pairwise comparison images into individual model images,
preserving column headers and left labels."""
from PIL import Image
from pathlib import Path

OUT = Path('./compare_imgs')

# Layout constants from batch_compare.py build_pairwise
top_h, mid_h, bot_h = 65, 70, 100
ph = 1000

VARIANTS = ['M3', 'M5', 'M6']

for vname in VARIANTS:
    src = OUT / f'M1_vs_{vname}_comparison.jpg'
    if not src.exists():
        print(f"SKIP: {src} not found")
        continue

    im = Image.open(src)
    w, h = im.size
    print(f"Processing {src.name} ({w}x{h})")

    # Column headers strip (top_h px)
    header_strip = im.crop((0, 0, w, top_h))

    # Bottom strip (bot_h px from bottom)
    bottom_strip = im.crop((0, h - bot_h, w, h))

    # M1 row: y from top_h to top_h + ph
    m1_row = im.crop((0, top_h, w, top_h + ph))

    # Variant row: y from top_h + ph + mid_h to top_h + ph + mid_h + ph
    var_y0 = top_h + ph + mid_h
    var_row = im.crop((0, var_y0, w, var_y0 + ph))

    for row_img, out_name in [(m1_row, f'M1_baseline_vs_{vname}'),
                               (var_row, f'{vname}_vs_M1')]:
        # Assemble: header + row + bottom
        single_h = top_h + ph + bot_h
        single = Image.new('RGB', (w, single_h), (255, 255, 255))
        single.paste(header_strip, (0, 0))
        single.paste(row_img, (0, top_h))
        single.paste(bottom_strip, (0, top_h + ph))

        out_p = OUT / f'{out_name}.jpg'
        single.save(out_p, quality=95)
        print(f"  -> {out_p.name} ({single.size})")

print("\nDone!")
