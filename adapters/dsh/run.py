"""Produce reproducible renders and a local evidence packet for DeepSeek Harness."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from run_daily import arguments, render


def preview(im):
    back = Image.new("RGBA", im.size, "#eeeeee")
    return Image.alpha_composite(back, im).convert("RGB")


def evidence(args):
    with Image.open(args.input) as im:
        original = im.convert("RGBA")
    w, h = original.size
    for l, t, r, b in args.crop:
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError(f"Crop outside {w} x {h}: {(l,t,r,b)}")
    render(args.input, args.output, args.passes)
    images = [original]
    for n in range(1, args.passes + 1):
        with Image.open(args.output / f"pass-{n}" / "result.png") as im:
            images.append(im.convert("RGBA"))
    for im in images[1:]:
        if im.size != original.size or im.getchannel("A").tobytes() != original.getchannel("A").tobytes():
            raise ValueError("Size or alpha changed; review stopped.")
    # Exclude fully transparent RGB; measure displayed colour on the same ground.
    plates = [preview(im) for im in images]
    first, last = [np.asarray(p).astype(np.int16) for p in (plates[0], plates[-1])]
    delta = np.abs(last-first).mean(axis=2)
    alpha = np.asarray(original.getchannel("A")) > 0
    grid = []
    thumb = plates[0].copy()
    thumb.thumbnail((720, 900))
    draw = ImageDraw.Draw(thumb)
    for y in range(6):
        for x in range(4):
            l,t,r,b = x*w//4,y*h//6,(x+1)*w//4,(y+1)*h//6
            valid = alpha[t:b,l:r]
            if not valid.any():
                continue
            name = f"{'ABCD'[x]}{y+1}"
            grid.append(dict(cell=name, box=[l,t,r,b], mean_abs_rgb=round(float(delta[t:b,l:r][valid].mean()),2)))
            px,py = l*thumb.width//w,t*thumb.height//h
            draw.rectangle((px,py,r*thumb.width//w-1,b*thumb.height//h-1),outline="#cf426b")
            draw.rectangle((px,py,px+19,py+15),fill="white")
            draw.text((px+1,py+1),name,fill="black")
    thumb.save(args.output / "grid.png")
    ranked = sorted(grid,key=lambda cell:-cell["mean_abs_rgb"])
    boxes = args.crop or [cell["box"] for cell in ranked[:3]]
    records = []
    for number,box in enumerate(boxes,1):
        l,t,r,b = box
        scale = min(5, 480/(r-l), 560/(b-t))
        size = (max(1,round((r-l)*scale)), max(1,round((b-t)*scale)))
        sheet = Image.new("RGB", ((size[0]+6)*len(plates), size[1]+28), "white")
        draw = ImageDraw.Draw(sheet)
        for i,plate in enumerate(plates):
            draw.text((i*(size[0]+6)+3,4), "Original" if i==0 else f"Pass {i}",fill="black")
            sheet.paste(plate.crop(box).resize(size,Image.Resampling.NEAREST),(i*(size[0]+6),24))
        filename = f"crop-{number:02}.png"
        sheet.save(args.output / filename)
        records.append(dict(file=filename,box=list(box)))
    report = dict(size=[w,h],passes=args.passes,alpha_equal=True,
                  metric="mean absolute displayed RGB change, 0..255, original to last pass",
                  grid=ranked,crops=records)
    (args.output / "evidence.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Evidence: {args.output / 'evidence.json'}; inspect grid.png and crop-*.png")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--crop", type=int, nargs=4, action="append", default=[],metavar=("LEFT","TOP","RIGHT","BOTTOM"))
    evidence(parser.parse_args())
