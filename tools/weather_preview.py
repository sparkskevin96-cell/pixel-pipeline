"""Previews of the weather / water / background assets -> Pictures/PixelPipeline:
weather_preview.png + .gif (every weather over a matching background), water_preview.gif,
parallax_preview.png (4 times of day).

    python tools/weather_preview.py [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pixel_pipeline import backgrounds as B, water as WA, weather as W

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Pictures" / "PixelPipeline"
CELL = (240, 135)
K = 2
# weather -> (time of day, background scene); None = a top-down grass map (cloud shadows)
PAIRS = {"rain": ("night", "forest"), "snow": ("dawn", "snowfield"), "fog": ("dawn", "mountains"),
         "god_rays": ("sunset", "forest"), "leaves": ("sunset", "autumn"), "petals": ("noon", "mountains"),
         "embers": ("night", "desert"), "fireflies": ("night", "forest"), "dust": ("noon", "desert"),
         "sandstorm": ("noon", "desert"), "lightning": ("night", "coast"), "cloud_shadows": ("noon", None)}


def backdrop(tod, scene):
    if scene is None:
        from pixel_pipeline.tilegen import builtin_swatch
        tex = builtin_swatch("sand", 32, 0).texture
        wt = WA.render_water(32, frames=2)
        img = np.tile(tex, (5, 8, 1))[:CELL[1], :CELL[0]].copy()
        lake = WA.example_map(wt, 0, 8, 5)[:CELL[1], :CELL[0]]
        m = lake[..., 2] > lake[..., 0] + 30
        img[m] = lake[m]
        return img
    return B.render_background(scene, tod, width=CELL[0], height=CELL[1], seed=2).composite()


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 6 * len(text) + 8, 14), fill=(10, 10, 16))
    d.text((4, 2), text, fill=(235, 235, 245))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = 4
    items = []
    for kind, (tod, scene) in PAIRS.items():
        w = W.render_weather(kind, tod=tod)
        items.append((kind, tod, w, backdrop(tod, scene)))
        print(kind, len(w.fx.frames), "frames", w.size)
    rows = -(-len(items) // cols)
    cw, ch = CELL[0] * K, CELL[1] * K

    def sheet(t_ms):
        canvas = Image.new("RGB", (cols * cw, rows * ch), (10, 10, 16))
        for j, (kind, tod, w, bg) in enumerate(items):
            durs = w.fx.frame_durations()
            cyc = sum(durs) + (0 if w.fx.loop else 900)
            tt, acc, i = t_ms % cyc, 0, len(durs) - 1
            for k, ms in enumerate(durs):
                if tt < acc + ms:
                    i = k
                    break
                acc += ms
            if not w.fx.loop and tt >= sum(durs):
                img = bg[..., :3]
            else:
                img = W.preview_over(w, bg, i, offset=(0, 0))
            im = Image.fromarray(img).resize((cw, ch), Image.NEAREST)
            label(im, f"{kind} ({tod})")
            canvas.paste(im, ((j % cols) * cw, (j // cols) * ch))
        return canvas

    still = sheet(300)
    still.save(OUT / "weather_preview.png")
    frames = [sheet(t) for t in range(0, 2400, 80)]
    pal = [f.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(OUT / "weather_preview.gif", save_all=True, append_images=pal[1:], duration=80, loop=0)

    # water: four themes as little lakes + side / iso strips
    themes = (("ocean", None), ("lagoon", "noon"), ("ocean", "sunset"), ("swamp", None), ("lava", None), ("night", "night"))
    wts = [WA.render_water(32, theme=t, tod=tod, frames=8) for t, tod in themes]
    wframes = []
    for k in range(8):
        row = []
        for (t, tod), wt in zip(themes, wts):
            lake = WA.example_map(wt, k, 8, 5)
            side = np.concatenate([np.concatenate([wt.side_surface[k], wt.side_body[k]], 0)] * 8, 1)
            d = wt.iso[k]
            dh, dw = d.shape[:2]
            iso = np.zeros((dh * 3, dw * 8, 4), np.uint8)
            for r in range(5):                                   # a diamond lattice
                for c in range(-1, 9):
                    x, y = c * dw + (r % 2) * dw // 2, r * dh // 2
                    xs, ys = max(0, x), max(0, y)
                    xe, ye = min(iso.shape[1], x + dw), min(iso.shape[0], y + dh)
                    if xe <= xs or ye <= ys:
                        continue
                    part = d[ys - y:ye - y, xs - x:xe - x]
                    m = part[..., 3] > 0
                    iso[ys:ye, xs:xe][m] = part[m]
            cell = np.zeros((lake.shape[0] + side.shape[0] + iso.shape[0] + 8, lake.shape[1], 3), np.uint8)
            cell[:] = (16, 16, 24)
            y = 0
            for part in (lake, side, iso):
                a = part[..., 3:4] / 255
                cell[y:y + part.shape[0], :part.shape[1]] = (part[..., :3] * a + cell[y:y + part.shape[0], :part.shape[1]] * (1 - a)).astype(np.uint8)
                y += part.shape[0] + 4
            im = Image.fromarray(cell)
            label(im, f"{t}{' ' + tod if tod else ''}")
            row.append(np.asarray(im))
        img = np.concatenate([np.pad(r, ((0, 0), (0, 4), (0, 0))) for r in row], 1)
        wframes.append(Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST))
    wframes[0].save(OUT / "water_preview.png")
    wframes[0].save(OUT / "water_preview.gif", save_all=True, append_images=wframes[1:], duration=140, loop=0)

    # parallax: 4 times of day x 3 scenes
    rows_img = []
    for scene in ("mountains", "coast", "desert"):
        row = []
        for tod in B.TODS:
            c = B.render_background(scene, tod, seed=1).composite()[..., :3]
            im = Image.fromarray(c)
            label(im, f"{scene} {tod}")
            row.append(np.pad(np.asarray(im), ((0, 4), (0, 4), (0, 0))))
        rows_img.append(np.concatenate(row, 1))
    img = np.concatenate(rows_img, 0)
    Image.fromarray(img).save(OUT / "parallax_preview.png")
    bg = B.render_background("mountains", "sunset", seed=1)
    pf = [Image.fromarray(bg.at_scroll(x)[..., :3]) for x in range(0, 480, 6)]
    pf[0].save(OUT / "parallax_preview.gif", save_all=True, append_images=pf[1:], duration=50, loop=0)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
