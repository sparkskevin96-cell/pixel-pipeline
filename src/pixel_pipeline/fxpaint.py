"""Optional GPU pass for effects: SDXL img2img "paints over" a procedural effect for a
hand-painted feel, then everything goes back onto the effect's own pixels.

The frames are packed into one square-ish grid (one ComfyUI call, so every frame gets the same
brushwork), painted at low denoise with the project's house style (assets.with_style kind
"effect"), cut back into frames, box-shrunk to the frame size and snapped to the effect's palette
(Lab) inside the ORIGINAL silhouette: shapes, timing, alpha and glow stay exactly as drawn.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

PAINT_PROMPT = ("hand-painted pixel art {what} spell effect, glowing magic, vibrant hue-shifted "
                "colours, crisp clean shapes, game vfx sprite sheet, dark background")
PAINT_NEGATIVE = "blurry, photo, 3d render, text, watermark, muddy, noisy"
CELL = 128                       # each frame is painted at this size (or bigger, whole number)


def _grid(frames, bg=(10, 8, 16)):
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = frames[0].shape[:2]
    k = max(1, CELL // max(w, h))
    sheet = Image.new("RGB", (cols * w * k, rows * h * k), bg)
    for i, f in enumerate(frames):
        im = Image.new("RGB", (w, h), bg)
        im.paste(Image.fromarray(f, "RGBA"), (0, 0), Image.fromarray(f, "RGBA"))
        sheet.paste(im.resize((w * k, h * k), Image.NEAREST), ((i % cols) * w * k, (i // cols) * h * k))
    return sheet, cols, k


def paint_effect(effect, *, client=None, style=None, seed: int = 0, denoise: float = 0.38,
                 prompt: str | None = None, **gen_opts):
    """effects.Effect -> a new Effect with painted colours (same frames' alpha, palette =
    the effect's own colours). client: generate.ComfyClient (default: the local ComfyUI)."""
    from .assets import project_style, with_style
    from .color import rgb_to_lab
    from .effects import Effect
    from .generate import generate_images
    st = style if style is not None else project_style()
    what = effect.kind.replace("_", " ")
    pos, neg = with_style(prompt or PAINT_PROMPT.format(what=what), PAINT_NEGATIVE, st, "effect")
    grid, cols, k = _grid(effect.frames)
    (img, _), = generate_images(pos, neg, client=client, seed=seed, init_image=np.asarray(grid.convert("RGBA")),
                                denoise=denoise, aspect=grid.width / grid.height, **gen_opts)
    img = img.convert("RGB").resize(grid.size, Image.LANCZOS)
    h, w = effect.frames[0].shape[:2]
    pal = effect.palette.astype(np.float32)
    pal_lab = rgb_to_lab(pal)
    out = []
    for i, f in enumerate(effect.frames):
        x, y = (i % cols) * w * k, (i // cols) * h * k
        cell = img.crop((x, y, x + w * k, y + h * k)).resize((w, h), Image.BOX)
        rgb = np.asarray(cell, np.float32).reshape(-1, 3)
        lab = rgb_to_lab(rgb)
        idx = ((lab[:, None] - pal_lab[None]) ** 2).sum(-1).argmin(1).reshape(h, w)
        g = f.copy()
        on = f[..., 3] > 0
        g[on, :3] = effect.palette[idx[on]]
        out.append(g)
    fx = Effect(effect.kind, out, effect.palette, effect.frame_ms, effect.loop, effect.theme, effect.ramp,
                effect.rim, durations=effect.durations, glow=effect.glow, blend=effect.blend, tags=effect.tags)
    return fx
