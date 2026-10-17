"""Normal maps for rendered sprites, so lights in a 3D (HD-2D) scene shade them like 3D.

The splat renderer (turnaround._splat) writes, next to each colour view, the camera-space
normal of the point it kept per pixel, encoded the way Godot reads normal maps (OpenGL
convention): RGB = n * 0.5 + 0.5 with x right, y up, z toward the camera; alpha = the colour
view's alpha. Normals always face the camera (z >= 0): meshes may have flipped normals.

normal_sprites() then takes those views through exactly the geometry the colour pipeline
uses (pipeline.prepare with crop_to_subject off, as turnaround_sprites does: resolve_size,
the outline inset, _fit_box's scale and placement, then patch x patch blocks), averaging the
vectors in each block and renormalising instead of snapping to a palette. The mask is the
finished colour sprite's alpha, so pixel (x, y) of the normal sprite is pixel (x, y) of the
colour sprite; the outline ring borrows its normal from the pixels it touches.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .export import pack_sheet
from .pipeline import Sprite, _fit_box
from .settings import Settings

FLAT = np.array([128, 128, 255], np.uint8)   # (0, 0, 1): facing the camera


def encode(n: np.ndarray) -> np.ndarray:
    """Unit normals (..., 3) -> uint8 RGB, n * 0.5 + 0.5."""
    return np.clip(np.rint(n * 127.5 + 127.5), 0, 255).astype(np.uint8)


def decode(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 vectors in -1..1 (not renormalised)."""
    return rgb.astype(np.float32) / 127.5 - 1.0


def face_camera(n: np.ndarray) -> np.ndarray:
    """Unit length, flipped where z < 0 (flipped mesh normals)."""
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return np.where(n[..., 2:3] < 0, -n, n)


def _fill_from_neighbours(vec: np.ndarray, have: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Give `want` pixels without a normal the mean of their neighbours' (repeatedly)."""
    vec, have = vec.copy(), have.copy()
    k = np.ones((3, 3), np.float32)
    for _ in range(max(vec.shape[:2])):
        todo = want & ~have
        if not todo.any():
            break
        w = cv2.filter2D(have.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        s = np.stack([cv2.filter2D(vec[..., c] * have, -1, k, borderType=cv2.BORDER_CONSTANT)
                      for c in range(3)], -1)
        new = todo & (w > 0)
        if not new.any():
            break
        vec[new] = s[new] / w[new][:, None]
        have |= new
    vec[want & ~have] = (0.0, 0.0, 1.0)
    return vec


def normal_sprite(view: np.ndarray, sprite: Sprite, settings: Settings) -> np.ndarray:
    """One encoded normal view (same size and framing as the colour view the sprite was made
    from) -> (H, W, 4) normal sprite on the sprite's grid and mask."""
    return normal_from_blocks(*normal_blocks(view, settings), sprite)


def normal_blocks(view: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """normal_sprite's first half, which needs only the view: (mean vectors, trusted) per
    sprite pixel. The second half (normal_from_blocks) needs the finished sprite's mask."""
    s = settings
    sh, sw = view.shape[:2]
    out_w, out_h = s.resolve_size(sw, sh)
    inset = 1 if s.outline and min(out_w, out_h) > 4 else 0
    cw, ch, nw, nh, x, y = _fit_box(sw, sh, out_w, out_h, inset, s)
    a = view[..., 3].astype(np.float32) / 255
    premul = decode(view[..., :3]) * a[..., None]
    interp = cv2.INTER_AREA if nw < sw else cv2.INTER_LINEAR
    a = np.clip(cv2.resize(a, (nw, nh), interpolation=interp), 0, 1).reshape(nh, nw)
    v = cv2.resize(premul, (nw, nh), interpolation=interp).reshape(nh, nw, 3)
    sx, sy, dx, dy = max(0, -x), max(0, -y), max(0, x), max(0, y)
    w, h = min(nw - sx, cw - dx), min(nh - sy, ch - dy)
    canvas = np.zeros((ch, cw, 4), np.float32)
    canvas[dy:dy + h, dx:dx + w, :3] = v[sy:sy + h, sx:sx + w]
    canvas[dy:dy + h, dx:dx + w, 3] = a[sy:sy + h, sx:sx + w]
    p = s.patch
    gh, gw = ch // p, cw // p
    blocks = canvas[:gh * p, :gw * p].reshape(gh, p, gw, p, 4).sum((1, 3))
    weight, vec = blocks[..., 3], blocks[..., :3]
    # trust blocks the subject mostly covers (slivers at the edge are noisy); the outline
    # ring and the rest borrow from their neighbours
    cover = weight / (p * p)
    have = (cover >= 0.25) & (np.linalg.norm(vec, axis=-1) > 1e-3 * weight)
    vec = np.where(have[..., None], vec / np.maximum(weight, 1e-9)[..., None], 0.0)
    return vec, have


def normal_from_blocks(vec: np.ndarray, have: np.ndarray, sprite: Sprite) -> np.ndarray:
    """normal_sprite's second half: the blocks on the finished sprite's mask, encoded."""
    mask = sprite.rgba[..., 3] > 0
    if mask.shape != have.shape:
        raise ValueError(f"normal grid {have.shape} doesn't match the sprite {mask.shape}")
    vec = face_camera(_fill_from_neighbours(vec, have & mask, mask))
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask, :3] = encode(vec[mask])
    out[mask, 3] = 255
    return out


def attach_normals(sprites: list[Sprite], views: list[np.ndarray], settings: Settings) -> None:
    """Set sprite.normal for every sprite from its normal view (in place)."""
    s = Settings(**{**settings.to_dict(), "crop_to_subject": False})
    for sp, v in zip(sprites, views):
        sp.normal = normal_sprite(v, sp, s)


def normal_sheet_path(out_png) -> Path:
    out = Path(out_png)
    return out.with_name(out.stem + "_n.png")


def write_normal_sheet(sprites: list[Sprite], out_png, columns: int, rows: list[int] | None = None) -> Path:
    """<sheet>_n.png packed exactly like <sheet>.png (same columns, same frame rects). rows: the
    frames in each sheet row when rows differ (anim3d.write_animated: 12-frame walks, 8-frame
    idles) - then packed row by row like the sheet."""
    missing = [i for i, s in enumerate(sprites) if s.normal is None]
    if missing:
        raise ValueError("these sprites have no normal map (render with normals=True)")
    if rows is not None:                       # row by row like the sheet (pack_rows wraps tall ones)
        from .export import pack_rows
        maps, at, lines = [s.normal for s in sprites], 0, []
        for n in rows:
            lines.append(maps[at:at + n])
            at += n
        sheet, _ = pack_rows(lines)
    else:
        sheet, _ = pack_sheet([s.normal for s in sprites], columns)
    path = normal_sheet_path(out_png)
    Image.fromarray(sheet, "RGBA").save(path)
    return path


def from_drawing(rgba: np.ndarray, bevel: float = 3.0, relief: float = 0.6) -> np.ndarray:
    """A normal map for a drawn sprite (no 3D behind it): the silhouette is bevelled (normals
    tilt outward near the edge) and brightness adds surface relief, so scene lights still shade
    it (HD-2D, Godot/Unity 2D lights). RGBA, OpenGL convention, alpha = the sprite's."""
    import cv2
    a = (rgba[..., 3] >= 128).astype(np.uint8)
    dist = cv2.distanceTransform(a, cv2.DIST_L2, 3).astype(np.float32)
    height = np.clip(dist / max(bevel, 1e-3), 0, 1)
    lum = rgba[..., :3].astype(np.float32).mean(-1) / 255.0
    height = height + relief * cv2.GaussianBlur(lum, (0, 0), 0.8) * (dist > 0)
    gy, gx = np.gradient(height)
    n = np.dstack([-gx, gy, np.full_like(gx, 0.35)])        # y up in the normal map
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    out = np.dstack([encode(face_camera(n)), rgba[..., 3]])
    out[a == 0] = 0
    return out.astype(np.uint8)


def material_labels(rgba: np.ndarray, palette_lab: np.ndarray | None = None, ab_step: float = 14.0,
                    l_step: float = 34.0) -> np.ndarray:
    """A material id per opaque pixel: colours binned by hue (Lab a/b) and coarse lightness, so
    the shades of one material (a cape's 4 reds) share an id and its neighbours (skin, steel)
    don't. -1 = see-through."""
    from .color import rgb_to_lab
    on = rgba[..., 3] > 0
    lab = rgb_to_lab(rgba[..., :3])
    key = (np.floor(lab[..., 1] / ab_step).astype(np.int32) * 1000
           + np.floor(lab[..., 2] / ab_step).astype(np.int32) * 10
           + np.floor(lab[..., 0] / l_step).astype(np.int32))
    return np.where(on, key, -1)


def smooth_by_material(normal: np.ndarray, rgba: np.ndarray, radius: int = 1, passes: int = 2,
                       keep: float = 0.35) -> np.ndarray:
    """A sprite normal map averaged over each pixel's neighbours OF THE SAME MATERIAL
    (material_labels), `passes` times: the noise of a bent drawing's normals (every pixel
    pointing its own way -> blotchy light) goes, while the edges between materials (where the
    shape folds: a cape over armour) stay. keep = share of the pixel's own normal kept."""
    import cv2
    on = rgba[..., 3] > 0
    if not on.any():
        return normal
    lab = material_labels(rgba)
    vec = decode(normal[..., :3]).astype(np.float32) * on[..., None]
    h, w = on.shape
    for _ in range(passes):
        acc = vec * keep
        wsum = np.full((h, w), keep, np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dy, dx) == (0, 0):
                    continue
                sh_v = np.roll(np.roll(vec, dy, 0), dx, 1)
                sh_l = np.roll(np.roll(lab, dy, 0), dx, 1)
                ok = (sh_l == lab) & on
                # no wrap-around at the borders
                if dy > 0: ok[:dy] = False
                if dy < 0: ok[dy:] = False
                if dx > 0: ok[:, :dx] = False
                if dx < 0: ok[:, dx:] = False
                acc += sh_v * ok[..., None]
                wsum += ok
        vec = acc / wsum[..., None]
        vec /= np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
        vec *= on[..., None]
    out = normal.copy()
    out[..., :3] = np.where(on[..., None], encode(face_camera(vec)), normal[..., :3])
    return out


def smooth_sheet(sheet_png, normal_png=None, out=None) -> Path | None:
    """smooth_by_material over a whole sheet + its _n.png (frames are separate islands of
    pixels, so neighbours never cross frames that don't touch). Writes `out` (default: the
    normal map in place) and returns it; None when the sheet has no normal map."""
    sheet_png = Path(sheet_png)
    normal_png = Path(normal_png) if normal_png else normal_sheet_path(sheet_png)
    if not normal_png.exists():
        return None
    rgba = np.asarray(Image.open(sheet_png).convert("RGBA"))
    nm = np.asarray(Image.open(normal_png).convert("RGBA"))
    if nm.shape[:2] != rgba.shape[:2]:
        return None
    sm = smooth_by_material(nm, rgba)
    dst = Path(out) if out else normal_png
    Image.fromarray(sm).save(dst)
    return dst
