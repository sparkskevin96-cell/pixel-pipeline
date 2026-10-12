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
    mask = sprite.rgba[..., 3] > 0
    if mask.shape != weight.shape:
        raise ValueError(f"normal grid {weight.shape} doesn't match the sprite {mask.shape}")
    # trust blocks the subject mostly covers (slivers at the edge are noisy); the outline
    # ring and the rest borrow from their neighbours
    cover = weight / (p * p)
    have = (cover >= 0.25) & (np.linalg.norm(vec, axis=-1) > 1e-3 * weight)
    vec = np.where(have[..., None], vec / np.maximum(weight, 1e-9)[..., None], 0.0)
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


def write_normal_sheet(sprites: list[Sprite], out_png, columns: int) -> Path:
    """<sheet>_n.png packed exactly like <sheet>.png (same columns, same frame rects)."""
    missing = [i for i, s in enumerate(sprites) if s.normal is None]
    if missing:
        raise ValueError("these sprites have no normal map (render with normals=True)")
    sheet, _ = pack_sheet([s.normal for s in sprites], columns)
    path = normal_sheet_path(out_png)
    Image.fromarray(sheet, "RGBA").save(path)
    return path
