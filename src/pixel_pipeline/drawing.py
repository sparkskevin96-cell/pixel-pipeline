"""Sprites drawn as text: a palette plus rows of palette-index characters.

This is how Claude draws a sprite itself (the MCP `draw_sprite` tool and the phone
app's "Draw from a prompt"):

    palette = ["#1a1c2c", "#b13e53", "#ffcd75"]
    rows    = ["..000..",
               ".01110.",
               "0112110", ...]       # "." transparent, 0-9 then a-z = palette index
"""
from __future__ import annotations

import numpy as np

from .palettes import hex_to_rgb
from .pipeline import Sprite, sprite_from_indices
from .quantize import nearest
from .color import rgb_to_lab


def parse_rows(rows: list[str], palette: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """-> (indices (H, W) int16 with -1 = transparent, palette (N, 3) uint8)."""
    pal = hex_to_rgb([str(c).lstrip("#")[:6] for c in palette])
    if not len(pal):
        raise ValueError("palette is empty")
    if not rows:
        raise ValueError("rows is empty")
    h, w = len(rows), max(len(r) for r in rows)
    idx = np.full((h, w), -1, np.int16)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in ". ":
                continue
            try:
                i = int(ch, 36)
            except ValueError:
                raise ValueError(f"row {y} has '{ch}': use '.' or 0-9 / a-z palette indexes") from None
            if i >= len(pal):
                raise ValueError(f"row {y} uses index '{ch}' but the palette has {len(pal)} colours")
            idx[y, x] = i
    return idx, pal


def sprite_from_rows(rows: list[str], palette: list[str], system: str = "off",
                     subject: str = "object") -> Sprite:
    """Build a sprite; with a retro system, colours are snapped to what it can show."""
    idx, pal = parse_rows(rows, palette)
    if system in (None, "off"):
        return sprite_from_indices(idx, pal)
    from .systems import SYSTEMS, system_palette
    mask = idx >= 0
    rgb = np.zeros(idx.shape + (3,), np.uint8)
    rgb[mask] = pal[idx[mask]]
    sys_pal = system_palette(system, rgb[mask], SYSTEMS[system].max_colors(subject), subject)
    new = np.full(idx.shape, -1, np.int16)
    new[mask] = nearest(rgb_to_lab(rgb[mask]), rgb_to_lab(sys_pal))
    return sprite_from_indices(new, sys_pal)
