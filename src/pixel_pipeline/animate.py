"""Procedural idle animations from a single sprite (pure pixel moves, palette-safe).

breathe  upper body squashes down by `amount` px and back (characters, animals)
bob      whole sprite floats up and down (pickups, flying things)
sway     top leans left/right, bottom stays planted (plants, trees, hair, flags)
rumble   body shakes over planted wheels/base (vehicles, machines idling)
"""
from __future__ import annotations

import numpy as np

IDLE_STYLES = ("breathe", "bob", "sway", "rumble")
IDLE_MS = {"breathe": 150, "bob": 150, "sway": 120, "rumble": 70}  # good default frame times
SUBJECT_IDLE = {"person": "breathe", "animal": "breathe", "object": "bob", "background": "sway"}


def _wave(frames: int, amount: int) -> list[int]:
    """0 -> amount -> 0 over the loop, e.g. 4 frames, 1px: [0, 1, 1, 0]."""
    t = (np.arange(frames) + 0.5) / frames
    return [int(v) for v in np.rint(amount * (1 - np.cos(2 * np.pi * t)) / 2)]


def _opaque_rows(rgba: np.ndarray) -> tuple[int, int]:
    rows = np.nonzero(rgba[..., 3].any(1))[0]
    if not len(rows):
        raise ValueError("Sprite is fully transparent")
    return int(rows[0]), int(rows[-1])


def _pad(rgba: np.ndarray, top=0, bottom=0, left=0, right=0) -> np.ndarray:
    return np.pad(rgba, ((top, bottom), (left, right), (0, 0)))


def breathe(rgba: np.ndarray, frames: int = 4, amount: int = 1, pivot: float = 0.45) -> list[np.ndarray]:
    """Remove `offset` rows at the chest (pivot = fraction down the body) and drop
    everything above it, so feet and legs stay still while the torso breathes."""
    top, bottom = _opaque_rows(rgba)
    cut = top + int(round((bottom - top) * pivot))
    out = []
    for off in _wave(frames, amount):
        f = rgba.copy()
        if off:
            f[top + off:cut + 1] = rgba[top:cut + 1 - off]
            f[:top + off] = 0
        out.append(f)
    return out


def bob(rgba: np.ndarray, frames: int = 4, amount: int = 1) -> list[np.ndarray]:
    top, _ = _opaque_rows(rgba)
    room = max(0, amount - top)
    base = _pad(rgba, top=room)
    out = []
    for off in _wave(frames, amount):
        f = np.zeros_like(base)
        h = base.shape[0]
        f[:h - off] = base[off:]
        out.append(f)
    return out


def sway(rgba: np.ndarray, frames: int = 8, amount: int = 1, pivot: float = 0.75) -> list[np.ndarray]:
    """Shear rows above `pivot` (fraction down the body); the top moves `amount` px."""
    top, bottom = _opaque_rows(rgba)
    base = _pad(rgba, left=amount, right=amount)
    anchor = top + (bottom - top) * pivot
    out = []
    for i in range(frames):
        lean = amount * np.sin(2 * np.pi * i / frames)
        f = base.copy()
        for y in range(top, int(anchor)):
            dx = int(np.rint(lean * (anchor - y) / max(1.0, anchor - top)))
            if dx:
                f[y] = np.roll(base[y], dx, axis=0)
        out.append(f)
    return out


def rumble(rgba: np.ndarray, frames: int = 2, amount: int = 1, pivot: float = 0.7) -> list[np.ndarray]:
    """Engine idle: everything above `pivot` (fraction down the body) hops up `amount` px on
    alternate frames, stretching the row above the wheels so the body stays attached."""
    top, bottom = _opaque_rows(rgba)
    room = max(0, amount - top)
    base = _pad(rgba, top=room)
    top, bottom = top + room, bottom + room
    cut = top + int(round((bottom - top) * pivot))
    out = []
    for i in range(frames):
        off = amount if i % 2 else 0
        f = base.copy()
        if off:
            f[top - off:cut + 1 - off] = base[top:cut + 1]
            f[cut + 1 - off:cut + 1] = base[cut]
        out.append(f)
    return out


def idle_frames(rgba: np.ndarray, style: str = "breathe", frames: int | None = None,
                amount: int = 1) -> list[np.ndarray]:
    if style == "breathe":
        return breathe(rgba, frames or 4, amount)
    if style == "bob":
        return bob(rgba, frames or 4, amount)
    if style == "sway":
        return sway(rgba, frames or 8, amount)
    if style == "rumble":
        return rumble(rgba, frames or 2, amount)
    raise ValueError(f"Unknown idle style '{style}' (use {', '.join(IDLE_STYLES)})")
