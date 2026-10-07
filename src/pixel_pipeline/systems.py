"""Retro system looks: one setting that applies a console's sprite limits.

Each system sets the colour space (a fixed hardware palette, or a colour depth that
every colour is snapped to), the colours a sprite may use (transparency is free),
typical sprite sizes per subject, and the outline style of that era. This is the
look, not the full hardware rules: per-8x8-tile palettes and attribute clash are not
enforced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import rgb_to_lab
from .palettes import BUILTIN, auto_palette, hex_to_rgb

NES_HEX = (
    "7c7c7c 0000fc 0000bc 4428bc 940084 a80020 a81000 881400 503000 007800 006800 005800 004058 000000 "
    "bcbcbc 0078f8 0058f8 6844fc d800cc e40058 f83800 e45c10 ac7c00 00b800 00a800 00a844 008888 "
    "f8f8f8 3cbcfc 6888fc 9878f8 f878f8 f85898 f87858 fca044 f8b800 b8f818 58d854 58f898 00e8d8 787878 "
    "fcfcfc a4e4fc b8b8f8 d8b8f8 f8b8f8 f8a4c0 f0d0b0 fce0a8 f8d878 d8f878 b8f8b8 b8f8d8 00fcfc f8d8f8"
).split()

RGB555 = [(i << 3) | (i >> 2) for i in range(32)]           # SNES / GBC / GBA
RGB333_MD = [0, 52, 87, 116, 144, 172, 206, 255]           # Genesis / Mega Drive DAC levels
RGB222 = [0, 85, 170, 255]                                 # Master System


@dataclass(frozen=True)
class System:
    name: str
    sprite_colors: int                  # colours per sprite (transparent not counted)
    bg_colors: int                      # colours for a background scene
    sizes: dict                         # subject -> (width, height)
    outline_style: str
    fixed: tuple[str, ...] | None = None  # hardware palette (hex) ...
    levels: tuple[int, ...] | None = None  # ... or per-channel levels of its colour depth

    def max_colors(self, subject: str) -> int:
        return self.bg_colors if subject == "background" else self.sprite_colors


SYSTEMS: dict[str, System] = {
    "nes": System("NES", 3, 13, {"person": (16, 24), "animal": (16, 16), "object": (16, 16),
                                 "background": (256, 240)}, "solid", fixed=tuple(NES_HEX)),
    "snes": System("SNES", 15, 64, {"person": (24, 32), "animal": (32, 32), "object": (16, 16),
                                    "background": (256, 224)}, "selout", levels=tuple(RGB555)),
    "gameboy": System("Game Boy", 3, 4, {"person": (16, 16), "animal": (16, 16), "object": (16, 16),
                                         "background": (160, 144)}, "solid",
                      fixed=tuple(BUILTIN["gameboy"])),
    "gbc": System("Game Boy Color", 3, 32, {"person": (16, 16), "animal": (16, 16), "object": (16, 16),
                                            "background": (160, 144)}, "solid", levels=tuple(RGB555)),
    "sms": System("Master System", 15, 31, {"person": (16, 24), "animal": (16, 16), "object": (16, 16),
                                            "background": (256, 192)}, "solid", levels=tuple(RGB222)),
    "genesis": System("Genesis / Mega Drive", 15, 61, {"person": (32, 48), "animal": (32, 32),
                                                        "object": (24, 24), "background": (320, 224)},
                      "selout", levels=tuple(RGB333_MD)),
    "gba": System("Game Boy Advance", 15, 64, {"person": (32, 32), "animal": (32, 32), "object": (16, 16),
                                               "background": (240, 160)}, "selout", levels=tuple(RGB555)),
    "pico8": System("PICO-8", 15, 16, {"person": (16, 16), "animal": (16, 16), "object": (8, 8),
                                       "background": (128, 128)}, "solid", fixed=tuple(BUILTIN["pico-8"])),
}
SYSTEM_NAMES = ("off", *SYSTEMS)


def system_overrides(system: str, subject: str) -> dict:
    """Settings a system implies for a subject (applied over the subject preset)."""
    if system in (None, "off"):
        return {}
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system '{system}' (use {', '.join(SYSTEM_NAMES)})")
    sysm = SYSTEMS[system]
    w, h = sysm.sizes[subject]
    out = dict(width=w, height=h, colors=sysm.max_colors(subject), palette="auto",
               outline_style=sysm.outline_style, outline_color="auto")
    if subject == "background":
        out["patch"] = 4  # big canvases: keep it quick
    return out


def _snap_levels(pal: np.ndarray, levels: tuple[int, ...]) -> np.ndarray:
    lv = np.array(levels, np.int32)
    idx = np.abs(pal.astype(np.int32)[..., None] - lv).argmin(-1)
    return lv[idx].astype(np.uint8)


def _error(lab_px: np.ndarray, lab_pal: np.ndarray) -> float:
    return float(((lab_px[:, None] - lab_pal[None]) ** 2).sum(-1).min(1).sum())


def system_palette(system: str, pixels: np.ndarray, colors: int, subject: str = "object") -> np.ndarray:
    """Best `colors` colours for these pixels that the system can display."""
    sysm = SYSTEMS[system]
    k = max(1, min(colors, sysm.max_colors(subject)))
    pixels = pixels.reshape(-1, 3)
    centers = auto_palette(pixels, k)
    if sysm.levels:
        pal = np.unique(_snap_levels(centers, sysm.levels), axis=0)
    else:
        hw = hex_to_rgb(list(dict.fromkeys(sysm.fixed)))  # dedupe, keep order
        hw_lab = rgb_to_lab(hw)
        nearest = ((rgb_to_lab(centers)[:, None] - hw_lab[None]) ** 2).sum(-1).argmin(1)
        chosen = list(dict.fromkeys(nearest.tolist()))
        # Two centres can snap to the same hardware colour: greedily add the colour that
        # lowers the error most until we have k (or run out of distinct pixel colours).
        sample = pixels[:: max(1, len(pixels) // 2000)]
        lab_px = rgb_to_lab(sample)
        distinct = len(np.unique(sample, axis=0))
        while len(chosen) < min(k, len(hw), distinct):
            best = min((c for c in range(len(hw)) if c not in chosen),
                       key=lambda c: _error(lab_px, hw_lab[chosen + [c]]))
            chosen.append(best)
        pal = hw[chosen]
    return pal[np.argsort(rgb_to_lab(pal)[:, 0], kind="stable")]
