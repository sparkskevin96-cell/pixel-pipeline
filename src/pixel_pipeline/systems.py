"""Retro system looks: one setting that applies a console's sprite limits.

Each system sets the colour space (a fixed hardware palette, or a colour depth that
every colour is snapped to), the colours a sprite may use (transparency is free),
typical sprite sizes per subject, and the outline style of that era. This is the
look, not the full hardware rules: per-8x8-tile palettes and attribute clash are not
enforced.

Two looks per system:
  readable  (default) the console's colours, at a size a photo stays recognizable
            (48x64 people) with the colours big multi-sprite characters had (NES games
            stacked 4 sprite palettes = 12 colours; SNES used several 15-colour ones).
  authentic one 8x8/16x16-era sprite: tiny sizes and one sprite palette (NES 16x24,
            3 colours). Faithful, but a photo usually turns into a blob.
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

# Four evenly spread DMG shades (the "pea soup" palette's two lightest greens are almost
# the same, which leaves photos with 3 usable shades).
DMG_HEX = ("081820", "346856", "88c070", "e0f8d0")
RGB555 = [(i << 3) | (i >> 2) for i in range(32)]           # SNES / GBC / GBA
RGB333_MD = [0, 52, 87, 116, 144, 172, 206, 255]           # Genesis / Mega Drive DAC levels
RGB222 = [0, 85, 170, 255]                                 # Master System


LOOKS = ("readable", "authentic")
READABLE_SIZES = {"person": (48, 64), "animal": (48, 48), "object": (32, 32)}


@dataclass(frozen=True)
class System:
    name: str
    sprite_colors: int                  # colours per sprite (transparent not counted)
    bg_colors: int                      # colours for a background scene
    sizes: dict                         # subject -> (width, height), authentic look
    outline_style: str
    fixed: tuple[str, ...] | None = None  # hardware palette (hex) ...
    levels: tuple[int, ...] | None = None  # ... or per-channel levels of its colour depth
    readable_colors: int = 0            # readable look: colours of a multi-palette character
    mono: bool = False                  # shades of one hue: photos map by brightness

    def max_colors(self, subject: str, look: str = "authentic") -> int:
        if subject == "background":
            return self.bg_colors
        return self.readable_colors if look == "readable" else self.sprite_colors

    def size(self, subject: str, look: str = "authentic") -> tuple[int, int]:
        if look == "readable" and subject in READABLE_SIZES:
            return READABLE_SIZES[subject]
        return self.sizes[subject]

    def gamut(self) -> np.ndarray | None:
        """Every colour the system can show when that is a short list, else None."""
        if self.fixed:
            return hex_to_rgb(list(dict.fromkeys(self.fixed)))  # dedupe, keep order
        if len(self.levels) ** 3 <= 64:
            lv = np.array(self.levels, np.uint8)
            return np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
        return None


SYSTEMS: dict[str, System] = {
    "nes": System("NES", 3, 13, {"person": (16, 24), "animal": (16, 16), "object": (16, 16),
                                 "background": (256, 240)}, "solid", fixed=tuple(NES_HEX),
                  readable_colors=12),
    "snes": System("SNES", 15, 64, {"person": (24, 32), "animal": (32, 32), "object": (16, 16),
                                    "background": (256, 224)}, "selout", levels=tuple(RGB555),
                   readable_colors=32),
    "gameboy": System("Game Boy", 3, 4, {"person": (16, 16), "animal": (16, 16), "object": (16, 16),
                                         "background": (160, 144)}, "solid",
                      fixed=DMG_HEX, readable_colors=4, mono=True),
    "gbc": System("Game Boy Color", 3, 32, {"person": (16, 16), "animal": (16, 16), "object": (16, 16),
                                            "background": (160, 144)}, "solid", levels=tuple(RGB555),
                  readable_colors=24),
    "sms": System("Master System", 15, 31, {"person": (16, 24), "animal": (16, 16), "object": (16, 16),
                                            "background": (256, 192)}, "solid", levels=tuple(RGB222),
                  readable_colors=15),
    "genesis": System("Genesis / Mega Drive", 15, 61, {"person": (32, 48), "animal": (32, 32),
                                                        "object": (24, 24), "background": (320, 224)},
                      "selout", levels=tuple(RGB333_MD), readable_colors=32),
    "gba": System("Game Boy Advance", 15, 64, {"person": (32, 32), "animal": (32, 32), "object": (16, 16),
                                               "background": (240, 160)}, "selout", levels=tuple(RGB555),
                  readable_colors=32),
    "pico8": System("PICO-8", 15, 16, {"person": (16, 16), "animal": (16, 16), "object": (8, 8),
                                       "background": (128, 128)}, "solid", fixed=tuple(BUILTIN["pico-8"]),
                    readable_colors=16),
}
SYSTEM_NAMES = ("off", *SYSTEMS)


def system_overrides(system: str, subject: str, look: str = "readable") -> dict:
    """Settings a system implies for a subject (applied over the subject preset)."""
    if system in (None, "off"):
        return {}
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system '{system}' (use {', '.join(SYSTEM_NAMES)})")
    if look not in LOOKS:
        raise ValueError(f"Unknown look '{look}' (use {', '.join(LOOKS)})")
    sysm = SYSTEMS[system]
    w, h = sysm.size(subject, look)
    out = dict(width=w, height=h, colors=sysm.max_colors(subject, look), palette="auto",
               outline_style=sysm.outline_style, outline_color="auto")
    if subject == "background":
        out["patch"] = 4  # big canvases: keep it quick
    return out


def tone_map(system: str, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shades-of-one-hue systems (Game Boy): recolour by brightness so light and dark
    parts land on different shades (like the Game Boy Camera) instead of every colour
    snapping to its nearest green. Brightness is half stretched to the subject's own
    range, half equalized so each shade gets a fair share of the sprite."""
    sysm = SYSTEMS[system]
    if not sysm.mono:
        return rgb
    ramp = sysm.gamut()
    ramp = ramp[np.argsort(rgb_to_lab(ramp)[:, 0], kind="stable")].astype(np.float32)
    lum = rgb_to_lab(rgb)[..., 0]
    vals = np.sort(lum[mask] if mask.any() else lum.ravel())
    lo, hi = np.percentile(vals, (2, 98))
    stretched = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    equalized = np.searchsorted(vals, lum, side="right") / len(vals)
    t = (stretched + equalized) / 2 * (len(ramp) - 1)
    i = np.minimum(t.astype(np.int32), len(ramp) - 2)
    f = (t - i)[..., None]
    return np.rint(ramp[i] * (1 - f) + ramp[i + 1] * f).astype(np.uint8)


def _snap_levels(pal: np.ndarray, levels: tuple[int, ...]) -> np.ndarray:
    lv = np.array(levels, np.int32)
    idx = np.abs(pal.astype(np.int32)[..., None] - lv).argmin(-1)
    return lv[idx].astype(np.uint8)


def _error(lab_px: np.ndarray, lab_pal: np.ndarray) -> float:
    return float(((lab_px[:, None] - lab_pal[None]) ** 2).sum(-1).min(1).sum())


def system_palette(system: str, pixels: np.ndarray, colors: int, subject: str = "object",
                   look: str = "authentic") -> np.ndarray:
    """Best `colors` colours for these pixels that the system can display."""
    sysm = SYSTEMS[system]
    k = max(1, min(colors, sysm.max_colors(subject, look)))
    pixels = pixels.reshape(-1, 3)
    centers = auto_palette(pixels, k)
    hw = sysm.gamut()
    if hw is None:  # deep colour (RGB555 etc.): snap each channel
        pal = np.unique(_snap_levels(centers, sysm.levels), axis=0)
    else:  # short list (NES, Game Boy, Master System's 64): nearest by eye, in Lab
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
