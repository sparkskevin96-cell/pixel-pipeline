"""Palettes: built-in classics, palette files, Lospec downloads, and auto (k-means)."""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .color import lab_to_rgb, luminance, rgb_to_lab

BUILTIN: dict[str, list[str]] = {
    "pico-8": [
        "000000", "1d2b53", "7e2553", "008751", "ab5236", "5f574f", "c2c3c7", "fff1e8",
        "ff004d", "ffa300", "ffec27", "00e436", "29adff", "83769c", "ff77a8", "ffccaa",
    ],
    "sweetie-16": [
        "1a1c2c", "5d275d", "b13e53", "ef7d57", "ffcd75", "a7f070", "38b764", "257179",
        "29366f", "3b5dc9", "41a6f6", "73eff7", "f4f4f4", "94b0c2", "566c86", "333c57",
    ],
    "endesga-32": [
        "be4a2f", "d77643", "ead4aa", "e4a672", "b86f50", "733e39", "3e2731", "a22633",
        "e43b44", "f77622", "feae34", "fee761", "63c74d", "3e8948", "265c42", "193c3e",
        "124e89", "0099db", "2ce8f5", "ffffff", "c0cbdc", "8b9bb4", "5a6988", "3a4466",
        "262b44", "181425", "ff0044", "68386c", "b55088", "f6757a", "e8b796", "c28569",
    ],
    "gameboy": ["0f380f", "306230", "8bac0f", "9bbc0f"],
    "1bit": ["000000", "ffffff"],
}

CACHE_DIR = Path.home() / ".cache" / "pixel-pipeline" / "palettes"
_HEX = re.compile(r"#?\b([0-9a-fA-F]{6})\b")


def hex_to_rgb(hexes: list[str]) -> np.ndarray:
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes], np.uint8)


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb[:3])


def load_palette(spec: str) -> np.ndarray | None:
    """Resolve a palette spec. Returns None for "auto" (build from the image).

    spec: "auto" | built-in name | "lospec:<slug>" | path to .gpl/.hex/.txt/.png/.gif
    """
    if spec == "auto":
        return None
    if spec in BUILTIN:
        return hex_to_rgb(BUILTIN[spec])
    if spec.startswith("lospec:"):
        return _lospec(spec.split(":", 1)[1])
    path = Path(spec)
    if not path.exists():
        raise ValueError(
            f"Unknown palette '{spec}'. Use auto, a built-in ({', '.join(BUILTIN)}), "
            "lospec:<slug>, or a .gpl/.hex/.png file.")
    suffix = path.suffix.lower()
    if suffix == ".gpl":
        pal = _parse_gpl(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix in (".hex", ".txt"):
        pal = _parse_hex(path.read_text(encoding="utf-8", errors="replace"))
    else:
        pal = _from_image(path)
    if len(pal) == 0:
        raise ValueError(f"No colours found in palette file {path}")
    return pal


def _parse_gpl(text: str) -> np.ndarray:
    colors = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "GIMP Palette", "Name:", "Columns:")):
            continue
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            colors.append([int(p) for p in parts[:3]])
    return np.array(colors, np.uint8).reshape(-1, 3)


def _parse_hex(text: str) -> np.ndarray:
    return hex_to_rgb(_HEX.findall(text)).reshape(-1, 3)


def _from_image(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA")).reshape(-1, 4)
    rgb = rgba[rgba[:, 3] >= 128][:, :3]
    _, first = np.unique(rgb, axis=0, return_index=True)
    return rgb[np.sort(first)][:256]


def _lospec(slug: str) -> np.ndarray:
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError(f"Bad Lospec palette slug '{slug}' (expected e.g. lospec:resurrect-64)")
    cached = CACHE_DIR / f"{slug}.hex"
    if not cached.exists():
        url = f"https://lospec.com/palette-list/{slug}.hex"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - surface any network error with context
            raise ValueError(f"Couldn't download Lospec palette '{slug}' from {url}: {e}") from e
        if not _HEX.search(text):
            raise ValueError(f"Lospec palette '{slug}' not found")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
    return _parse_hex(cached.read_text(encoding="utf-8"))


def auto_palette(pixels: np.ndarray, colors: int, seed: int = 0) -> np.ndarray:
    """Pick `colors` representative colours (k-means in Lab), sorted dark -> light."""
    pixels = pixels.reshape(-1, 3)
    unique = np.unique(pixels, axis=0)
    if len(unique) <= colors:
        pal = unique
    else:
        lab = rgb_to_lab(pixels).astype(np.float32)
        cv2.setRNGSeed(seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, _, centers = cv2.kmeans(lab, colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        pal = np.unique(lab_to_rgb(centers), axis=0)
    return pal[np.argsort(luminance(pal), kind="stable")]
