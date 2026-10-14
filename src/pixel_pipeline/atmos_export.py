"""Library exports for weather, water and background assets (exports.KIND_EXPORTS hook).

    weather     every preset: the sheet as usual (sliced sprites / SpriteFrames / strips) PLUS
                particles/*.png and weather.json; Godot presets also the overlay / particle
                scenes, shaders and weather_fx.gd (names kept: the scenes point at them);
                Unity: .meta for every particle texture + weather.json (TextAsset)
    water       the open-water sheet as usual PLUS the strips, the animated shore TileSet
                (Godot .tres / Tiled .tsx) and water.json (+ .meta files for Unity)
    background  instead of the usual: the layer PNGs; Godot = Parallax2D scene; Unity = layers
                with .meta, parallax.json, PixelParallax.cs
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

GODOT_FILES = ("*.tscn", "*.gd", "*.gdshader", "sheet.png", "sheet_glow.png", "sheet.tres", "sheet_glow.tres",
               "sheet_add.tres")


def _copy(src: Path, dst: Path, out: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    out.append(dst)


def _unity_png_meta(png: Path, key: str, ppu: float, out: list) -> None:
    from .exports import unity_guid, unity_meta
    with Image.open(png) as im:
        size = im.size
    meta = png.with_name(png.name + ".meta")
    meta.write_text(unity_meta(unity_guid(key), size, None, ppu=ppu, pivot=(0.5, 0.5)), encoding="utf-8")
    out.append(meta)


def _unity_text_meta(path: Path, key: str, out: list) -> None:
    from .exports import unity_guid
    meta = path.with_name(path.name + ".meta")
    meta.write_text(f"fileFormatVersion: 2\nguid: {unity_guid(key)}\nTextScriptImporter:\n  externalObjects: {{}}\n"
                    "  userData: \n  assetBundleName: \n  assetBundleVariant: \n", encoding="utf-8")
    out.append(meta)


def export_kind(src, preset: str, out: Path, default, ppu: float | None = None, **_):
    folder = Path(src.folder)
    written: list[Path] = []
    unity = preset == "unity"
    upp = float(ppu or 32)
    if src.kind == "background":
        if unity:
            for p in sorted((folder / "unity").glob("*")):
                _copy(p, out / p.name, written)
            return written
        for p in sorted(folder.glob("*.png")) + [folder / "parallax.json"]:
            if p.exists() and p.name != "thumb.png":
                _copy(p, out / p.name, written)
        if preset.startswith("godot"):
            for p in folder.glob("*.tscn"):
                _copy(p, out / p.name, written)
        return written
    written += list(default())
    base = src.file_name
    if src.kind == "weather":
        for p in sorted((folder / "particles").glob("*.png")):
            dst = out / "particles" / p.name
            _copy(p, dst, written)
            if unity:
                _unity_png_meta(dst, f"pixel-pipeline/weather/{base}/{p.name}", upp, written)
        js = out / f"{base}_weather.json"
        _copy(folder / "weather.json", js, written)
        if unity:
            _unity_text_meta(js, f"pixel-pipeline/weather/{base}/weather.json", written)
        if preset.startswith("godot"):
            for pat in GODOT_FILES:
                for p in folder.glob(pat):
                    if not (out / p.name).exists() or p.suffix in (".tscn", ".gd", ".gdshader"):
                        _copy(p, out / p.name, written)
    elif src.kind == "water":
        names = ["water.png", "water_glow.png", "side_surface.png", "side_body.png", "iso_water.png",
                 "shore_tileset.png", "water.json"]
        if not unity:
            names += ["shore_tileset.tres", "shore_tileset.tsx"]
        for n in names:
            p = folder / n
            if not p.exists():
                continue
            dst = out / f"{base}_{n}" if unity else out / n
            _copy(p, dst, written)
            if unity:
                if dst.suffix == ".png":
                    _unity_png_meta(dst, f"pixel-pipeline/water/{base}/{n}", upp, written)
                else:
                    _unity_text_meta(dst, f"pixel-pipeline/water/{base}/{n}", written)
    return written
