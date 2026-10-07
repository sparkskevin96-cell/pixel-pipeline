"""Command line: pixel-pipeline convert | sheet | tileset | palettes"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .aseprite import write_aseprite
from .cutout import CutoutUnavailable
from .export import load_rgba, write_sheet, write_tileset
from .palettes import BUILTIN
from .pipeline import convert_batch
from .pixelize import DOWNSCALE_MODES
from .quantize import DITHER_MODES
from .settings import ANCHORS, FITS, SUBJECTS, settings_for

BOOL = argparse.BooleanOptionalAction


def _add_convert(sub):
    p = sub.add_parser("convert", help="photo(s) -> sprite(s)",
                       description="Turn photos into sprites. Unset options use the --subject preset.")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path,
                   help="output .png (one input) or folder (several); default: next to input")
    p.add_argument("-s", "--subject", choices=SUBJECTS, default="object")
    p.add_argument("-W", "--width", type=int)
    p.add_argument("-H", "--height", type=int, help="0 = keep the photo's aspect ratio")
    p.add_argument("--fit", choices=FITS)
    p.add_argument("--anchor", choices=ANCHORS)
    p.add_argument("--bg-removal", dest="remove_background", action=BOOL,
                   help="cut the subject out of its background")
    p.add_argument("--model", dest="cutout_model", help="rembg model, e.g. u2net_human_seg")
    p.add_argument("--crop", dest="crop_to_subject", action=BOOL, help="crop to the subject")
    p.add_argument("--patch", type=int, help="photo pixels per sprite pixel (default 8)")
    p.add_argument("--detail", type=int, help="edge thickening 0-4")
    p.add_argument("--downscale", choices=DOWNSCALE_MODES)
    p.add_argument("--palette", help=f"auto | {' | '.join(BUILTIN)} | lospec:<slug> | file")
    p.add_argument("--colors", type=int, help="palette size for --palette auto")
    p.add_argument("--dither", choices=DITHER_MODES)
    p.add_argument("--dither-strength", type=float)
    p.add_argument("--outline", action=BOOL, help="1px silhouette outline")
    p.add_argument("--outline-color", help="auto | #rrggbb")
    p.add_argument("--brightness", type=float)
    p.add_argument("--contrast", type=float)
    p.add_argument("--saturation", type=float)
    p.add_argument("--tileable", action=BOOL, help="make the result wrap seamlessly")
    p.add_argument("--scale", type=int, default=1, help="also upscale the PNG (nearest) by N")
    p.add_argument("--aseprite", action="store_true", help="also write .aseprite next to each PNG")
    p.add_argument("--separate-palettes", action="store_true",
                   help="give each input its own palette (default: shared across a batch)")
    p.set_defaults(func=_cmd_convert)


SETTING_ARGS = ("width", "height", "fit", "anchor", "remove_background", "cutout_model",
                "crop_to_subject", "patch", "detail", "downscale", "palette", "colors", "dither",
                "dither_strength", "outline", "outline_color", "brightness", "contrast",
                "saturation", "tileable")


def _cmd_convert(a) -> int:
    settings = settings_for(a.subject, **{k: getattr(a, k) for k in SETTING_ARGS})
    for path in a.inputs:
        if not path.exists():
            raise SystemExit(f"error: {path} not found")
    single_file = a.out is not None and len(a.inputs) == 1 and a.out.suffix.lower() == ".png"
    if a.out is not None and not single_file:
        a.out.mkdir(parents=True, exist_ok=True)
    sprites = convert_batch(a.inputs, settings, shared_palette=not a.separate_palettes)
    for path, sprite in zip(a.inputs, sprites):
        if single_file:
            out = a.out
        else:
            out = (a.out or path.parent) / f"{path.stem}.sprite.png"
        sprite.save(out, a.scale)
        written = [out]
        if a.aseprite:
            ase = out.with_suffix(".aseprite")
            write_aseprite(ase, [sprite.rgba], sprite.palette)
            written.append(ase)
        w, h = sprite.size
        print(f"{path.name} -> {', '.join(str(p) for p in written)} "
              f"({w}x{h}, {len(sprite.palette)} colours)")
    return 0


def _add_sheet(sub):
    p = sub.add_parser("sheet", help="sprites -> sprite sheet + JSON (and .aseprite)")
    p.add_argument("frames", nargs="+", type=Path, help="frame PNGs in order")
    p.add_argument("-o", "--out", type=Path, required=True, help="sheet .png")
    p.add_argument("--columns", type=int)
    p.add_argument("--duration", type=int, default=100, help="ms per frame")
    p.add_argument("--tag", help="animation name, e.g. walk")
    p.add_argument("--aseprite", action="store_true", help="also write an animated .aseprite")
    p.set_defaults(func=_cmd_sheet)


def _cmd_sheet(a) -> int:
    frames = [load_rgba(f) for f in a.frames]
    out_json = write_sheet(frames, a.out, a.columns, a.duration, a.tag, [f.stem for f in a.frames])
    written = [a.out, out_json]
    if a.aseprite:
        if len({f.shape for f in frames}) != 1:
            raise SystemExit("error: .aseprite animation needs all frames the same size")
        ase = a.out.with_suffix(".aseprite")
        tags = [(a.tag, 0, len(frames) - 1)] if a.tag else []
        write_aseprite(ase, frames, durations=a.duration, tags=tags)
        written.append(ase)
    print(f"{len(frames)} frames -> {', '.join(str(p) for p in written)}")
    return 0


def _add_tileset(sub):
    p = sub.add_parser("tileset", help="pixel-art background -> tileset PNG + Tiled map (.tmj)")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="tileset .png")
    p.add_argument("--tile", type=int, default=16, help="tile size in pixels")
    p.add_argument("--columns", type=int, default=8)
    p.set_defaults(func=_cmd_tileset)


def _cmd_tileset(a) -> int:
    tmj = write_tileset(load_rgba(a.image), a.tile, a.out, a.columns)
    print(f"{a.image.name} -> {a.out}, {tmj}")
    return 0


def _cmd_palettes(_a) -> int:
    for name, colors in BUILTIN.items():
        print(f"{name:12} {len(colors):3} colours")
    print("lospec:<slug>  any palette from https://lospec.com/palette-list")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pixel-pipeline", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_convert(sub)
    _add_sheet(sub)
    _add_tileset(sub)
    sub.add_parser("palettes", help="list built-in palettes").set_defaults(func=_cmd_palettes)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CutoutUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
