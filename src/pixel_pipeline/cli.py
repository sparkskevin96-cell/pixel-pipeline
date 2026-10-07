"""Command line: pixel-pipeline convert | video | idle | sheet | tileset | palettes"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .animate import IDLE_STYLES, idle_frames
from .aseprite import write_aseprite
from .cutout import CutoutUnavailable
from .export import load_rgba, load_sprite, write_animation, write_tileset
from .godot import write_tileset as write_godot_tileset
from .palettes import BUILTIN
from .pipeline import convert_batch
from .pixelize import DOWNSCALE_MODES
from .quantize import DITHER_MODES
from .settings import ANCHORS, FITS, OUTLINE_STYLES, SUBJECTS, settings_for
from .video import ALIGN_MODES, video_to_sprites

BOOL = argparse.BooleanOptionalAction

SETTING_ARGS = ("width", "height", "fit", "anchor", "remove_background", "cutout_model",
                "crop_to_subject", "patch", "detail", "downscale", "palette", "colors", "dither",
                "dither_strength", "cleanup", "outline", "outline_style", "outline_color",
                "brightness", "contrast", "saturation", "tileable")


def _add_setting_args(p, default_subject="object"):
    p.add_argument("-s", "--subject", choices=SUBJECTS, default=default_subject)
    p.add_argument("-W", "--width", type=int)
    p.add_argument("-H", "--height", type=int, help="0 = keep the photo's aspect ratio")
    p.add_argument("--fit", choices=FITS)
    p.add_argument("--anchor", choices=ANCHORS)
    p.add_argument("--bg-removal", dest="remove_background", action=BOOL,
                   help="cut the subject out of its background")
    p.add_argument("--model", dest="cutout_model",
                   help="background-removal model, e.g. u2net_human_seg, birefnet-general")
    p.add_argument("--crop", dest="crop_to_subject", action=BOOL, help="crop to the subject")
    p.add_argument("--patch", type=int, help="photo pixels per sprite pixel (default 8)")
    p.add_argument("--detail", type=int, help="edge thickening 0-4")
    p.add_argument("--downscale", choices=DOWNSCALE_MODES)
    p.add_argument("--palette", help=f"auto | {' | '.join(BUILTIN)} | lospec:<slug> | file")
    p.add_argument("--colors", type=int, help="palette size for --palette auto")
    p.add_argument("--dither", choices=DITHER_MODES)
    p.add_argument("--dither-strength", type=float)
    p.add_argument("--cleanup", action=BOOL, help="remove stray single pixels")
    p.add_argument("--outline", action=BOOL, help="1px silhouette outline")
    p.add_argument("--outline-style", choices=OUTLINE_STYLES,
                   help="solid colour, or selout (darker shade of what it touches)")
    p.add_argument("--outline-color", help="solid outline: auto | #rrggbb")
    p.add_argument("--brightness", type=float)
    p.add_argument("--contrast", type=float)
    p.add_argument("--saturation", type=float)
    p.add_argument("--tileable", action=BOOL, help="make the result wrap seamlessly")


def _settings(a):
    return settings_for(a.subject, **{k: getattr(a, k) for k in SETTING_ARGS})


def _add_animation_out(p, default_tag=None, duration_help="ms per frame (default 100)"):
    p.add_argument("-o", "--out", type=Path, required=True, help="sprite sheet .png")
    p.add_argument("--tag", default=default_tag, help="animation name, e.g. walk")
    p.add_argument("--duration", type=int, help=duration_help)
    p.add_argument("--columns", type=int, help="sheet columns (default: one row)")
    p.add_argument("--aseprite", action="store_true", help="also write an animated .aseprite")
    p.add_argument("--godot", action="store_true",
                   help="also write a Godot SpriteFrames .tres (for AnimatedSprite2D)")


def _print_written(label: str, written: list[Path]) -> None:
    print(f"{label} -> {', '.join(str(p) for p in written)}")


# ---- convert -------------------------------------------------------------------------------

def _add_convert(sub):
    p = sub.add_parser("convert", help="photo(s) -> sprite(s)",
                       description="Turn photos into sprites. Unset options use the --subject preset.")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path,
                   help="output .png (one input) or folder (several); default: next to input")
    _add_setting_args(p)
    p.add_argument("--scale", type=int, default=1, help="also upscale the PNG (nearest) by N")
    p.add_argument("--aseprite", action="store_true", help="also write .aseprite next to each PNG")
    p.add_argument("--separate-palettes", action="store_true",
                   help="give each input its own palette (default: shared across a batch)")
    p.set_defaults(func=_cmd_convert)


def _cmd_convert(a) -> int:
    settings = _settings(a)
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


# ---- video ---------------------------------------------------------------------------------

def _add_video(sub):
    p = sub.add_parser("video", help="video clip -> animated sprite sheet",
                       description="Sample frames from a video (e.g. your dog walking), cut out the "
                                   "subject, and make an animation with one shared palette.")
    p.add_argument("video", type=Path)
    _add_animation_out(p, duration_help="ms per frame (default: real time between sampled frames)")
    p.add_argument("--frames", type=int, default=8, help="how many frames (default 8)")
    p.add_argument("--start", type=float, help="start time, seconds")
    p.add_argument("--end", type=float, help="end time, seconds (a loop should end where it starts)")
    p.add_argument("--fps", type=float, help="sample at this rate instead of spreading --frames evenly")
    p.add_argument("--align", choices=ALIGN_MODES, default="track",
                   help="track = animate in place (default), fixed = keep movement across the shot")
    _add_setting_args(p, default_subject="animal")
    p.set_defaults(func=_cmd_video)


def _cmd_video(a) -> int:
    if not a.video.exists():
        raise SystemExit(f"error: {a.video} not found")
    sprites, ms = video_to_sprites(a.video, _settings(a), a.frames, a.start, a.end, a.fps, a.align)
    frames = [s.rgba for s in sprites]
    written = write_animation(frames, a.out, a.duration or ms, a.tag, a.columns, a.aseprite,
                              a.godot, palette=sprites[0].palette)
    w, h = sprites[0].size
    _print_written(f"{a.video.name}: {len(frames)} frames {w}x{h} @ {a.duration or ms} ms", written)
    return 0


# ---- idle ----------------------------------------------------------------------------------

def _add_idle(sub):
    p = sub.add_parser("idle", help="sprite -> looping idle animation (breathe / bob / sway)")
    p.add_argument("sprite", type=Path, help="a pixel-art sprite PNG (e.g. from convert)")
    _add_animation_out(p, default_tag="idle", duration_help="ms per frame (default 150)")
    p.add_argument("--style", choices=IDLE_STYLES, default="breathe",
                   help="breathe (characters), bob (floating items), sway (plants, hair)")
    p.add_argument("--frames", type=int, help="frames in the loop (default 4; sway 8)")
    p.add_argument("--amount", type=int, default=1, help="movement in pixels (default 1)")
    p.set_defaults(func=_cmd_idle)


def _cmd_idle(a) -> int:
    rgba = load_sprite(a.sprite)
    frames = idle_frames(rgba, a.style, a.frames, a.amount)
    written = write_animation(frames, a.out, a.duration or 150, a.tag, a.columns, a.aseprite, a.godot)
    _print_written(f"{a.sprite.name}: {len(frames)}-frame {a.style}", written)
    return 0


# ---- sheet / tileset / palettes ------------------------------------------------------------

def _add_sheet(sub):
    p = sub.add_parser("sheet", help="sprites -> sprite sheet + JSON (+ .aseprite / Godot)")
    p.add_argument("frames", nargs="+", type=Path, help="frame PNGs in order")
    _add_animation_out(p)
    p.set_defaults(func=_cmd_sheet)


def _cmd_sheet(a) -> int:
    frames = [load_rgba(f) for f in a.frames]
    written = write_animation(frames, a.out, a.duration or 100, a.tag, a.columns, a.aseprite,
                              a.godot, names=[f.stem for f in a.frames])
    _print_written(f"{len(frames)} frames", written)
    return 0


def _add_tileset(sub):
    p = sub.add_parser("tileset", help="pixel-art background -> tileset PNG + Tiled map (+ Godot)")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="tileset .png")
    p.add_argument("--tile", type=int, default=16, help="tile size in pixels")
    p.add_argument("--columns", type=int, default=8)
    p.add_argument("--godot", action="store_true", help="also write a Godot TileSet .tres")
    p.set_defaults(func=_cmd_tileset)


def _cmd_tileset(a) -> int:
    tmj = write_tileset(load_rgba(a.image), a.tile, a.out, a.columns)
    written = [a.out, tmj]
    if a.godot:
        import json
        ts = json.loads(tmj.read_text())["tilesets"][0]
        written.append(write_godot_tileset(a.out.with_suffix(".tres"), a.out.name, a.tile,
                                           ts["columns"], ts["tilecount"]))
    _print_written(a.image.name, written)
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
    _add_video(sub)
    _add_idle(sub)
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
