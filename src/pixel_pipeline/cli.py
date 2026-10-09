"""Command line: pixel-pipeline convert | generate | video | idle | parallax | sheet | tileset | palettes"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .animate import IDLE_MS, IDLE_STYLES, idle_frames
from .aseprite import write_aseprite
from .cutout import CutoutUnavailable
from .export import load_rgba, load_sprite, write_animation, write_tileset
from .generate import ComfyClient, ComfyUIError
from .godot import write_tileset as write_godot_tileset
from .palettes import BUILTIN
from .pipeline import convert_batch
from .pixelize import DOWNSCALE_MODES
from .quantize import DITHER_MODES
from .settings import ANCHORS, FITS, OUTLINE_STYLES, SUBJECTS, settings_for
from .systems import LOOKS, SYSTEM_NAMES
from .turnaround import DIRECTION_COUNTS, ModelUnavailable
from .video import ALIGN_MODES, video_to_sprites

BOOL = argparse.BooleanOptionalAction

SETTING_ARGS = ("system", "look", "width", "height", "fit", "anchor", "remove_background", "cutout_model",
                "crop_to_subject", "patch", "detail", "downscale", "palette", "colors", "dither",
                "dither_strength", "cleanup", "outline", "outline_style", "outline_color",
                "brightness", "contrast", "saturation", "tileable")


def _add_setting_args(p, default_subject="object"):
    p.add_argument("-s", "--subject", choices=SUBJECTS, default=default_subject)
    p.add_argument("--system", choices=SYSTEM_NAMES,
                   help="retro console look: colours per sprite, palette and sizes of that system")
    p.add_argument("--look", choices=LOOKS,
                   help="with --system: readable (default; console colours, recognizable size) or "
                        "authentic (one tiny sprite of that era, e.g. NES 16x24 in 3 colours)")
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


# ---- generate (local GPU via ComfyUI) --------------------------------------------------------

def _add_generate(sub):
    p = sub.add_parser("generate", help="text prompt -> sprite, made on your own GPU (ComfyUI)",
                       description="Generate pixel art from a prompt with ComfyUI running on this PC "
                                   "(see docs/LOCAL_AI.md), then snap it to your size, palette and "
                                   "retro system like any photo.")
    p.add_argument("prompt", nargs="?", help='what to make, e.g. "a knight with a red cape"')
    p.add_argument("-o", "--out", type=Path, help="output .png (default: named after the prompt)")
    p.add_argument("--check", action="store_true", help="check ComfyUI and the models, then exit")
    p.add_argument("--count", type=int, default=1, help="how many variations (different seeds)")
    p.add_argument("--seed", type=int, help="re-make one you liked")
    p.add_argument("--fast", action="store_true", help="LCM-LoRA: ~3x faster, a little rougher")
    p.add_argument("--init", type=Path, help="start from this sketch (img2img), e.g. a sprite Claude drew")
    p.add_argument("--denoise", type=float, default=0.6, help="how much the GPU may change --init (0-1)")
    p.add_argument("--template", type=Path, help="your own ComfyUI workflow (API format) with {{placeholders}}")
    p.add_argument("--checkpoint", help="checkpoint file in ComfyUI/models/checkpoints")
    p.add_argument("--lora", help='LoRA in ComfyUI/models/loras ("none" to turn off)')
    p.add_argument("--vae", help="VAE in ComfyUI/models/vae")
    p.add_argument("--negative", help="negative prompt (default: a built-in one)")
    p.add_argument("--raw-prompt", action="store_true", help="send the prompt exactly as typed")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--comfyui", help="ComfyUI address (default $COMFYUI_URL or http://127.0.0.1:8188)")
    p.add_argument("--scale", type=int, default=1, help="also upscale the PNG (nearest) by N")
    p.add_argument("--aseprite", action="store_true", help="also write .aseprite")
    _add_setting_args(p)
    p.set_defaults(func=_cmd_generate)


def _gen_options(a) -> dict:
    opts = dict(fast=a.fast or None, checkpoint=a.checkpoint, lora=a.lora, vae=a.vae,
                negative=a.negative, steps=a.steps, cfg=a.cfg,
                template=str(a.template) if a.template else None,
                shape_prompt=False if a.raw_prompt else None)
    return {k: v for k, v in opts.items() if v is not None}


def _cmd_generate(a) -> int:
    from .generate import check_setup, generate_sprite
    client = ComfyClient(a.comfyui) if a.comfyui else None
    if a.check:
        ok, report = check_setup(client, **_gen_options(a))
        print(report)
        return 0 if ok else 1
    if not a.prompt:
        raise SystemExit('error: give a prompt, e.g. pixel-pipeline generate "a red fox" -s animal')
    if a.init and not a.init.exists():
        raise SystemExit(f"error: {a.init} not found")
    settings = _settings(a)
    results = generate_sprite(a.prompt, settings, client=client, subject=a.subject,
                              system=None if settings.system == "off" else settings.system,
                              seed=a.seed, count=a.count, init_image=a.init, denoise=a.denoise,
                              **_gen_options(a))
    slug = "-".join("".join(c if c.isalnum() else " " for c in a.prompt.lower()).split())[:40]
    out = a.out or Path(f"{slug or 'sprite'}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, (sprite, raw, seed) in enumerate(results, 1):
        path = out if len(results) == 1 else out.with_name(f"{out.stem}-{i}{out.suffix}")
        sprite.save(path, a.scale)
        raw_path = path.with_name(path.stem + ".raw.png")
        raw.save(raw_path)
        written = [path, raw_path]
        if a.aseprite:
            ase = path.with_suffix(".aseprite")
            write_aseprite(ase, [sprite.rgba], sprite.palette)
            written.append(ase)
        w, h = sprite.size
        print(f"seed {seed}: {', '.join(str(p) for p in written)} ({w}x{h}, {len(sprite.palette)} colours)")
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


# ---- turnaround ----------------------------------------------------------------------------

def _add_turnaround(sub):
    p = sub.add_parser(
        "turnaround", help="3D model, or photos/video around a subject -> 4/8/16-direction sprites",
        description="One sprite per facing direction (E, ESE, SE ... clockwise, as Godot measures "
                    "angles), all with one palette and one scale. Give a 3D model (.glb .gltf .obj "
                    ".ply .stl, e.g. from an image-to-3D AI), a video of you walking all the way "
                    "around a still subject (start in front, walk to your right), or one photo per "
                    "direction in that order.")
    p.add_argument("source", nargs="+", type=Path,
                   help="3D model, video, or photos/folder (one photo per direction, in order)")
    p.add_argument("-o", "--out", type=Path, required=True, help="sprite sheet .png")
    p.add_argument("--directions", type=int, choices=DIRECTION_COUNTS, default=16)
    p.add_argument("--elevation", type=float, default=30.0,
                   help="3D: camera tilt in degrees (0 = side view, 30 = top-down/isometric)")
    p.add_argument("--up", choices=("y", "z"), default="y", help="3D: the model's up axis (glTF = y)")
    p.add_argument("--yaw", type=float, default=0.0,
                   help="3D: degrees to turn the model if its front doesn't face +z")
    p.add_argument("--light", type=float, default=0.35, help="3D: shading 0 (flat) to 1")
    p.add_argument("--reverse", action="store_true", help="photos/video: you walked to your left")
    p.add_argument("--mirror", action="store_true",
                   help="photos/video: only the front-to-back half, mirror the other side")
    p.add_argument("--start", type=float, help="video: start time, seconds")
    p.add_argument("--end", type=float, help="video: end time, seconds")
    p.add_argument("--animate", action="store_true",
                   help="3D: play the model's animations (rigged .glb/.gltf): a row per animation "
                        "and direction, Godot animations like walk_SE. More .glb files can add "
                        "animations (e.g. one Mixamo download each)")
    p.add_argument("--animations", type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
                   help="--animate: which ones, comma-separated, in order; rename with new=old "
                        "(default: all)")
    p.add_argument("--frames", type=int, default=8, help="--animate: frames per animation (default 8)")
    p.add_argument("--no-in-place", dest="in_place", action="store_false",
                   help="--animate: keep root motion (default: walks stay on the spot)")
    p.add_argument("--colours-from", type=Path, metavar="MODEL",
                   help="--animate: take colours from this model (the original, when rigging "
                        "on Mixamo dropped them)")
    p.add_argument("--columns", type=int, help="sheet columns (default 8)")
    p.add_argument("--aseprite", action="store_true", help="also write .aseprite (a tag per direction)")
    p.add_argument("--godot", action="store_true",
                   help="also write Godot SpriteFrames (an animation per direction) + facing script")
    p.add_argument("--preview", type=int, metavar="SCALE",
                   help="also write a spinning preview GIF enlarged by SCALE")
    _add_setting_args(p, default_subject="person")
    p.set_defaults(func=_cmd_turnaround)


def _cmd_turnaround(a) -> int:
    from .turnaround import turnaround_from, write_preview, write_turnaround
    for path in a.source:
        if not path.exists():
            raise SystemExit(f"error: {path} not found")
    if a.animate or a.animations:
        return _animated_turnaround(a)
    sprites, what = turnaround_from(a.source, _settings(a), a.directions, elevation=a.elevation,
                                    up=a.up, yaw=a.yaw, light=a.light, reverse=a.reverse,
                                    mirror=a.mirror, start=a.start, end=a.end)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_turnaround(sprites, a.out, a.aseprite, a.godot, a.columns)
    if a.preview:
        written.append(write_preview(sprites, a.out.with_name(a.out.stem + "_spin.gif"), a.preview))
    w, h = sprites[0].size
    _print_written(f"{what} -> {len(sprites)} directions {w}x{h}, "
                   f"{len(sprites[0].palette)} colours", written)
    return 0


def _animated_turnaround(a) -> int:
    from .anim3d import animated_turnaround, write_animated, write_animated_preview
    anim, what = animated_turnaround(a.source, _settings(a), a.directions, a.frames, a.animations,
                                     elevation=a.elevation, yaw=a.yaw, light=a.light,
                                     in_place=a.in_place, colours_from=a.colours_from)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_animated(anim, a.out, a.aseprite, a.godot)
    if a.preview:
        written.append(write_animated_preview(anim, a.out.with_name(a.out.stem + "_anim.gif"),
                                              a.preview))
    first = anim.sprites[0][0][0]
    w, h = first.size
    _print_written(f"{what} -> {len(anim.actions)} animations x {len(anim.directions)} directions x "
                   f"{anim.frames} frames, {w}x{h}, {len(first.palette)} colours", written)
    return 0


# ---- ai (online providers) -----------------------------------------------------------------

def _add_ai(sub):
    from .online import PIXELLAB_DIRECTIONS, PIXELLAB_MODELS, PIXELLAB_VIEWS, PROVIDERS
    p = sub.add_parser(
        "ai", help="prompt -> sprite from PixelLab, Retro Diffusion and your GPU, side by side",
        description="Makes the same prompt with each provider and writes <name>_<provider>.png (on "
                    "your settings' palette), the raw <name>_<provider>.raw.png and "
                    "<name>_compare.png. Only the prompt is sent. Keys: PIXELLAB_API_KEY / "
                    "RETRO_DIFFUSION_API_KEY, or --save-key, or the desktop app's Keys button.")
    p.add_argument("prompt", nargs="?", help="what to make, e.g. 'a knight with a red cape'")
    p.add_argument("-o", "--out", type=Path, help="base .png name")
    p.add_argument("--providers", default="pixellab,retrodiffusion,gpu",
                   help=f"comma-separated: {', '.join(PROVIDERS)}")
    p.add_argument("--pixellab-model", choices=list(PIXELLAB_MODELS), default="pixen")
    p.add_argument("--rd-style", default="rd_fast__default",
                   help="Retro Diffusion style: rd_fast__default ($0.03), rd_plus__default ($0.06), "
                        "rd_pro__default ($0.18), ...")
    p.add_argument("--view", choices=PIXELLAB_VIEWS, help="PixelLab camera")
    p.add_argument("--direction", choices=PIXELLAB_DIRECTIONS, help="PixelLab: which way it faces")
    p.add_argument("--seed", type=int)
    p.add_argument("--balance", action="store_true", help="show what's left on each account and stop")
    p.add_argument("--save-key", action="append", default=[], metavar="PROVIDER=KEY",
                   help="remember a key for this user (pixellab=..., retrodiffusion=...)")
    p.add_argument("--comfyui", help="ComfyUI address for the gpu provider")
    _add_setting_args(p, default_subject="person")
    p.set_defaults(func=_cmd_ai)


def _cmd_ai(a) -> int:
    from . import online
    if a.save_key:
        keys = online.load_keys()
        for item in a.save_key:
            prov, _, key = item.partition("=")
            if prov not in ("pixellab", "retrodiffusion") or not key:
                raise SystemExit("error: --save-key pixellab=KEY or retrodiffusion=KEY")
            keys[prov] = key
        print(f"Keys saved to {online.save_keys(keys)}")
        if not a.prompt and not a.balance:
            return 0
    providers = [x.strip() for x in a.providers.split(",") if x.strip()]
    if a.balance:
        for prov in providers:
            if prov == "gpu":
                continue
            check = online.pixellab_balance if prov == "pixellab" else online.retrodiffusion_balance
            try:
                print(f"{online.PROVIDERS[prov]['name']}: {check()}")
            except online.OnlineError as e:
                print(f"{online.PROVIDERS[prov]['name']}: {e}")
        return 0
    if not a.prompt or not a.out:
        raise SystemExit("error: give a prompt and -o")
    client = None
    if "gpu" in providers and a.comfyui:
        from .generate import ComfyClient
        client = ComfyClient(a.comfyui)
    cands = online.prompt_candidates(a.prompt, _settings(a), providers, seed=a.seed,
                                     pixellab_model=a.pixellab_model, rd_style=a.rd_style,
                                     view=a.view, direction=a.direction, comfy_client=client,
                                     progress=print)
    for c in cands:
        name = online.PROVIDERS[c.provider]["name"]
        print(f"  {name}: " + (f"failed: {c.error}" if c.error
                                else f"{c.seconds:.0f} s {c.cost}".rstrip()))
    written = online.write_candidates(cands, a.out)
    _print_written(f"“{a.prompt}”", written)
    return 0 if any(c.sprite is not None for c in cands) else 1


# ---- demo ----------------------------------------------------------------------------------

def _add_demo(sub):
    p = sub.add_parser(
        "demo", help="a ready-to-open Godot 4 project: terrain map + units that walk where you click",
        description="Writes a Godot 4 project (640x360, x3 = 1080p, integer scaling): a map from map "
                    "tiles and units in 8/16 directions with team colours. Click your unit, then "
                    "a tile: it walks there (A* around water), then greys out; Space = next turn. "
                    "With no --unit, a built-in knight with idle and walk animations is used.")
    p.add_argument("-o", "--out", type=Path, required=True, help="project folder (new or a demo)")
    p.add_argument("--unit", nargs="+", type=Path,
                   help="3D model(s) (rigged + animated .glb plays its animations), or photos/a "
                        "video around a subject, as for turnaround")
    p.add_argument("--still", dest="animate", action="store_false", default=None,
                   help="don't play the model's animations")
    p.add_argument("--animations", type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
                   help="which animations (default all); the demo uses the ones named idle and walk")
    p.add_argument("--colours-from", type=Path, metavar="MODEL", help="as for turnaround")
    p.add_argument("--directions", type=int, choices=DIRECTION_COUNTS, default=8)
    p.add_argument("--frames", type=int, default=6, help="frames per animation")
    p.add_argument("--elevation", type=float, default=30.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--terrains", default="grass,water,sand,road",
                   help="base first; built-ins: grass water sand road dirt stone snow; or "
                        "name=photo.jpg / name=prompt:<text> as for maptiles. Water blocks walking")
    p.add_argument("--tile", type=int, default=32, help="tile size in px (default 32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true",
                   help="also save <out>_preview.png next to the folder: the start screen")
    _add_setting_args(p, default_subject="person")
    p.set_defaults(func=_cmd_demo)


def _cmd_demo(a) -> int:
    from PIL import Image

    from .demo import make_demo, parse_terrains
    for path in [*(a.unit or []), *([a.colours_from] if a.colours_from else [])]:
        if not path.exists():
            raise SystemExit(f"error: {path} not found")
    if a.width is None and a.height is None and not a.system:
        a.width = a.height = 40                   # a bit over one 32 px tile
    s = _settings(a)
    names, sources = parse_terrains(t for t in a.terrains.split(",") if t.strip())
    written, preview = make_demo(a.out, a.unit, s, animate=a.animate, animations=a.animations,
                                 colours_from=a.colours_from, directions=a.directions,
                                 frames=a.frames, elevation=a.elevation, yaw=a.yaw,
                                 terrains=names, terrain_sources=sources, tile=a.tile, seed=a.seed)
    if a.preview:
        shot = a.out.parent / f"{a.out.name}_preview.png"
        Image.fromarray(preview, "RGBA").save(shot)
        print(f"Preview of the start screen -> {shot}")
    print(f"Godot project -> {a.out} ({len(written)} files). Open {a.out / 'project.godot'} "
          "in Godot 4.3+ and press F5.")
    return 0


# ---- model ---------------------------------------------------------------------------------

def _add_model(sub):
    p = sub.add_parser(
        "model", help="prompt or picture -> coloured 3D model (.glb) on your GPU, + direction sprites",
        description="Prompt -> front picture (SDXL) -> 3D shape (Hunyuan3D in ComfyUI) -> painted "
                    "from the picture -> .glb. Add --directions to also make 4/8/16-direction "
                    "sprites. Setup: docs/LOCAL_AI.md ('Prompt to 3D model').")
    p.add_argument("prompt", nargs="?", help="what to make, e.g. 'a knight with a red cape'")
    p.add_argument("-o", "--out", type=Path, required=True, help="coloured model .glb")
    p.add_argument("-s", "--subject", choices=("person", "animal", "object"), default="person")
    p.add_argument("--image", type=Path, help="use this front picture instead of a prompt")
    p.add_argument("--back", type=Path, help="optional back-view picture for the far side")
    p.add_argument("--shape", type=Path, help="use this ready shape (.glb) instead of ComfyUI")
    p.add_argument("--seed", type=int)
    p.add_argument("--checkpoint", help="Hunyuan3D checkpoint (default: first one installed)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--octree", type=int, default=256, help="mesh detail 256-512 (default 256)")
    p.add_argument("--template", type=Path,
                   help="your own 3D workflow (API format, {{image}} in, SaveGLB out)")
    p.add_argument("--comfyui", help="ComfyUI address (default $COMFYUI_URL or http://127.0.0.1:8188)")
    p.add_argument("--cutout", default="isnet-general-use", help="background-removal model")
    p.add_argument("--obj", action="store_true",
                   help="also write <out>.obj to upload to Mixamo (skeleton + animations); then "
                        "turnaround --animate <download>.glb --colours-from <out>.glb")
    g = p.add_argument_group("direction sprites")
    g.add_argument("--directions", type=int, choices=DIRECTION_COUNTS,
                   help="also render 4/8/16 facing sprites")
    g.add_argument("--sprites", type=Path, help="sprite sheet .png (default: <out>_dirs.png)")
    g.add_argument("-W", "--width", type=int, default=48)
    g.add_argument("-H", "--height", type=int, default=48)
    g.add_argument("--colors", type=int, default=24)
    g.add_argument("--elevation", type=float, default=30.0)
    g.add_argument("--godot", action="store_true")
    g.add_argument("--aseprite", action="store_true")
    g.add_argument("--preview", type=int, metavar="SCALE")
    p.set_defaults(func=_cmd_model)


def _cmd_model(a) -> int:
    from .generate import ComfyClient
    from .model3d import ShapeOptions, prompt_to_model
    if not a.prompt and not a.image:
        raise SystemExit("error: give a prompt or --image")
    for path in (a.image, a.back, a.shape, a.template):
        if path and not path.exists():
            raise SystemExit(f"error: {path} not found")
    opts = ShapeOptions(checkpoint=a.checkpoint, steps=a.steps, octree=a.octree,
                        template=str(a.template) if a.template else None)
    needs_gpu = a.image is None or a.shape is None
    client = ComfyClient(a.comfyui) if needs_gpu else None
    res = prompt_to_model(a.prompt, a.out, subject=a.subject, picture=a.image, back=a.back,
                          shape=a.shape, client=client, seed=a.seed, shape_options=opts,
                          cutout_model=a.cutout, progress=lambda step, text: print(text))
    written = [res.model, res.picture] + ([res.raw] if res.raw else [])
    if a.obj:
        from .model3d import export_for_rigging
        written.append(export_for_rigging(res.model, res.model.with_suffix(".obj")))
    if a.directions:
        from .turnaround import render_model, turnaround_sprites, union_crop, write_preview, write_turnaround
        s = settings_for(a.subject, width=a.width, height=a.height, colors=a.colors)
        views = union_crop(render_model(res.model, a.directions, a.elevation), s.margin, s.alpha_threshold)
        sprites = turnaround_sprites(views, s)
        sheet = a.sprites or a.out.with_name(a.out.stem + "_dirs.png")
        written += write_turnaround(sprites, sheet, a.aseprite, a.godot)
        if a.preview:
            written.append(write_preview(sprites, sheet.with_name(sheet.stem + "_spin.gif"), a.preview))
    seed = f" (seed {res.seed})" if res.seed is not None else ""
    _print_written(f"{a.prompt or a.image.name}{seed}", written)
    return 0


# ---- teams / portrait ----------------------------------------------------------------------

def _add_teams(sub):
    p = sub.add_parser("teams", help="sprite/sheet -> team colour versions (+ Godot palette-swap shader)",
                       description="Fire Emblem style team colours: the sprite's main saturated "
                                   "colour becomes blue (player), red (enemy), green (ally), "
                                   "purple (other), plus a greyed 'spent' version.")
    p.add_argument("sprite", type=Path, help="sprite or sprite sheet PNG (1x)")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: next to the sprite)")
    p.add_argument("--colour", "--color", dest="colour",
                   help="the team colour to swap, #rrggbb (default: the main saturated colour)")
    p.add_argument("--godot", action="store_true",
                   help="also write the palette strip, team_colours.gdshader and a material per team")
    p.add_argument("--no-png", action="store_true", help="only the Godot files, no PNG per team")
    p.set_defaults(func=_cmd_teams)


def _cmd_teams(a) -> int:
    from PIL import Image

    from .teams import team_variants, write_godot_teams
    if not a.sprite.exists():
        raise SystemExit(f"error: {a.sprite} not found")
    rgba = load_sprite(a.sprite)
    variants, colours, pals = team_variants(rgba, a.colour)
    out = a.out or a.sprite.parent
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if not a.no_png:
        for team, img in variants.items():
            path = out / f"{a.sprite.stem}_{team}.png"
            Image.fromarray(img).save(path)
            written.append(path)
    if a.godot:
        written += write_godot_teams(colours, pals, out, a.sprite.stem)
    changed = int((pals["player"] != colours).any(1).sum()) if "player" in pals else 0
    _print_written(f"{a.sprite.name}: {changed} of {len(colours)} colours are team colours", written)
    if not changed:
        print("No strong team colour found; pass --colour #rrggbb to choose one.")
    return 0


def _add_portrait(sub):
    p = sub.add_parser("portrait", help="photo / picture / prompt -> head-and-shoulders portrait sprite",
                       description="Dialogue portrait: finds the face (or the top of the subject), "
                                   "crops head and shoulders, cuts out the background and pixelizes.")
    p.add_argument("source", nargs="?", type=Path, help="photo or picture (omit with --prompt)")
    p.add_argument("-o", "--out", type=Path, required=True, help="portrait .png")
    p.add_argument("--prompt", help="make the picture on your GPU (ComfyUI) instead")
    p.add_argument("-W", "--width", type=int, default=96)
    p.add_argument("-H", "--height", type=int, default=96)
    p.add_argument("--colors", type=int, default=32)
    p.add_argument("--flip", action="store_true", help="face the other way")
    p.add_argument("--scale", type=int, default=1, help="also save an enlarged copy")
    p.add_argument("--seed", type=int)
    p.add_argument("--comfyui", help="ComfyUI address")
    p.set_defaults(func=_cmd_portrait)


def _cmd_portrait(a) -> int:
    from .teams import PORTRAIT_PROMPT, make_portrait
    if not a.source and not a.prompt:
        raise SystemExit("error: give a picture or --prompt")
    picture = a.source
    if picture is not None and not picture.exists():
        raise SystemExit(f"error: {picture} not found")
    if picture is None:
        from .generate import ComfyClient, generate_images
        from .model3d import MODEL_NEGATIVE
        (img, seed), = generate_images(PORTRAIT_PROMPT.format(prompt=a.prompt), MODEL_NEGATIVE,
                                       client=ComfyClient(a.comfyui), seed=a.seed, aspect=1.0,
                                       lora="none")
        picture = img
        a.out.parent.mkdir(parents=True, exist_ok=True)
        img.save(a.out.with_suffix(".picture.png"))
    s = settings_for("person", width=a.width, height=a.height, colors=a.colors,
                     outline_style="selout")
    sprite = make_portrait(picture, s, a.flip)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sprite.save(a.out, 1)
    written = [a.out]
    if a.scale > 1:
        big = a.out.with_name(f"{a.out.stem}@{a.scale}x.png")
        sprite.save(big, a.scale)
        written.append(big)
    _print_written(f"portrait {a.width}x{a.height}", written)
    return 0


# ---- map tiles -----------------------------------------------------------------------------

def _add_maptiles(sub):
    from .maptiles import BUILTIN as TERRAINS
    p = sub.add_parser(
        "maptiles", help="terrains -> seamless map tiles with corner transitions + Godot TileSet",
        description="The first terrain is the base; each other terrain gets 15 corner tiles "
                    "blending it into the base. Godot: paint with the TileMapLayer's Terrains tab. "
                    f"Built-in terrains: {', '.join(TERRAINS)}.")
    p.add_argument("-o", "--out", type=Path, required=True, help="atlas .png (TileSet .tres next to it)")
    p.add_argument("-t", "--terrain", action="append", metavar="NAME[=SOURCE]",
                   help="repeat, base first. SOURCE: built-in (default), a texture photo, or "
                        "prompt:<text> (made on your GPU). e.g. -t grass -t water "
                        "-t sand=beach.jpg -t 'road=prompt:cobblestone road'")
    p.add_argument("--tile", type=int, default=32, help="tile size in pixels (default 32: 640x360 maps)")
    p.add_argument("--variants", type=int, default=4, help="base tile variations (default 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-godot", dest="godot", action="store_false")
    p.add_argument("--example", default="20x11", help="example map size COLSxROWS, or 'none'")
    p.add_argument("--comfyui", help="ComfyUI address for prompt: terrains")
    p.set_defaults(func=_cmd_maptiles)


def _cmd_maptiles(a) -> int:
    from .maptiles import build_terrain_set, terrain, write_terrain
    specs = a.terrain or ["grass", "water", "sand", "road"]
    client = None
    if any("=prompt:" in t for t in specs):
        from .generate import ComfyClient
        client = ComfyClient(a.comfyui)
    terrains = []
    for spec in specs:
        name, _, source = spec.partition("=")
        terrains.append(terrain(name.strip(), source.strip() or None, a.tile, seed=a.seed, client=client))
    ts = build_terrain_set(terrains, a.variants, a.seed)
    example = None if a.example == "none" else tuple(int(v) for v in a.example.lower().split("x"))
    written = write_terrain(ts, a.out, a.godot, example, a.seed)
    _print_written(f"{len(terrains)} terrains ({', '.join(ts.names)}), {a.tile}px tiles", written)
    return 0


# ---- idle ----------------------------------------------------------------------------------

def _add_idle(sub):
    p = sub.add_parser("idle", help="sprite -> looping idle animation (breathe / bob / sway)")
    p.add_argument("sprite", type=Path, help="a pixel-art sprite PNG (e.g. from convert)")
    _add_animation_out(p, default_tag="idle",
                       duration_help="ms per frame (default: 150, sway 120, rumble 70)")
    p.add_argument("--style", choices=IDLE_STYLES, default="breathe",
                   help="breathe (characters), bob (floating items), sway (plants, hair), "
                        "rumble (vehicles, machines)")
    p.add_argument("--frames", type=int, help="frames in the loop (default 4; sway 8; rumble 2)")
    p.add_argument("--amount", type=int, default=1, help="movement in pixels (default 1)")
    p.set_defaults(func=_cmd_idle)


def _cmd_idle(a) -> int:
    rgba = load_sprite(a.sprite)
    frames = idle_frames(rgba, a.style, a.frames, a.amount)
    written = write_animation(frames, a.out, a.duration or IDLE_MS[a.style], a.tag, a.columns,
                              a.aseprite, a.godot)
    _print_written(f"{a.sprite.name}: {len(frames)}-frame {a.style}", written)
    return 0


# ---- parallax ------------------------------------------------------------------------------

def _floats(text: str) -> list[float]:
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got '{text}'") from None


def _add_parallax(sub):
    p = sub.add_parser("parallax", help="landscape photo -> near/mid/far parallax layers (+ Godot scene)",
                       description="Estimate depth, split the photo into layers, pixelize them with "
                                   "one shared palette, and write PNG layers plus a Godot 4 "
                                   "Parallax2D scene. Works best on scenery.")
    p.add_argument("photo", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="output folder")
    p.add_argument("--name", help="file name prefix (default: photo name)")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("-W", "--width", type=int, default=320)
    p.add_argument("-H", "--height", type=int, default=180)
    p.add_argument("--palette", default="auto")
    p.add_argument("--colors", type=int, default=32)
    p.add_argument("--dither", choices=DITHER_MODES, default="bayer4")
    p.add_argument("--wrap", action=BOOL, default=True, help="layers repeat seamlessly left-right")
    p.add_argument("--scroll", type=_floats, help="scroll speed per layer far->near, e.g. 0.2,0.5,0.85")
    p.add_argument("--depth", type=Path, help="use this depth map instead of the model (white = near)")
    p.add_argument("--cuts", type=_floats, help="depth cut points 0-1, e.g. 0.3,0.6")
    p.add_argument("--fill", choices=("mirror", "rows", "telea", "ns"), default="mirror",
                   help="how hidden areas behind nearer layers are filled")
    p.add_argument("--godot", action=BOOL, default=True, help="write the Godot Parallax2D scene")
    p.add_argument("--scale", type=int, default=1, help="also write an upscaled preview")
    p.add_argument("--save-depth", action="store_true", help="also write the depth map")
    p.set_defaults(func=_cmd_parallax)


def _cmd_parallax(a) -> int:
    from .parallax import make_parallax
    if not a.photo.exists():
        raise SystemExit(f"error: {a.photo} not found")
    res = make_parallax(a.photo, a.out, a.name, a.layers, a.width, a.height, a.palette, a.colors,
                        a.dither, a.wrap, a.scroll, depth=a.depth, thresholds=a.cuts, fill=a.fill,
                        godot=a.godot, scale=a.scale, save_depth=a.save_depth)
    written = list(res["layers"]) + [res["preview"]] + [res[k] for k in ("godot", "depth") if res.get(k)]
    _print_written(f"{a.photo.name}: {len(res['layers'])} layers, scroll {res['scroll_scales']}", written)
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
    _add_generate(sub)
    _add_video(sub)
    _add_turnaround(sub)
    _add_demo(sub)
    _add_ai(sub)
    _add_model(sub)
    _add_teams(sub)
    _add_portrait(sub)
    _add_maptiles(sub)
    _add_idle(sub)
    _add_parallax(sub)
    _add_sheet(sub)
    _add_tileset(sub)
    sub.add_parser("palettes", help="list built-in palettes").set_defaults(func=_cmd_palettes)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CutoutUnavailable, ComfyUIError, ModelUnavailable, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
