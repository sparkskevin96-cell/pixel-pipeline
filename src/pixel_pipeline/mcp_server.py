"""MCP server so Claude (Claude Code / Claude Desktop) can drive pixel-pipeline.

    python -m pixel_pipeline.mcp_server        # stdio; or the pixel-pipeline-mcp script

Adding a tool: write a plain function with `Annotated[..., Field(description=...)]`
params and a docstring that helps Claude pick settings, then decorate it with
`@tool(title=...)`. Raise `ToolError` for anything the caller can fix; ValueError,
CutoutUnavailable and file errors from the pipeline are converted for you. Return
`[summary_text, _png(preview)]` so Claude can see what it made.
"""
import functools
import inspect
import io
import json
import logging
import math
import re
import threading
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.mcpserver import Image as ImageBlock
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydantic import Field

from . import __version__
from .aseprite import write_aseprite
from .cutout import CutoutUnavailable
from .generate import ComfyUIError
from .animate import IDLE_MS, IDLE_STYLES, idle_frames
from .export import load_rgba, load_sprite, write_animation, write_tileset
from .godot import write_tileset as write_godot_tileset
from .palettes import BUILTIN, rgb_to_hex
from .pipeline import convert_batch
from .quantize import DITHER_MODES
from .settings import PRESETS, SUBJECTS, Settings, settings_for
from .systems import LOOKS, READABLE_SIZES, SYSTEM_NAMES, SYSTEMS
from .turnaround import ModelUnavailable, direction_names, turnaround_from, write_turnaround
from .video import ALIGN_MODES, video_to_sprites

log = logging.getLogger("pixel_pipeline.mcp")

PREVIEW_TARGET = 512   # small pixel art is enlarged (nearest) to about this many px
PREVIEW_MAX = 1024     # hard cap on either side of any image returned to Claude
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
OUTPUT_MARKERS = (".sprite.", ".sprite@")  # our outputs; convert_photos skips them in folders/globs

ToolOutput = list[str | ImageBlock]
Subject = Literal[SUBJECTS]
Dither = Literal[DITHER_MODES]
IdleStyle = Literal[IDLE_STYLES]
RetroSystem = Literal[SYSTEM_NAMES]
Look = Literal[LOOKS]
SYSTEM_DOC = ("Retro console look: " + "; ".join(
    f"{k} = {v.name} ({v.readable_colors} colours readable / {v.sprite_colors} authentic)"
    for k, v in SYSTEMS.items())
    + ". Sets the console's palette/colour depth, colours, sizes and outline style "
      "(explicit width/height/colors still win). off = free colours.")
LOOK_DOC = ("With a system: readable (default) keeps a photo recognizable: the console's colours "
            "at " + ", ".join(f"{k} {w}x{h}" for k, (w, h) in READABLE_SIZES.items())
            + " with a multi-palette character's colours. authentic = one sprite of that era "
              "(e.g. NES 16x24, 3 colours): faithful, but photos usually become unreadable blobs.")
Align = Literal[ALIGN_MODES]

GODOT_DOC = ("Also write a Godot 4 SpriteFrames .tres next to the sheet (load it into an "
             "AnimatedSprite2D; texture path is relative, keep the PNG beside it).")

server = MCPServer(
    "pixel-pipeline",
    version=__version__,
    instructions=(
        "pixel-pipeline turns photos into pixel-art game sprites, sprite sheets, tilesets and "
        ".aseprite files, locally and free (classical image processing, no AI service).\n"
        "Typical flow: convert_photos -> look at the returned preview -> adjust width/height, "
        "palette, colors, detail, dither or outline and re-run -> make_sprite_sheet from the "
        "*.sprite@1x.png frames, or make_tileset for backgrounds. preview_image shows any file.\n"
        "Use absolute paths. Relative paths resolve against the server's working directory "
        f"({Path.cwd()}).\n"
        "ASSET STUDIO (building a game): list_projects / create_project (type + engine) -> "
        "set_house_style -> make / make_character / make_item / make_prop / make_terrain_tiles / "
        "make_effect / make_atmosphere / make_portrait / make_card / make_ui_kit / make_map "
        "(plan_asset = dry run with GPU minutes). GPU work runs as background jobs, one at a time: "
        "poll job_status(job, wait_s=60). Characters stop after the FRONT drawing: look at it "
        "(character_review), then character_approve or character_redraw_front / "
        "character_change_words. Always LOOK at the returned previews and quality warnings; "
        "pick_variation picks another icon/drawing/art/texture. Finally export_to_game(project_path, "
        "engine) writes the files + pixel_pipeline_manifest.json into the game (verify=true checks "
        "Godot). Read docs/FOR_CLAUDE.md in the pixel-pipeline repo for timings and gotchas."
    ),
)

# One heavy conversion at a time: keeps memory sane and the rembg session cache single-threaded.
_work_lock = threading.Lock()


# --------------------------------------------------------------------------- registration

def _guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn expected pipeline failures into readable tool errors instead of crashes."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except ModelUnavailable as e:
            raise ToolError(str(e)) from e
        except CutoutUnavailable as e:
            raise ToolError(f"{e}. Pass remove_background=false for photos that already have "
                            "a transparent background, or for scenes.") from e
        except ComfyUIError as e:
            raise ToolError(str(e)) from e
        except UnidentifiedImageError as e:
            raise ToolError(f"Not a readable image: {e}") from e
        except (ValueError, OSError) as e:
            raise ToolError(str(e)) from e
        except Exception as e:  # local tool: show Claude the real cause
            log.exception("%s failed", fn.__name__)
            raise ToolError(f"Unexpected {type(e).__name__}: {e}") from e
    return wrapper


def tool(*, title: str, read_only: bool = False, open_world: bool = False, extra_doc: str = ""):
    """Register `fn` as an MCP tool (docstring + extra_doc = the description Claude reads)."""
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        doc = inspect.cleandoc(fn.__doc__ or "") + (f"\n\n{extra_doc}" if extra_doc else "")
        server.add_tool(
            _guarded(fn), name=fn.__name__, title=title, description=doc, structured_output=False,
            annotations=ToolAnnotations(read_only_hint=read_only, destructive_hint=False,
                                        idempotent_hint=True, open_world_hint=open_world))
        return fn
    return register


# --------------------------------------------------------------------------- paths

def _natural(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def _path(raw: str) -> Path:
    return Path(raw.strip().strip('"').strip("'")).expanduser().absolute()


def _input_files(paths: list[str], skip_outputs: bool = False) -> list[Path]:
    """Files, folders (every image inside) and glob patterns -> existing files, in order.
    skip_outputs drops our own *.sprite*.png results from folder/glob matches."""
    if not paths:
        raise ToolError("Give at least one image path.")
    def keep(h: Path) -> bool:
        return h.is_file() and not (skip_outputs and any(m in h.name for m in OUTPUT_MARKERS))

    out: list[Path] = []
    for raw in paths:
        p = _path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            hits = [h for h in p.iterdir() if h.suffix.lower() in IMAGE_SUFFIXES and keep(h)]
            if not hits:
                raise ToolError(f"No images in folder {p}")
            out += sorted(hits, key=_natural)
        elif any(c in raw for c in "*?["):
            hits = [h for h in Path(p.anchor).glob(str(p.relative_to(p.anchor))) if keep(h)]
            if not hits:
                raise ToolError(f"No files match {p}")
            out += sorted(hits, key=_natural)
        else:
            raise ToolError(f"File not found: {p}")
    return out


def _output_png(raw: str, suffix: str = ".png") -> Path:
    p = _path(raw)
    if p.suffix.lower() != suffix:
        raise ToolError(f"Output must be a {suffix} path, got {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- previews

def _checker(w: int, h: int, cell: int) -> Image.Image:
    light = (np.arange(h)[:, None] // cell + np.arange(w)[None, :] // cell) % 2 == 1
    return Image.fromarray(np.where(light, np.uint8(204), np.uint8(153)), "L")


def _on_checker(img: Image.Image, cell: int) -> Image.Image:
    """Flatten transparency onto a grey checkerboard so the silhouette is visible."""
    img = img.convert("RGBA")
    if img.getextrema()[3][0] == 255:
        return img.convert("RGB")
    bg = _checker(img.width, img.height, max(cell, 8)).convert("RGBA")
    return Image.alpha_composite(bg, img).convert("RGB")


def _enlarge(img: Image.Image, scale: int | None = None) -> tuple[Image.Image, int]:
    """Nearest-upscale small pixel art (default: to ~PREVIEW_TARGET), shrink anything > PREVIEW_MAX."""
    w, h = img.size
    if scale is None:
        scale = max(1, PREVIEW_TARGET // max(w, h))
    scale = max(1, min(scale, PREVIEW_MAX // max(w, h)))
    out = img.convert("RGBA")
    if scale > 1:
        out = out.resize((w * scale, h * scale), Image.NEAREST)
    elif max(w, h) > PREVIEW_MAX:
        out.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return _on_checker(out, scale), scale


def _contact_sheet(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    """Grid of several sprites, enlarged alike, each labelled, on a dark background."""
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw = max(a.shape[1] for a in images)
    ch = max(a.shape[0] for a in images)
    gap, label_h = 8, 14
    scale = max(1, min(PREVIEW_TARGET // max(cw, ch),
                       (PREVIEW_MAX - gap * (cols + 1)) // (cols * cw),
                       (PREVIEW_MAX - (gap + label_h) * rows - gap) // (rows * ch)))
    cell_w, cell_h = cw * scale, ch * scale
    sheet = Image.new("RGB", (gap + cols * (cell_w + gap), gap + rows * (cell_h + label_h + gap)),
                      (48, 48, 48))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (rgba, label) in enumerate(zip(images, labels)):
        x = gap + (i % cols) * (cell_w + gap)
        y = gap + (i // cols) * (cell_h + label_h + gap)
        max_chars = max(4, cell_w // 6)
        draw.text((x, y), label if len(label) <= max_chars else label[:max_chars - 1] + "~",
                  fill=(235, 235, 235), font=font)
        sprite = Image.fromarray(rgba, "RGBA")
        sprite = sprite.resize((sprite.width * scale, sprite.height * scale), Image.NEAREST)
        cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        cell.paste(sprite, ((cell_w - sprite.width) // 2, cell_h - sprite.height))
        sheet.paste(_on_checker(cell, scale), (x, y + label_h))
    if max(sheet.size) > PREVIEW_MAX:
        sheet.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return sheet


def _png(img: Image.Image) -> ImageBlock:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return ImageBlock(data=buf.getvalue(), format="png")


def _hex_list(palette: np.ndarray, limit: int = 64) -> str:
    codes = [rgb_to_hex(c) for c in palette[:limit]]
    more = f" ... (+{len(palette) - limit} more)" if len(palette) > limit else ""
    return " ".join(codes) + more


def _colours_used(rgba: np.ndarray) -> int:
    opaque = rgba.reshape(-1, 4)
    opaque = opaque[opaque[:, 3] > 0]
    return len(np.unique(opaque, axis=0)) if len(opaque) else 0


# --------------------------------------------------------------------------- settings

# Explicit convert_photos parameters; every other Settings field goes through extra_settings.
_DIRECT = {"width", "height", "palette", "colors", "dither", "outline", "detail", "remove_background",
           "system", "look", "subject"}


def _setting_names() -> list[str]:
    return [f.name for f in fields(Settings) if f.name not in _DIRECT]


def _coerce(name: str, value: Any) -> Any:
    """Make an extra_settings value match the Settings field's type (Claude may send strings)."""
    default = {f.name: f.default for f in fields(Settings)}[name]
    try:
        if isinstance(default, bool):
            if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no"):
                return value.lower() in ("true", "1", "yes")
            if not isinstance(value, bool):
                raise TypeError
            return value
        if isinstance(default, int) or default is None:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value)
        if isinstance(default, float):
            return float(value)
        return str(value)
    except (TypeError, ValueError):
        kind = type(default).__name__ if default is not None else "int"
        raise ToolError(f"Setting '{name}' expects {kind}, got {value!r}") from None


def _build_settings(subject: str, explicit: dict, extra: dict | None) -> Settings:
    extra = dict(extra or {})
    unknown = sorted(set(extra) - {f.name for f in fields(Settings)})
    if unknown:
        raise ToolError(f"Unknown setting(s) {', '.join(unknown)}. "
                        f"Valid extra_settings: {', '.join(_setting_names())}")
    values = {k: _coerce(k, v) for k, v in extra.items() if k != "subject"}
    values.update({k: v for k, v in explicit.items() if v is not None})
    return settings_for(subject, **values)


def _presets_doc() -> str:
    lines = ["Subject presets (any parameter left as null keeps the preset value):"]
    for name in PRESETS:
        s = settings_for(name)
        lines.append(
            f"- {name}: {s.width}x{s.height or 'auto'} px, {s.colors} colours, detail {s.detail}, "
            f"dither {s.dither}, outline {'on' if s.outline else 'off'}, background removal "
            f"{'on' if s.remove_background else 'off'}, fit {s.fit}, anchor {s.anchor}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- tools

@tool(title="List palettes", read_only=True)
def list_palettes() -> ToolOutput:
    """List the colour palettes convert_photos accepts (built-ins with their colours, Lospec, files)."""
    lines = ["Built-in palettes (palette=<name>):"]
    for name, colours in BUILTIN.items():
        lines.append(f"  {name:<11} {len(colours):>3} colours: " + " ".join(f"#{c}" for c in colours))
    lines += [
        "",
        "Other palette values:",
        ("  auto (default)  k-means palette picked from the photo(s); size = `colors` (presets: "
         + ", ".join(f"{k} {settings_for(k).colors}" for k in PRESETS) + ")."),
        ("  lospec:<slug>   any palette on https://lospec.com/palette-list, using the slug from "
         "its URL, e.g. lospec:resurrect-64, lospec:apollo, lospec:aap-64, lospec:nyx8. "
         "Downloaded once, cached in ~/.cache/pixel-pipeline/palettes."),
        ("  <file path>     .gpl (GIMP / Aseprite), .hex or .txt (one hex colour per line), or an "
         "image (.png/.gif swatch: every distinct opaque colour, up to 256)."),
        "",
        ("Fixed palettes ignore `colors`. Pick a small one (gameboy, pico-8) for a strong retro "
         "look, auto for faithful colours."),
    ]
    return ["\n".join(lines)]


@tool(title="Convert photos to pixel-art sprites", open_world=True, extra_doc=_presets_doc())
def convert_photos(
    paths: Annotated[list[str], Field(
        description="Photos to convert (absolute paths). Folders (every image inside) and glob "
                    "patterns like C:/photos/walk_*.jpg also work. Several photos = one batch.")],
    subject: Annotated[Subject, Field(
        description="Kind of photo; loads that preset. person/animal/object cut the subject out "
                    "and outline it; background = wide scene, dithered, no cut-out.")] = "object",
    system: Annotated[RetroSystem, Field(description=SYSTEM_DOC)] = "off",
    look: Annotated[Look, Field(description=LOOK_DOC)] = "readable",
    width: Annotated[int | None, Field(
        ge=1, le=1024,
        description="Sprite width in true pixels (before `scale`). Typical: 16-64 for characters "
                    "and items, 160-320 for backgrounds. null = preset.")] = None,
    height: Annotated[int | None, Field(
        ge=0, le=1024,
        description="Sprite height in pixels. 0 = keep the photo's aspect ratio. null = preset.")] = None,
    palette: Annotated[str, Field(
        description="auto (picked from the photo, `colors` entries) | pico-8 | sweetie-16 | "
                    "endesga-32 | gameboy | 1bit | lospec:<slug> | path to a .gpl/.hex/.png "
                    "palette. See list_palettes.")] = "auto",
    colors: Annotated[int | None, Field(
        ge=2, le=256,
        description="Palette size for palette=auto. 4-8 = flat/very retro, 12-16 = classic "
                    "sprite, 24-32+ = painterly or backgrounds. null = preset.")] = None,
    dither: Annotated[Dither | None, Field(
        description="none = clean flat colour (best for characters/items). bayer2/4/8 = ordered "
                    "retro pattern for gradients and backgrounds. floyd = smoothest gradients but "
                    "noisy. null = preset.")] = None,
    outline: Annotated[bool | None, Field(
        description="1px outline around the silhouette (darkest palette colour). null = preset "
                    "(on for person/animal/object).")] = None,
    detail: Annotated[int | None, Field(
        ge=0, le=4,
        description="Edge thickening so thin features (eyes, legs, whiskers) survive the shrink: "
                    "0 = off/soft, 1-2 = normal, 3-4 = bold linework. null = preset.")] = None,
    remove_background: Annotated[bool | None, Field(
        description="Cut the subject out of the photo's background (needs the optional rembg "
                    "extra; first use downloads a ~170 MB model). Images that already have "
                    "transparency are used as-is. null = preset.")] = None,
    scale: Annotated[int, Field(
        ge=1, le=32,
        description="Nearest-neighbour upscale for <stem>.sprite.png. A true-size "
                    "<stem>.sprite@1x.png is always written too.")] = 8,
    out_dir: Annotated[str | None, Field(
        description="Folder for the outputs (created if missing). null = next to each photo.")] = None,
    aseprite: Annotated[bool, Field(
        description="Also write an editable <stem>.sprite.aseprite (indexed colour with the "
                    "palette, so palette swaps work in Aseprite).")] = False,
    shared_palette: Annotated[bool, Field(
        description="Batches only: one palette across all photos, so animation frames or a "
                    "sprite set share identical colours. false = a palette per photo.")] = True,
    extra_settings: Annotated[dict[str, str | int | float | bool] | None, Field(
        description="Advanced: any other setting as {name: value}, e.g. {\"fit\": \"cover\", "
                    "\"anchor\": \"center\", \"patch\": 6, \"tileable\": true, \"outline_color\": "
                    "\"#1a1c2c\", \"dither_strength\": 0.5, \"brightness\": 0.1, \"contrast\": 1.2, "
                    "\"saturation\": 1.3, \"cutout_model\": \"u2net_human_seg\"}. Valid names: "
                    + ", ".join(_setting_names()) + ".")] = None,
) -> ToolOutput:
    """Turn photos into pixel-art sprites (cut out -> crop/fit -> pixelize -> palette -> dither -> outline).

    Writes, per photo: <stem>.sprite.png (enlarged by `scale`, for viewing), <stem>.sprite@1x.png
    (true pixels; use these for make_sprite_sheet / make_tileset / game engines) and optionally
    <stem>.sprite.aseprite. Returns a summary (files, size, colours, palette) and a preview image;
    look at it and re-run with tweaked settings if needed (e.g. raise detail if features vanish,
    lower colors for a flatter look, change width/height for a different pixel density).
    """
    files = _input_files(paths, skip_outputs=True)
    explicit = {"width": width, "height": height, "palette": None if system != "off" else palette,
                "colors": colors, "dither": dither, "outline": outline, "detail": detail,
                "remove_background": remove_background, "system": system, "look": look}
    settings = _build_settings(subject, explicit, extra_settings)
    folder = _path(out_dir) if out_dir else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)

    with _work_lock:
        sprites = convert_batch(files, settings, shared_palette=shared_palette)

    used: set[Path] = set()
    lines, labels = [], []
    for i, (src, sprite) in enumerate(zip(files, sprites), 1):
        where = folder or src.parent
        stem, n = src.stem, 2
        while (where / f"{stem}.sprite.png") in used:  # same stem twice in one batch
            stem, n = f"{src.stem}-{n}", n + 1
        main_png = where / f"{stem}.sprite.png"
        used.add(main_png)
        one_x = where / f"{stem}.sprite@1x.png"
        sprite.save(main_png, scale)
        sprite.save(one_x, 1)
        w, h = sprite.size
        n_used = _colours_used(sprite.rgba)
        lines.append(f"{i}. {src}\n   {w}x{h} px, {n_used} colours used\n"
                     f"   sprite x{scale}: {main_png}\n   1x:        {one_x}")
        if n_used == 0:
            lines.append("   WARNING: empty sprite (nothing opaque). Background removal found no "
                         "subject; try remove_background=false or another cutout_model.")
        if aseprite:
            ase = where / f"{stem}.sprite.aseprite"
            write_aseprite(ase, [sprite.rgba], sprite.palette)
            lines.append(f"   aseprite:  {ase}")
        if not shared_palette or len(sprites) == 1:
            lines.append(f"   palette ({len(sprite.palette)}): {_hex_list(sprite.palette)}")
        labels.append(f"{i} {src.stem}")

    head = (f"Converted {len(files)} photo(s), subject preset '{subject}'"
            + (", shared palette" if shared_palette and len(sprites) > 1 else "") + ".")
    if shared_palette and len(sprites) > 1:
        head += f"\nPalette ({len(sprites[0].palette)}): {_hex_list(sprites[0].palette)}"
    used_settings = " ".join(f"{k}={v}" for k, v in settings.to_dict().items())
    text = f"{head}\n" + "\n".join(lines) + f"\nSettings: {used_settings}"

    if len(sprites) == 1:
        preview, factor = _enlarge(Image.fromarray(sprites[0].rgba, "RGBA"))
        text += f"\nPreview: enlarged x{factor}, transparency shown as a grey checkerboard."
    else:
        preview = _contact_sheet([s.rgba for s in sprites], labels)
        text += "\nPreview: contact sheet in input order (labels = numbers above), checkerboard = transparent."
    return [text, _png(preview)]


@tool(title="Make a sprite sheet")
def make_sprite_sheet(
    frame_paths: Annotated[list[str], Field(
        description="Frame images in animation order (absolute paths; globs like "
                    "C:/out/walk_*.sprite@1x.png and folders work, sorted naturally). Use the "
                    "true-size *.sprite@1x.png files from convert_photos.")],
    out_png: Annotated[str, Field(description="Sheet .png to write; a .json goes next to it.")],
    tag: Annotated[str | None, Field(
        description="Animation name, e.g. idle, walk, run (written as a frame tag).")] = None,
    duration_ms: Annotated[int, Field(ge=1, le=60000, description="Milliseconds per frame.")] = 100,
    columns: Annotated[int | None, Field(
        ge=1, description="Frames per row. null = roughly square grid; use the frame count for "
                          "a single-row strip.")] = None,
    aseprite: Annotated[bool, Field(
        description="Also write an animated .aseprite next to the sheet (all frames must be the "
                    "same size).")] = False,
    godot: Annotated[bool, Field(description=GODOT_DOC)] = False,
) -> ToolOutput:
    """Pack frames into a sprite sheet PNG plus a .json atlas (Aseprite json-array format, which
    Godot, Unity, GameMaker and Phaser importers read), optionally an animated .aseprite."""
    files = _input_files(frame_paths)
    out = _output_png(out_png)
    frames = [load_rgba(f) for f in files]
    sizes = sorted({(f.shape[1], f.shape[0]) for f in frames})
    if aseprite and len(sizes) != 1:
        raise ToolError("An animated .aseprite needs all frames the same size; got "
                        + ", ".join(f"{w}x{h}" for w, h in sizes))
    cols = columns or math.ceil(math.sqrt(len(frames)))
    written = write_animation(frames, out, duration_ms, tag, cols, aseprite, godot,
                              names=[f.stem for f in files])
    meta = json.loads(written[1].read_text(encoding="utf-8"))["meta"]["size"]
    text = (f"{len(frames)} frames ({', '.join(f'{w}x{h}' for w, h in sizes)}) -> "
            f"{cols} columns, sheet {meta['w']}x{meta['h']} px, {duration_ms} ms/frame"
            + (f", tag '{tag}'" if tag else "") + "\n"
            + "\n".join(f"  {p}" for p in written)
            + "\nFrames in order:\n" + "\n".join(f"  {i}. {f}" for i, f in enumerate(files, 1)))
    if any(f.name.endswith(".sprite.png") and f.with_name(f.name[:-4] + "@1x.png").exists()
           for f in files):
        text += ("\nNote: some frames are the enlarged *.sprite.png previews; the true-size "
                 "*.sprite@1x.png files are usually what a sheet should use.")
    with Image.open(out) as sheet:
        preview, factor = _enlarge(sheet)
    return [text + f"\nPreview: sheet enlarged x{factor}.", _png(preview)]


@tool(title="Make a tileset")
def make_tileset(
    image_path: Annotated[str, Field(
        description="Pixel-art image to cut up, usually a 1x background from convert_photos "
                    "(subject=background, ideally with extra_settings {\"tileable\": true}).")],
    out_png: Annotated[str, Field(description="Tileset .png to write; a Tiled .tmj map goes next to it.")],
    tile: Annotated[int, Field(ge=1, le=512, description="Tile size in pixels (8, 16, 32...).")] = 16,
    columns: Annotated[int, Field(ge=1, le=256, description="Tiles per row in the tileset image.")] = 8,
    godot: Annotated[bool, Field(
        description="Also write a Godot 4 TileSet .tres (one atlas source) for TileMapLayer.")] = False,
) -> ToolOutput:
    """Cut a pixel-art image into tile x tile tiles, drop duplicates and empty tiles, and write
    the tileset PNG plus a Tiled map (.tmj) that rebuilds the image (Godot, LDtk and most
    engines import Tiled maps). Width/height should be multiples of `tile`; leftovers on the
    right/bottom edge are ignored."""
    src = _input_files([image_path])[0]
    out = _output_png(out_png)
    rgba = load_rgba(src)
    tmj_path = write_tileset(rgba, tile, out, columns)
    tmj = json.loads(tmj_path.read_text(encoding="utf-8"))
    count = tmj["tilesets"][0]["tilecount"]
    grid = f"{tmj['width']}x{tmj['height']}"
    h, w = rgba.shape[:2]
    text = (f"{src.name} ({w}x{h} px) -> {count} unique {tile}px tiles from a {grid} grid "
            f"({tmj['width'] * tmj['height']} cells)\n  {out}\n  {tmj_path}")
    if godot:
        ts = tmj["tilesets"][0]
        tres = write_godot_tileset(out.with_suffix(".tres"), out.name, tile, ts["columns"], count)
        text += f"\n  {tres}"
    if w % tile or h % tile:
        text += f"\nNote: {w % tile} px right / {h % tile} px bottom didn't fill a whole tile and were dropped."
    with Image.open(out) as tiles:
        preview, factor = _enlarge(tiles)
    return [text + f"\nPreview: tileset enlarged x{factor}.", _png(preview)]


def _animation_result(head: str, frames: list[np.ndarray], written: list[Path]) -> ToolOutput:
    text = head + "\n" + "\n".join(f"  {p}" for p in written)
    preview = _contact_sheet(frames, [str(i) for i in range(len(frames))])
    return [text + "\nPreview: frames in order (checkerboard = transparent).", _png(preview)]


@tool(title="Video to sprite animation", open_world=True, extra_doc=_presets_doc())
def video_to_animation(
    video_path: Annotated[str, Field(description="Video file (.mp4/.mov/...), absolute path.")],
    out_png: Annotated[str, Field(description="Sprite sheet .png to write (.json etc. go next to it).")],
    subject: Annotated[Subject, Field(description="Preset for the subject in the clip.")] = "animal",
    system: Annotated[RetroSystem, Field(description=SYSTEM_DOC)] = "off",
    look: Annotated[Look, Field(description=LOOK_DOC)] = "readable",
    frames: Annotated[int, Field(ge=2, le=64, description="How many frames to sample.")] = 8,
    start: Annotated[float | None, Field(ge=0, description="Start time in seconds.")] = None,
    end: Annotated[float | None, Field(
        ge=0, description="End time in seconds. For a seamless loop pick one full stride/cycle "
                          "(frames are spread evenly, end exclusive).")] = None,
    fps: Annotated[float | None, Field(
        gt=0, le=60, description="Sample at this rate instead of spreading `frames` evenly.")] = None,
    align: Annotated[Align, Field(
        description="track = subject stays centred on the floor (walk/run in place, works with a "
                    "hand-held camera). fixed = one crop for all frames, keeps real movement "
                    "(jumps, crossing the shot); needs a still camera.")] = "track",
    width: Annotated[int | None, Field(ge=1, le=1024, description="Frame width. null = preset.")] = None,
    height: Annotated[int | None, Field(
        ge=0, le=1024, description="Frame height, 0 = aspect. null = preset.")] = None,
    palette: Annotated[str, Field(description="Same values as convert_photos.")] = "auto",
    colors: Annotated[int | None, Field(ge=2, le=256, description="Palette size for auto.")] = None,
    remove_background: Annotated[bool | None, Field(
        description="Cut the subject out of every frame (slow-ish: ~1-4 s per frame on CPU). "
                    "null = preset.")] = None,
    tag: Annotated[str, Field(description="Animation name.")] = "walk",
    duration_ms: Annotated[int | None, Field(
        ge=1, le=60000, description="ms per frame. null = real time between sampled frames.")] = None,
    aseprite: Annotated[bool, Field(description="Also write an animated .aseprite.")] = True,
    godot: Annotated[bool, Field(description=GODOT_DOC)] = False,
    extra_settings: Annotated[dict[str, str | int | float | bool] | None, Field(
        description="Any other setting, as in convert_photos (e.g. outline_style: selout).")] = None,
) -> ToolOutput:
    """Turn a short video (e.g. a phone clip of a dog walking) into a looping sprite animation:
    samples frames, cuts out the subject, aligns every frame to the same box, pixelizes them with
    ONE shared palette, and writes a one-row sprite sheet + .json (+ .aseprite / Godot .tres)."""
    src = _input_files([video_path])[0]
    out = _output_png(out_png)
    explicit = {"width": width, "height": height, "palette": None if system != "off" else palette,
                "colors": colors, "remove_background": remove_background, "system": system, "look": look}
    settings = _build_settings(subject, explicit, extra_settings)
    with _work_lock:
        sprites, ms = video_to_sprites(src, settings, frames, start, end, fps, align)
    rgbas = [s.rgba for s in sprites]
    ms = duration_ms or ms
    written = write_animation(rgbas, out, ms, tag, None, aseprite, godot, palette=sprites[0].palette)
    w, h = sprites[0].size
    head = (f"{src.name}: {len(rgbas)} frames {w}x{h} px @ {ms} ms, align={align}, "
            f"palette ({len(sprites[0].palette)}): {_hex_list(sprites[0].palette)}")
    if any(not f[..., 3].any() for f in rgbas):
        head += "\nWARNING: some frames are empty; try remove_background=false or another cutout_model."
    return _animation_result(head, rgbas, written)


@tool(title="Turnaround: 4, 8 or 16 facing directions", open_world=True, extra_doc=_presets_doc())
def turnaround(
    source: Annotated[list[str], Field(
        description="ONE 3D model (.glb .gltf .obj .ply .stl, e.g. from an image-to-3D AI like "
                    "Hunyuan3D / Stable Fast 3D / TripoSR / Tripo / Meshy), OR one video of walking "
                    "all the way around a still subject (start in front, walk to your right), OR "
                    "one photo per direction in that order (files, a folder or a glob).")],
    out_png: Annotated[str, Field(description="Sprite sheet .png to write (.json etc. go next to it).")],
    directions: Annotated[Literal[4, 8, 16], Field(description="Facing directions.")] = 16,
    subject: Annotated[Subject, Field(description="Preset (size, colours, outline).")] = "person",
    system: Annotated[RetroSystem, Field(description=SYSTEM_DOC)] = "off",
    look: Annotated[Look, Field(description=LOOK_DOC)] = "readable",
    width: Annotated[int | None, Field(ge=4, le=1024, description="Sprite width. null = preset.")] = None,
    height: Annotated[int | None, Field(ge=4, le=1024, description="Sprite height. null = preset.")] = None,
    colors: Annotated[int | None, Field(ge=2, le=256, description="Palette size. null = preset.")] = None,
    elevation: Annotated[float, Field(
        ge=0, le=90, description="3D model: camera tilt in degrees. 0 = side view, 30 = classic "
                                 "top-down / isometric, 60+ = nearly overhead.")] = 30.0,
    up: Annotated[Literal["y", "z"], Field(description="3D model: its up axis (glTF = y).")] = "y",
    yaw: Annotated[float, Field(
        description="3D model: degrees to turn it if its front doesn't face +z (check the S view: "
                    "it should face the viewer).")] = 0.0,
    light: Annotated[float, Field(ge=0, le=1, description="3D model: shading, 0 = flat.")] = 0.35,
    reverse: Annotated[bool, Field(description="Photos/video: walked around to the left.")] = False,
    mirror: Annotated[bool, Field(
        description="Photos/video: only the front-to-back half was captured "
                    "(directions/2 + 1 views); mirror the other side.")] = False,
    start: Annotated[float | None, Field(ge=0, description="Video: start time, seconds.")] = None,
    end: Annotated[float | None, Field(ge=0, description="Video: end time, seconds.")] = None,
    auto_rig: Annotated[bool, Field(
        description="3D humanoid without animations (e.g. from prompt_to_model): give it a "
                    "skeleton and idle/walk/run/attack/hurt/death first, then animate. `animations` "
                    "picks which.")] = False,
    animal: Annotated[bool, Field(
        description="auto_rig: a four-legged animal (also when subject is animal): skeleton + "
                    "idle/walk/run/sit/happy/lie_down/death. The rigged model faces +z.")] = False,
    head: Annotated[Literal["auto", "+x", "-x", "+z", "-z"], Field(
        description="auto_rig animal: which end of the model is the head (auto = from the "
                    "shape).")] = "auto",
    animate: Annotated[bool, Field(
        description="3D model: play its animations (a rigged, animated .glb/.gltf, e.g. from "
                    "Mixamo). Extra .glb files in `source` add their animations to the first "
                    "model's bones by name. Godot animations are named action_direction "
                    "(walk_SE) and the script has face(velocity) + act(&\"walk\").")] = False,
    frames: Annotated[int, Field(ge=1, le=64, description="animate: frames per animation.")] = 8,
    animations: Annotated[list[str] | None, Field(
        description="animate: which animations, in order (default all); rename with "
                    "'new=old'.")] = None,
    in_place: Annotated[bool, Field(
        description="animate: remove steady root motion so walks stay on the spot.")] = True,
    colours_from: Annotated[str | None, Field(
        description="animate: the original coloured model, when rigging it (Mixamo) dropped "
                    "its colours. Colours are matched by shape.")] = None,
    aseprite: Annotated[bool, Field(description="Also write .aseprite with a tag per direction.")] = True,
    godot: Annotated[bool, Field(
        description="Also write Godot SpriteFrames (an animation per direction, E ESE SE ... "
                    "clockwise like Vector2.angle()) and a facing script.")] = True,
    extra_settings: Annotated[dict[str, str | int | float | bool] | None, Field(
        description="Any other setting, as in convert_photos.")] = None,
) -> ToolOutput:
    """Make one sprite per facing direction (4, 8 or 16), all with one palette and one scale:
    from a 3D model (rendered from every direction, like Diablo/StarCraft pre-rendered sprites)
    or from photos/a video taken while walking around a still subject. To go from a prompt:
    generate or draw a front view, turn it into a 3D model with an image-to-3D AI, then pass the
    .glb here. Look at the preview: if S doesn't face the viewer, fix `yaw` (3D) or `reverse`."""
    srcs = _input_files(source)
    out = _output_png(out_png)
    explicit = {"width": width, "height": height, "colors": colors, "system": system, "look": look}
    settings = _build_settings(subject, explicit, extra_settings)
    if auto_rig:
        rig_out = out.with_name(out.stem + ".rigged.glb")
        if animal or subject == "animal":
            from .quadruped import ACTIONS, auto_rig_quadruped
            wanted = [a for a in (animations or ACTIONS) if a in ACTIONS] or list(ACTIONS)
            with _work_lock:
                rigged = auto_rig_quadruped(srcs[0], rig_out, actions=wanted, head=head, up=up, yaw=yaw)
        else:
            from .autorig import ACTIONS
            from .autorig import auto_rig as rig_model
            wanted = [a for a in (animations or ACTIONS) if a in ACTIONS] or list(ACTIONS)
            with _work_lock:
                rigged = rig_model(srcs[0], rig_out, actions=wanted, up=up, yaw=yaw)
        srcs, animate, animations, yaw = [rigged.path], True, None, 0.0
    if animate or animations:
        from .anim3d import animated_turnaround, write_animated
        with _work_lock:
            anim, what = animated_turnaround(srcs, settings, directions, frames, animations,
                                             elevation=elevation, yaw=yaw, light=light,
                                             in_place=in_place,
                                             colours_from=_input_files([colours_from])[0]
                                             if colours_from else None)
        written = write_animated(anim, out, aseprite, godot)
        first = anim.sprites[0][0][0]
        w, h = first.size
        text = (f"{what} -> {len(anim.actions)} animations x {len(anim.directions)} directions x "
                f"{anim.frames} frames, {w}x{h} px, palette ({len(first.palette)}): "
                f"{_hex_list(first.palette)}\n" + "\n".join(f"  {p}" for p in written))
        d = anim.directions.index("SE" if "SE" in anim.directions else "S")
        preview = _contact_sheet([s.rgba for a in anim.sprites for s in a[d]],
                                 [f"{act.name} {i}" for act in anim.actions for i in range(anim.frames)])
        return [text + f"\nPreview: every frame of each animation facing {anim.directions[d]}.",
                _png(preview)]
    with _work_lock:
        sprites, what = turnaround_from(srcs, settings, directions, elevation=elevation, up=up,
                                        yaw=yaw, light=light, reverse=reverse, mirror=mirror,
                                        start=start, end=end)
    written = write_turnaround(sprites, out, aseprite, godot)
    w, h = sprites[0].size
    text = (f"{what} -> {len(sprites)} directions, {w}x{h} px, palette ({len(sprites[0].palette)}): "
            f"{_hex_list(sprites[0].palette)}\n" + "\n".join(f"  {p}" for p in written))
    preview = _contact_sheet([s.rgba for s in sprites], direction_names(directions))
    return [text + "\nPreview: every direction, clockwise from facing right (E).", _png(preview)]


@tool(title="Prompt to 3D model (local GPU)", open_world=True)
def prompt_to_model(
    out_glb: Annotated[str, Field(description="Coloured model to write (.glb). The front picture "
                                              "(<name>.front.png) and raw shape go next to it.")],
    prompt: Annotated[str | None, Field(
        description="What to make, e.g. 'a knight with a red cape'. Plain words; it's wrapped "
                    "into a clean full-body front-view prompt.")] = None,
    subject: Annotated[Literal["person", "animal", "object"], Field(
        description="Shapes the picture prompt.")] = "person",
    picture: Annotated[str | None, Field(
        description="Use this front picture instead of a prompt (absolute path).")] = None,
    back: Annotated[str | None, Field(
        description="Optional back-view picture, used to colour the far side.")] = None,
    shape: Annotated[str | None, Field(
        description="Use this ready shape (.glb) instead of ComfyUI; needs `picture`.")] = None,
    seed: Annotated[int | None, Field(description="Re-make a specific result.")] = None,
    directions: Annotated[Literal[0, 4, 8, 16], Field(
        description="Also make this many facing sprites (0 = model only).")] = 16,
    width: Annotated[int, Field(ge=8, le=512, description="Sprite width.")] = 48,
    height: Annotated[int, Field(ge=8, le=512, description="Sprite height.")] = 48,
    colors: Annotated[int, Field(ge=2, le=64, description="Sprite palette size.")] = 24,
    elevation: Annotated[float, Field(ge=0, le=89, description="Sprite camera tilt, degrees.")] = 30.0,
) -> ToolOutput:
    """Make a coloured 3D model on the user's own GPU: prompt -> clean front picture (SDXL) ->
    3D shape (Hunyuan3D in ComfyUI) -> painted from the picture -> .glb, then (by default)
    16-direction sprites with Godot/Aseprite files. Takes 1-3 minutes. The back is guessed
    from the front unless `back` is given. If ComfyUI or the Hunyuan3D checkpoint is
    missing, the error says what to install (docs/LOCAL_AI.md)."""
    from .model3d import prompt_to_model as _make
    out = _output_png(out_glb, ".glb")
    pic = _input_files([picture])[0] if picture else None
    bk = _input_files([back])[0] if back else None
    shp = _input_files([shape])[0] if shape else None
    if not prompt and pic is None:
        raise ToolError("Give a prompt or a picture.")
    steps: list[str] = []
    with _work_lock:
        res = _make(prompt, out, subject=subject, picture=pic, back=bk, shape=shp, seed=seed,
                    progress=lambda step, text: steps.append(text))
        written = [res.model, res.picture] + ([res.raw] if res.raw else [])
        sprites = None
        if directions:
            from .turnaround import render_model, turnaround_sprites, union_crop
            s = settings_for(subject, width=width, height=height, colors=colors)
            views = union_crop(render_model(res.model, directions, elevation),
                               s.margin, s.alpha_threshold)
            sprites = turnaround_sprites(views, s)
            written += write_turnaround(sprites, res.model.with_name(res.model.stem + "_dirs.png"),
                                        aseprite=True, godot=True)
    seed_text = f" (seed {res.seed})" if res.seed is not None else ""
    text = ("\n".join(steps) + f"\nModel{seed_text}:\n" + "\n".join(f"  {p}" for p in written))
    blocks: ToolOutput = [text]
    front = Image.open(res.picture).convert("RGBA") if res.picture.suffix == ".png" else None
    if front is not None:
        blocks.append(_png(_enlarge(front, 1)[0] if max(front.size) <= 512 else front.resize(
            (512, round(512 * front.size[1] / front.size[0])))))
    if sprites:
        blocks.append(_png(_contact_sheet([sp.rgba for sp in sprites], direction_names(directions))))
    return blocks


@tool(title="Team colours (Fire Emblem style)")
def team_colours(
    sprite_path: Annotated[str, Field(description="Sprite or sprite sheet PNG (1x), absolute path.")],
    out_dir: Annotated[str | None, Field(description="Folder for the results (default: next to it).")] = None,
    colour: Annotated[str | None, Field(
        description="The colour to swap, #rrggbb (default: the main saturated colour, "
                    "e.g. the armour).")] = None,
    godot: Annotated[bool, Field(
        description="Also write the palette strip, team_colours.gdshader and a ShaderMaterial per "
                    "team, so one sprite sheet serves every team in Godot.")] = True,
) -> ToolOutput:
    """Recolour a sprite for every side: player blue, enemy red, ally green, other purple and a
    greyed 'spent' (already moved) version. Only the team colour family changes; skin, gold
    trim and outlines stay. Check the preview: if the wrong colour moved, pass `colour`."""
    from .teams import team_variants, write_godot_teams
    src = _input_files([sprite_path])[0]
    rgba = load_sprite(src)
    variants, colours, pals = team_variants(rgba, colour)
    out = _path(out_dir) if out_dir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for team, img in variants.items():
        path = out / f"{src.stem}_{team}.png"
        Image.fromarray(img).save(path)
        written.append(path)
    if godot:
        written += write_godot_teams(colours, pals, out, src.stem)
    changed = int((pals["player"] != colours).any(1).sum())
    text = (f"{changed} of {len(colours)} colours are team colours"
            + ("" if changed else " (none found: pass `colour`)") + "\n"
            + "\n".join(f"  {p}" for p in written))
    return [text, _png(_contact_sheet([rgba, *variants.values()], ["drawn", *variants]))]


@tool(title="Portrait (dialogue bust)", open_world=True)
def portrait(
    out_png: Annotated[str, Field(description="Portrait .png to write.")],
    source: Annotated[str | None, Field(description="Photo or picture, absolute path.")] = None,
    prompt: Annotated[str | None, Field(
        description="Make the picture on the user's GPU (ComfyUI) instead, e.g. 'a young knight "
                    "with red hair'.")] = None,
    width: Annotated[int, Field(ge=16, le=512)] = 96,
    height: Annotated[int, Field(ge=16, le=512)] = 96,
    colors: Annotated[int, Field(ge=2, le=64)] = 32,
    flip: Annotated[bool, Field(description="Face the other way (dialogue partner).")] = False,
    seed: Annotated[int | None, Field(description="Re-make a specific prompt result.")] = None,
) -> ToolOutput:
    """Head-and-shoulders dialogue portrait: finds the face (or the top of the subject), cuts
    out the background and pixelizes with a sel-out outline."""
    from .teams import PORTRAIT_PROMPT, make_portrait
    out = _output_png(out_png)
    if source:
        picture = _input_files([source])[0]
    elif prompt:
        from .generate import generate_images
        from .model3d import MODEL_NEGATIVE
        with _work_lock:
            (picture, seed), = generate_images(PORTRAIT_PROMPT.format(prompt=prompt), MODEL_NEGATIVE,
                                               seed=seed, aspect=1.0, lora="none")
        picture.save(out.with_suffix(".picture.png"))
    else:
        raise ToolError("Give a source picture or a prompt.")
    s = settings_for("person", width=width, height=height, colors=colors, outline_style="selout")
    with _work_lock:
        sprite = make_portrait(picture, s, flip)
    sprite.save(out, 1)
    preview, factor = _enlarge(Image.fromarray(sprite.rgba, "RGBA"))
    return [f"Portrait {width}x{height}, {len(sprite.palette)} colours"
            + (f", seed {seed}" if prompt else "") + f"\n  {out}\nPreview x{factor}.", _png(preview)]


@tool(title="Map tiles: top-down, side-scroller or isometric", open_world=True)
def map_tiles(
    out_png: Annotated[str, Field(description="Atlas .png to write (TileSet .tres and example map next to it).")],
    terrains: Annotated[list[str], Field(
        description="Base terrain first, then the others, each 'name' (built-in: grass, water, "
                    "sand, road, dirt, stone, snow), 'name=C:/path/texture.jpg' (a texture photo) "
                    "or 'name=prompt:<text>' (made on the user's GPU).")] = ["grass", "water", "sand", "road"],  # noqa: B006
    tile: Annotated[int, Field(ge=8, le=256, description="Tile size (32 suits a 640x360 map view).")] = 32,
    variants: Annotated[int, Field(ge=1, le=15, description="Base tile variations.")] = 4,
    seed: Annotated[int, Field(description="Pattern seed.")] = 0,
    style: Annotated[Literal["topdown", "sidescroller", "isometric"], Field(
        description="topdown: corner transitions into the base. sidescroller: terrains = ground "
                    "then its top (e.g. dirt, grass): 16 edge tiles with collision + one-way "
                    "platforms. isometric: a floor diamond and a block per terrain (tile = width).")] = "topdown",
) -> ToolOutput:
    """Seamless pixel-art terrain tiles for a tactics/top-down map: each non-base terrain gets
    15 corner tiles blending it into the base (shores, edges), plus a Godot TileSet with a
    Match Corners terrain set so painting in Godot picks the right tiles, and an example map.
    Terrains only blend into the base: keep base between two other terrains."""
    from .maptiles import terrain
    from .tilesets import DEFAULT_TERRAINS, build_style, write_style
    out = _output_png(out_png)
    if style != "topdown" and terrains == ["grass", "water", "sand", "road"]:
        terrains = DEFAULT_TERRAINS[style]
    built = []
    with _work_lock:
        for spec in terrains:
            name, _, source = spec.partition("=")
            built.append(terrain(name.strip(), source.strip() or None, tile, seed=seed))
        ts = build_style(style, built, variants, seed)
        written = write_style(style, ts, out, True, True, seed)
    example = Image.open(out.with_name(out.stem + "_example.png")).convert("RGBA")
    return [f"{style}: {len(built)} terrains ({', '.join(t.name for t in built)}), {tile}px tiles\n"
            + "\n".join(f"  {p}" for p in written) + "\nPreview: the example map.", _png(example)]


@tool(title="Prompt to sprite: PixelLab vs Retro Diffusion vs your GPU", open_world=True,
      extra_doc=_presets_doc())
def online_sprites(
    prompt: Annotated[str, Field(description="What to make, plain words (only this text is sent).")],
    out_png: Annotated[str, Field(description="Base .png name; writes <name>_<provider>.png, "
                                              ".raw.png and <name>_compare.png.")],
    providers: Annotated[list[Literal["pixellab", "retrodiffusion", "gpu"]], Field(
        description="Who makes it. pixellab / retrodiffusion need the user's API keys "
                    "(env or ~/.pixel-pipeline/keys.json); gpu = ComfyUI on this PC.")] = ["pixellab", "retrodiffusion"],  # noqa: B006
    subject: Annotated[Subject, Field(description="Preset (colours, outline).")] = "person",
    width: Annotated[int, Field(ge=8, le=512, description="Sprite width.")] = 48,
    height: Annotated[int, Field(ge=8, le=512, description="Sprite height.")] = 64,
    colors: Annotated[int | None, Field(ge=2, le=256, description="Palette size. null = preset.")] = None,
    pixellab_model: Annotated[Literal["pixen", "pixflux", "bitforge", "pro"], Field(
        description="pixen = fast, pro = best (queued).")] = "pixen",
    rd_style: Annotated[str, Field(
        description="Retro Diffusion style id: rd_fast__default ($0.03), rd_plus__default ($0.06), "
                    "rd_pro__default ($0.18), ...")] = "rd_fast__default",
    view: Annotated[Literal["side", "low top-down", "high top-down"] | None, Field(
        description="PixelLab camera.")] = None,
    seed: Annotated[int | None, Field(description="Seed (null = random).")] = None,
) -> ToolOutput:
    """The same prompt made by each provider, side by side: what the service drew and the
    sprite snapped to the settings' palette (service pixel art is kept, not resampled). Costs
    the user's credits for pixellab / retrodiffusion. Look at the comparison and say which one
    reads best at game size."""
    from .online import prompt_candidates, write_candidates
    out = _output_png(out_png)
    settings = _build_settings(subject, {"width": width, "height": height, "colors": colors}, None)
    with _work_lock:
        cands = prompt_candidates(prompt, settings, list(providers), seed=seed,
                                  pixellab_model=pixellab_model, rd_style=rd_style, view=view)
    written = write_candidates(cands, out)
    lines = [f"{c.provider}: " + (f"failed: {c.error}" if c.error else f"{c.seconds:.0f} s {c.cost}".strip())
             for c in cands]
    sheet = Image.open(out.with_name(f"{out.stem}_compare.png")).convert("RGB")
    return ["\n".join(lines) + "\n" + "\n".join(f"  {p}" for p in written)
            + "\nPreview: per provider, what it drew (left) and the sprite on your settings (right).",
            _png(sheet)]


@tool(title="Godot demo project", open_world=True, extra_doc=_presets_doc())
def godot_demo(
    out_dir: Annotated[str, Field(description="Folder for the Godot project (new, or an earlier demo).")],
    unit: Annotated[list[str] | None, Field(
        description="The unit, as for turnaround: a 3D model (a rigged + animated .glb plays its "
                    "animations, idle and walk), more .glb files with animations, or photos/a "
                    "video around a subject. null = a built-in knight with idle and walk.")] = None,
    animate: Annotated[bool | None, Field(
        description="Play the model's animations. null = when it has some.")] = None,
    animations: Annotated[list[str] | None, Field(description="Which animations (default all).")] = None,
    colours_from: Annotated[str | None, Field(description="As for turnaround.")] = None,
    directions: Annotated[Literal[4, 8, 16], Field(description="Facing directions.")] = 8,
    frames: Annotated[int, Field(ge=1, le=32, description="Frames per animation.")] = 6,
    elevation: Annotated[float, Field(ge=0, le=90, description="Camera tilt, degrees.")] = 30.0,
    yaw: Annotated[float, Field(description="Degrees to turn the model so S faces the viewer.")] = 0.0,
    subject: Annotated[Subject, Field(description="Preset (colours, outline).")] = "person",
    width: Annotated[int, Field(ge=8, le=256, description="Unit sprite width.")] = 40,
    height: Annotated[int, Field(ge=8, le=256, description="Unit sprite height.")] = 40,
    colors: Annotated[int | None, Field(ge=2, le=256, description="Palette size. null = preset.")] = None,
    terrains: Annotated[list[str], Field(
        description="As for map_tiles: base first; 'water' blocks walking.")] = ["grass", "water", "sand", "road"],  # noqa: B006
    tile: Annotated[int, Field(ge=8, le=128, description="Tile size; 32 suits 640x360.")] = 32,
    seed: Annotated[int, Field(description="Map seed.")] = 0,
) -> ToolOutput:
    """Write a ready-to-open Godot 4 project that uses the outputs together: 640x360 (x3 =
    1080p, integer scaling, nearest filtering), a terrain map from map tiles, and units in 8/16
    directions with team colours (2 player, 2 enemy). In Godot: click your unit, then a tile:
    it walks there (A* around water) facing where it goes, then greys out; Space = next turn.
    The preview is the start screen drawn here."""
    from .demo import make_demo, parse_terrains
    out = Path(out_dir).expanduser()
    if not out.is_absolute():
        raise ToolError("Use an absolute folder path")
    explicit = {"width": width, "height": height, "colors": colors}
    settings = _build_settings(subject, explicit, None)
    names, sources = parse_terrains(terrains)
    with _work_lock:
        written, preview = make_demo(
            out, _input_files(unit) if unit else None, settings, animate=animate,
            animations=animations, colours_from=_input_files([colours_from])[0] if colours_from else None,
            directions=directions, frames=frames, elevation=elevation, yaw=yaw, terrains=names,
            terrain_sources=sources, tile=tile, seed=seed, progress=lambda text: None)
    text = (f"Godot project in {out} ({len(written)} files): open project.godot in Godot 4.3+ "
            "and press F5.\n" + (out / "README.txt").read_text(encoding="utf-8").splitlines()[-1]
            + "\nPreview: the start screen (player units blue on the left, enemies red).")
    return [text, _png(Image.fromarray(preview, "RGBA"))]


@tool(title="Idle animation from a sprite")
def idle_animation(
    sprite_path: Annotated[str, Field(
        description="A pixel-art sprite PNG (e.g. a *.sprite@1x.png; enlarged exports are "
                    "detected and shrunk back to 1x automatically).")],
    out_png: Annotated[str, Field(description="Sprite sheet .png to write.")],
    style: Annotated[IdleStyle, Field(
        description="breathe = torso squashes 1px, feet planted (characters, animals). bob = whole "
                    "sprite floats (pickups, flying). sway = top leans side to side (plants, "
                    "trees, hair, flags). rumble = body shakes over planted wheels (vehicles, "
                    "machines idling).")] = "breathe",
    frames: Annotated[int | None, Field(
        ge=2, le=32, description="Frames in the loop. null = 4 (sway 8, rumble 2).")] = None,
    amount: Annotated[int, Field(ge=1, le=8, description="Movement in pixels.")] = 1,
    duration_ms: Annotated[int | None, Field(
        ge=1, le=60000, description="ms per frame. null = style default (150; sway 120; rumble 70).")] = None,
    tag: Annotated[str, Field(description="Animation name.")] = "idle",
    aseprite: Annotated[bool, Field(description="Also write an animated .aseprite.")] = True,
    godot: Annotated[bool, Field(description=GODOT_DOC)] = False,
) -> ToolOutput:
    """Make a looping idle animation from one sprite with pure pixel moves (palette-safe, no AI):
    one-row sprite sheet + .json, optionally .aseprite and Godot SpriteFrames."""
    src = _input_files([sprite_path])[0]
    out = _output_png(out_png)
    rgba = load_sprite(src)
    rgbas = idle_frames(rgba, style, frames, amount)
    duration_ms = duration_ms or IDLE_MS[style]
    written = write_animation(rgbas, out, duration_ms, tag, None, aseprite, godot)
    h, w = rgbas[0].shape[:2]
    return _animation_result(f"{src.name}: {len(rgbas)}-frame {style} loop, {w}x{h} px, "
                             f"{duration_ms} ms/frame", rgbas, written)


@tool(title="Parallax background layers", open_world=True)
def parallax_layers(
    photo_path: Annotated[str, Field(description="A scenery photo (landscape, street, forest...).")],
    out_dir: Annotated[str, Field(description="Folder for the layer PNGs, preview and Godot scene.")],
    layers: Annotated[int, Field(ge=2, le=6, description="Number of depth layers, far to near.")] = 3,
    width: Annotated[int, Field(ge=16, le=1920, description="Layer width in pixels.")] = 320,
    height: Annotated[int, Field(ge=16, le=1080, description="Layer height in pixels.")] = 180,
    palette: Annotated[str, Field(description="Same values as convert_photos; one palette for all layers.")] = "auto",
    colors: Annotated[int, Field(ge=2, le=256, description="Palette size for auto.")] = 32,
    dither: Annotated[Dither, Field(description="Dither for all layers.")] = "bayer4",
    wrap: Annotated[bool, Field(description="Layers repeat seamlessly left-right.")] = True,
    scroll_scales: Annotated[list[float] | None, Field(
        description="Godot scroll speed per layer, far to near (default 0.2 / 0.5 / 0.85).")] = None,
    godot: Annotated[bool, Field(description="Write a Godot 4 Parallax2D scene (.tscn).")] = True,
    depth_path: Annotated[str | None, Field(
        description="Optional greyscale depth map to use instead of the model (white = near), "
                    "e.g. one painted in Aseprite.")] = None,
) -> ToolOutput:
    """Split a scenery photo into far/mid/near parallax layers using a local depth model
    (Depth Anything V2 Small, ~95 MB download on first use), pixelize every layer with ONE
    shared palette, and write the layer PNGs, a preview and a Godot 4 Parallax2D scene.
    Best on landscapes; objects that span depths (a cup's rim vs its body) can tear."""
    from .parallax import make_parallax
    src = _input_files([photo_path])[0]
    folder = _path(out_dir)
    with _work_lock:
        res = make_parallax(src, folder, None, layers, width, height, palette, colors, dither,
                            wrap, scroll_scales, godot=godot,
                            depth=_input_files([depth_path])[0] if depth_path else None)
    written = list(res["layers"]) + [res["preview"]] + ([res["godot"]] if res.get("godot") else [])
    text = (f"{src.name}: {len(res['layers'])} layers {width}x{height}, scroll "
            f"{res['scroll_scales']}, depth cuts {[round(c, 3) for c in res['cuts']]}\n"
            + "\n".join(f"  {p}" for p in written))
    with Image.open(res["preview"]) as prev:
        preview, factor = _enlarge(prev)
    return [text + f"\nPreview: all layers stacked, enlarged x{factor}.", _png(preview)]


@tool(title="Generate a sprite on the local GPU", open_world=True, extra_doc=_presets_doc())
def generate_sprite(
    prompt: Annotated[str, Field(description="What to make, e.g. 'a knight with a red cape'. Plain "
                                             "words; it is wrapped into a sprite prompt for `subject`.")],
    out_png: Annotated[str, Field(description="Output .png; the raw AI image is saved as <stem>.raw.png.")],
    subject: Annotated[Subject, Field(description="Shapes the prompt and the conversion preset.")] = "object",
    system: Annotated[RetroSystem, Field(description=SYSTEM_DOC)] = "off",
    look: Annotated[Look, Field(description=LOOK_DOC)] = "readable",
    width: Annotated[int | None, Field(ge=1, le=1024, description="Sprite width. null = preset/system.")] = None,
    height: Annotated[int | None, Field(ge=0, le=1024, description="Sprite height. null = preset/system.")] = None,
    palette: Annotated[str, Field(description="Same values as convert_photos (ignored with a system).")] = "auto",
    colors: Annotated[int | None, Field(ge=2, le=256, description="Palette size. null = preset/system.")] = None,
    count: Annotated[int, Field(ge=1, le=8, description="Variations to make (different seeds).")] = 1,
    seed: Annotated[int | None, Field(description="Re-make a specific result.")] = None,
    fast: Annotated[bool, Field(description="LCM-LoRA: ~3x faster, a little rougher.")] = False,
    init_image: Annotated[str | None, Field(
        description="Sketch to start from (img2img), e.g. a draw_sprite @1x PNG: you draw the "
                    "layout/colours, the GPU adds detail, the converter snaps it back.")] = None,
    denoise: Annotated[float, Field(ge=0.05, le=1.0, description="How much the GPU may change init_image.")] = 0.6,
    negative: Annotated[str | None, Field(description="Negative prompt (null = built-in).")] = None,
    scale: Annotated[int, Field(ge=1, le=32, description="Enlargement for the viewing PNG.")] = 8,
    aseprite: Annotated[bool, Field(description="Also write .aseprite files.")] = False,
    extra_settings: Annotated[dict[str, str | int | float | bool] | None, Field(
        description="Any other conversion setting, as in convert_photos.")] = None,
) -> ToolOutput:
    """Make pixel art from a text prompt on the user's own NVIDIA GPU (ComfyUI + SDXL + Pixel
    Art XL LoRA running locally; setup in docs/LOCAL_AI.md), then convert it exactly like a
    photo (cut-out, size, palette, retro system, outline). Each image takes ~5-15 s. Look at
    the preview: re-run with another seed/count, a sharper prompt, or draw_sprite a sketch and
    pass it as init_image. If ComfyUI isn't running the error says how to start it."""
    from .generate import generate_sprite as _generate
    explicit = {"width": width, "height": height, "palette": None if system != "off" else palette,
                "colors": colors, "system": system, "look": look}
    settings = _build_settings(subject, explicit, extra_settings)
    out = _output_png(out_png)
    init = _input_files([init_image])[0] if init_image else None
    opts = {k: v for k, v in {"fast": fast or None, "negative": negative}.items() if v is not None}
    with _work_lock:
        results = _generate(prompt, settings, subject=subject,
                            system=None if system == "off" else system, seed=seed, count=count,
                            init_image=init, denoise=denoise, **opts)
    lines, labels = [], []
    for i, (sprite, raw, used_seed) in enumerate(results, 1):
        path = out if len(results) == 1 else out.with_name(f"{out.stem}-{i}{out.suffix}")
        one_x = path.with_name(path.stem + "@1x.png")
        raw_path = path.with_name(path.stem + ".raw.png")
        sprite.save(path, scale)
        sprite.save(one_x, 1)
        raw.save(raw_path)
        line = f"{i}. seed {used_seed}: {path} | 1x {one_x} | raw {raw_path}"
        if aseprite:
            ase = path.with_suffix(".aseprite")
            write_aseprite(ase, [sprite.rgba], sprite.palette)
            line += f" | {ase}"
        lines.append(line)
        labels.append(f"{i} seed {used_seed}")
    w, h = results[0][0].size
    text = f"Generated {len(results)} sprite(s) {w}x{h} for '{prompt}':\n" + "\n".join(lines)
    if len(results) == 1:
        preview, _ = _enlarge(Image.fromarray(results[0][0].rgba, "RGBA"))
    else:
        preview = _contact_sheet([r[0].rgba for r in results], labels)
    return [text + "\nPreview: the converted sprite(s).", _png(preview)]


@tool(title="Draw a sprite pixel by pixel")
def draw_sprite(
    rows: Annotated[list[str], Field(
        description="The sprite, one string per pixel row, top to bottom, all the same length. "
                    "'.' = transparent; 0-9 then a-z = index into `palette`.")],
    palette: Annotated[list[str], Field(description="Colours as #rrggbb, indexed by the row characters.")],
    out_png: Annotated[str, Field(description="Where to save the sprite (.png); 1x copy and .aseprite go next to it.")],
    system: Annotated[RetroSystem, Field(
        description="Snap the colours to a retro console's palette/colour depth and its colours-per-"
                    "sprite limit. " + SYSTEM_DOC)] = "off",
    look: Annotated[Look, Field(description="readable = a multi-palette character's colours, "
                                            "authentic = one sprite palette.")] = "readable",
    subject: Annotated[Subject, Field(description="Used for the system's colour limit (background = more).")] = "object",
    scale: Annotated[int, Field(ge=1, le=32, description="Enlargement for the viewing PNG.")] = 8,
    aseprite: Annotated[bool, Field(description="Also write an editable .aseprite.")] = True,
) -> ToolOutput:
    """YOU draw the sprite yourself, pixel by pixel. Good for icons, items, simple characters
    and as a rough sketch for generate_sprite's init_image (the local GPU then renders the
    detail and the converter snaps it back to the grid). Pixel-art tips: 16-48 px, a clear
    silhouette, 1px dark outline, 3-5 shades per material lit from the top-left, no
    anti-aliasing, characters in side view feet on the bottom row. Look at the returned
    preview and redraw anything that reads wrong. The result works with idle_animation,
    make_sprite_sheet and the other tools (use the @1x PNG)."""
    from .drawing import sprite_from_rows
    out = _output_png(out_png)
    sprite = sprite_from_rows(rows, palette, system, subject, look)
    one_x = out.with_name(out.stem + "@1x.png")
    sprite.save(out, scale)
    sprite.save(one_x, 1)
    written = [out, one_x]
    if aseprite:
        ase = out.with_suffix(".aseprite")
        write_aseprite(ase, [sprite.rgba], sprite.palette)
        written.append(ase)
    w, h = sprite.size
    text = (f"Drew {w}x{h} px, {_colours_used(sprite.rgba)} colours"
            + (f" (snapped to {SYSTEMS[system].name})" if system != "off" else "")
            + f": {_hex_list(sprite.palette)}\n" + "\n".join(f"  {p}" for p in written))
    preview, factor = _enlarge(Image.fromarray(sprite.rgba, "RGBA"))
    return [text + f"\nPreview: enlarged x{factor}.", _png(preview)]


@tool(title="Preview an image", read_only=True)
def preview_image(
    path: Annotated[str, Field(description="Any image file: input photo, sprite, sheet, tileset.")],
    scale: Annotated[int | None, Field(
        ge=1, le=64,
        description="Nearest-neighbour enlargement. null = automatic (small pixel art enlarged to "
                    "~512 px, big photos shrunk to fit 1024 px).")] = None,
) -> ToolOutput:
    """Show an image so you can inspect inputs and outputs. Pixel art is enlarged with
    nearest-neighbour so every pixel stays crisp; transparency shows as a grey checkerboard.
    Also reports size and, for small images, the distinct colours."""
    src = _input_files([path])[0]
    with Image.open(src) as img:
        mode, (w, h) = img.mode, img.size
        img = ImageOps.exif_transpose(img).convert("RGBA")  # phone photos: show upright
    big = max(img.size) > PREVIEW_MAX
    if big:
        img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    rgba = np.asarray(img)
    text = f"{src}\n{w}x{h} px, mode {mode}"
    alpha = rgba[..., 3]
    if alpha.min() < 255:
        text += f", {100 * (alpha == 0).mean():.0f}% transparent"
    if not big:
        opaque = rgba[alpha > 0][:, :3]
        colours = np.unique(opaque, axis=0) if len(opaque) else np.empty((0, 3), np.uint8)
        text += f", {len(colours)} distinct colours"
        if 0 < len(colours) <= 32:
            text += f": {_hex_list(colours)}"
    preview, factor = _enlarge(img, scale)
    if scale and factor != scale:
        text += f"\nScale reduced to x{factor} to stay within {PREVIEW_MAX} px."
    shown = f"shrunk to {preview.width}x{preview.height}" if big else f"x{factor}"
    return [text + f"\nPreview: {shown}.", _png(preview)]


# =========================================================================== the asset studio
#
# Everything the app's Make / Characters / Library pages do, for building games with Claude:
# projects + house style, makers (as background jobs: one GPU job at a time, resumable),
# front-first character review, variations, animations, maps, exports into a Godot / Unity
# project with a manifest. The logic lives in agent.py (also used by the CLI's --json); these
# wrappers turn its dicts into JSON text + the preview images so Claude can SEE the result.

from . import agent as _A  # noqa: E402

Kind = Literal["character", "item", "prop", "tileset", "effect", "portrait", "card", "ui", "map",
               "weather", "water", "background"]
AssetKind = Literal["character", "item", "prop", "map", "portrait", "effect", "card", "tileset",
                    "ui", "weather", "water", "background"]
WAIT_DOC = ("Seconds to wait for the job before returning (0 = return the job id at once; poll "
            "job_status). null = 0 for GPU work, up to 300 s for CPU-only work.")
REF_DOC = "Asset id from list_assets (e.g. 'character/knight', 'item/rusty-iron-sword'), or its name."
PROJECT_DOC = "Project name (null = the current project)."
MAX_IMAGES = 3


def _image_block(path: str | Path, small: bool = False) -> ImageBlock | None:
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        return None
    try:
        with Image.open(p) as im:
            im.load()
            img = im.convert("RGBA")
    except (OSError, UnidentifiedImageError):
        return None
    if small:
        img.thumbnail((384, 384), Image.NEAREST)
    elif max(img.size) < 256:
        img, _ = _enlarge(img)
    elif max(img.size) > PREVIEW_MAX:
        img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return _png(_on_checker(img, 8) if img.mode == "RGBA" else img)


def _previews_of(d: dict) -> list[str]:
    """Preview paths inside a result dict (asset summary, job result), best first."""
    out: list[str] = []

    def add(p):
        if p and isinstance(p, str) and p not in out:
            out.append(p)

    def walk(x):
        if isinstance(x, dict):
            pv = x.get("previews")
            if isinstance(pv, dict):
                for k in ("preview", "animations", "variations", "tries", "icon"):
                    add(pv.get(k))
            if isinstance(x.get("variations"), dict):
                add(x["variations"].get("preview"))
            for k in ("asset", "result"):
                if k in x:
                    walk(x[k])
    walk(d)
    return out


def _reply(d: dict, images: bool = True, extra: list | None = None) -> ToolOutput:
    blocks: ToolOutput = [json.dumps(d, indent=1, default=str)]
    if images:
        for p in (list(extra or []) + _previews_of(d))[:MAX_IMAGES]:
            b = _image_block(p)
            if b is not None:
                blocks.append(b)
    return blocks


def studio_tool(*, title: str, read_only: bool = False, destructive: bool = False):
    """Like `tool`, for the studio tools (they may change the project: not idempotent)."""
    def register(fn):
        doc = inspect.cleandoc(fn.__doc__ or "")
        server.add_tool(
            _guarded(fn), name=fn.__name__, title=title, description=doc, structured_output=False,
            annotations=ToolAnnotations(read_only_hint=read_only, destructive_hint=destructive,
                                        idempotent_hint=read_only, open_world_hint=False))
        return fn
    return register


# --------------------------------------------------------------------------- projects + house style

@studio_tool(title="List projects", read_only=True)
def list_projects() -> ToolOutput:
    """Projects in the studio (one per game: its own house style, palette, sizes and assets), which
    is current, and the project types create_project accepts."""
    return _reply(_A.projects(), False)


@studio_tool(title="Create a project")
def create_project(
    name: Annotated[str, Field(description="Project name, usually the game's name.")],
    type: Annotated[Literal["hd2d", "topdown-rpg", "side-scroller", "isometric", "cards-ui", "custom"], Field(
        description="Project type: sets camera, directions, sizes, tile size, clips and exports "
                    "(hd2d = sprites in a 3D world, 16 directions; topdown-rpg = 4 directions, 16 px "
                    "tiles; side-scroller = 2 directions; isometric; cards-ui).")] = "custom",
    size: Annotated[Literal["small", "normal", "large"], Field(description="Sprite size preset of the type.")] = "normal",
    engine: Annotated[Literal["godot", "unity"], Field(description="Target engine.")] = "godot",
    switch: Annotated[bool, Field(description="Make it the current project.")] = True,
    style: Annotated[dict[str, Any] | None, Field(
        description="House-style fields to set right away, e.g. {\"style_prompt\": \"16-bit SNES "
                    "JRPG, bold outlines, warm palette\", \"shading\": \"cel\"}.")] = None,
) -> ToolOutput:
    """Create a project (a game) with a type preset; its house style makes every asset look like one artist."""
    return _reply(_A.create_project(name, type, size, engine, switch, style), False)


@studio_tool(title="Switch project")
def switch_project(name: Annotated[str, Field(description="Project name.")]) -> ToolOutput:
    """Make another project current (the app's current project switches too)."""
    return _reply(_A.switch_project(name), False)


@studio_tool(title="Get the house style", read_only=True)
def get_house_style(project: Annotated[str | None, Field(description=PROJECT_DOC)] = None) -> ToolOutput:
    """The project's house style (Style): style words, palette, colours, outline, character
    height, camera tilt, directions, tile size, ppu, engine, clip set, style LoRA; and the
    prompt words generators add."""
    return _reply(_A.get_style(project), False)


@studio_tool(title="Set the house style")
def set_house_style(
    changes: Annotated[dict[str, Any], Field(
        description="Style fields to change: style_prompt, style_negative, shading (soft|cel|flat), "
                    "proportions (chibi|normal|tall), detail (chunky|normal|fine), palette (auto, a "
                    "built-in, lospec:<slug> or a file), colours, outline (selout|solid|none), "
                    "person_height (px), elevation (deg), directions (2|4|8|16), tile, ppu, icon, "
                    "engine (godot|unity), animations (clip list).")],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Change the house style. New assets use it at once; the reply says what restyle_project
    would redo for existing ones (nothing is redone here)."""
    return _reply(_A.set_style(changes, project), False)


@studio_tool(title="Lock the project palette")
def lock_palette(
    colours: Annotated[int, Field(ge=4, le=128, description="Palette size (24-32 suits most games).")] = 32,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Build ONE palette from the project's approved assets (tag good ones 'approved') and make
    every later asset snap to it."""
    return _reply(_A.lock_palette(colours, project), False)


# --------------------------------------------------------------------------- library

@studio_tool(title="List / search assets", read_only=True)
def list_assets(
    kind: Annotated[AssetKind | None, Field(description="Only this kind.")] = None,
    query: Annotated[str, Field(description="Words that must all appear in name/tags/prompt.")] = "",
    tag: Annotated[str | None, Field(description="Only assets with this tag.")] = None,
    collection: Annotated[str | None, Field(description="Only assets in this collection.")] = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """The project's assets, newest first: id, name, kind, tags, collections (characters: their
    state, e.g. draft waiting for an OK). Use the id in every other tool."""
    return _reply(_A.list_assets(kind, query, tag, collection, project, limit), False)


@studio_tool(title="Asset details + previews", read_only=True)
def get_asset(
    ref: Annotated[str, Field(description=REF_DOC)],
    images: Annotated[bool, Field(description="Attach the preview images.")] = True,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Everything about one asset: state, files (sheet, json, normal/glow maps, Godot .tres,
    models), frame size, animations, directions, variations, quality checks (subject check,
    drawn-key IoU/colour scores, empty frames) with warnings, and preview images (all
    directions, every animation facing the viewer, the variations). LOOK at the previews."""
    return _reply(_A.get_asset(ref, project), images)


@studio_tool(title="Delete an asset", destructive=True)
def delete_asset(
    ref: Annotated[str, Field(description=REF_DOC)],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Move an asset to <project>/.trash (recoverable: move the folder back)."""
    return _reply(_A.delete_asset(ref, project), False)


@studio_tool(title="Tag an asset / put it in collections")
def tag_asset(
    ref: Annotated[str, Field(description=REF_DOC)],
    add_tags: Annotated[list[str], Field(description="Tags to add ('approved' feeds lock_palette).")] = [],  # noqa: B006
    remove_tags: Annotated[list[str], Field(description="Tags to remove.")] = [],  # noqa: B006
    add_collections: Annotated[list[str], Field(description="Collections to add it to (export by collection).")] = [],  # noqa: B006
    remove_collections: Annotated[list[str], Field(description="Collections to take it out of.")] = [],  # noqa: B006
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Change an asset's tags and collections."""
    return _reply(_A.tag_asset(ref, add_tags, remove_tags, add_collections, remove_collections, project), False)


@studio_tool(title="List collections", read_only=True)
def list_collections(project: Annotated[str | None, Field(description=PROJECT_DOC)] = None) -> ToolOutput:
    """Collections and the asset ids in each."""
    return _reply(_A.collections(project), False)


# --------------------------------------------------------------------------- make

@studio_tool(title="What can be made (kinds, settings, GPU minutes)", read_only=True)
def what_can_be_made() -> ToolOutput:
    """Every kind `make` knows with its editable settings (choices) and the measured GPU minutes
    table (RTX 4070 laptop). CPU-only kinds take seconds."""
    return _reply(_A.kinds(), False)


@studio_tool(title="Plan (dry run): what would be made + cost", read_only=True)
def plan_asset(
    description: Annotated[str, Field(description="Plain words, e.g. 'a dwarf warrior with a big axe', "
                                                  "'rusty iron sword', 'cobblestone road tiles'.")],
    kind: Annotated[Kind | None, Field(description="Force the kind (null = the router decides).")] = None,
    params: Annotated[dict[str, Any] | None, Field(description="Settings to override (see fields).")] = None,
    name: Annotated[str | None, Field(description="Asset name (null = from the words).")] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """What `make` would do, without doing it: kind, name, GPU prompt, settings, a one-line
    summary, GPU/CPU minutes (characters: plus the minutes after the front is OK'd), other kinds
    the words could be, and the editable fields."""
    return _reply(_A.plan(description, kind, params, name, project), False)


@studio_tool(title="Make anything from words")
def make(
    description: Annotated[str, Field(description="What to make, plain words. The router picks the "
                                                  "kind (character, item, prop/building, tileset, effect, "
                                                  "portrait, card, ui kit, map, weather, water, background).")],
    kind: Annotated[Kind | None, Field(description="Force the kind (null = the router decides).")] = None,
    params: Annotated[dict[str, Any] | None, Field(description="Settings to override (plan_asset lists them).")] = None,
    name: Annotated[str | None, Field(description="Asset name (null = from the words).")] = None,
    dry_run: Annotated[bool, Field(description="Only plan (same as plan_asset).")] = False,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Make an asset in the project's house style, like the app's Make tab. GPU work runs as a
    background job (one at a time, queued, paced for heat): the reply has the job id, the plan
    and minutes; poll job_status. When done the result is the asset summary + previews +
    quality checks. Characters stop after the FRONT drawing for your review (character_review ->
    character_approve / character_redraw_front / character_change_words)."""
    return _reply(_A.make(description, kind, params, name, dry_run, wait_s, project))


def _kind_tool(kind: str, description: str, name, wait_s, project, **params) -> ToolOutput:
    return _reply(_A.make_kind(kind, description, name=name, wait_s=wait_s, project=project, **params))


@studio_tool(title="Make a character (drawn, front first)")
def make_character(
    description: Annotated[str, Field(description="e.g. 'a dwarf warrior with a big axe and a green cloak'.")],
    body: Annotated[Literal["person", "animal"] | None, Field(description="null = from the words.")] = None,
    size: Annotated[Literal["small", "normal", "medium", "large"] | None, Field(
        description="People small/normal/large, animals small/medium/large.")] = None,
    role: Annotated[str | None, Field(description="Tactics class (knight, archer, mage...).")] = None,
    review: Annotated[bool, Field(description="Stop after the front drawing for review (recommended: "
                                              "~3.3 min instead of ~15).")] = True,
    drawn: Annotated[bool, Field(description="GPU-drawn sprites bent by the skeleton (false = 3D render).")] = True,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Prompt -> picture -> 3D model -> auto-rig -> all the project's clips -> drawn sprites in
    every direction. With review (default) it stops after the front drawing + idle: a draft.
    Then character_review and character_approve (~0.75 min/direction + bending)."""
    return _kind_tool("character", description, name, wait_s, project, body=body, size=size,
                      role=role, review=review, drawn=drawn)


@studio_tool(title="Make an item (icon variations + world sprite)")
def make_item(
    description: Annotated[str, Field(description="e.g. 'rusty iron sword', 'healing potion'.")],
    category: Annotated[Literal["weapon", "armour", "potion", "food", "key", "misc"] | None, Field(
        description="null = from the words.")] = None,
    size: Annotated[Literal["tiny", "small", "medium", "large"] | None, Field(description="Icon size.")] = None,
    world: Annotated[bool | None, Field(description="Also a world (ground) sprite.")] = None,
    variations: Annotated[int | None, Field(ge=1, le=6, description="Icon variations to pick from.")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Inventory item: SDXL icon variations (best first by the one-subject check) and a world
    sprite. pick_variation picks another icon."""
    return _kind_tool("item", description, name, wait_s, project, category=category, size=size,
                      world=world, variations=variations)


@studio_tool(title="Make a prop or building")
def make_prop(
    description: Annotated[str, Field(description="e.g. 'wooden barrel', 'thatched cottage', 'stone well'.")],
    height_people: Annotated[float | None, Field(ge=0.2, le=8, description="Height in people (null = guessed).")] = None,
    directions: Annotated[Literal[1, 2, 4] | None, Field(description="Sides drawn (1 = one view).")] = None,
    look: Annotated[Literal["drawn", "3d"] | None, Field(description="drawn pixel art (default) or 3D render.")] = None,
    variations: Annotated[int | None, Field(ge=1, le=6, description="Drawings to pick from (1 side).")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Prop / building drawn as pixel art at the project's camera and scale (normal + glow maps
    so lights work)."""
    return _kind_tool("prop", description, name, wait_s, project, height_people=height_people,
                      directions=directions, look=look, variations=variations)


@studio_tool(title="Make a tileset (terrain + words)")
def make_terrain_tiles(
    description: Annotated[str, Field(description="e.g. 'cobblestone road tiles', 'mossy forest floor'.")],
    terrain: Annotated[str | None, Field(
        description="grass, dirt, sand, snow, stone, road, water, cliff, fence, wall (null = from the words).")] = None,
    count: Annotated[int | None, Field(ge=1, le=6, description="Textures to make (best kept first).")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Seamless terrain tiles at the project's tile size: ground = 16 corner tiles, water/cliff =
    47 blob, fence/wall = 16 line tiles, with variants; Godot TileSet .tres + Tiled .tsx. The first
    texture is kept; pick_variation(job=..., index=n) rebuilds from another. (make_tileset cuts an
    existing picture into tiles instead.)"""
    return _kind_tool("tileset", description, name, wait_s, project, terrain=terrain, count=count)


@studio_tool(title="Make a spell / hit effect")
def make_effect(
    description: Annotated[str, Field(description="e.g. 'ice nova', 'fireball', 'sword slash', 'heal'.")],
    effect: Annotated[str | None, Field(
        description="fire_burst fireball ice_shard frost_nova lightning_bolt chain_lightning sword_combo "
                    "hit_spark holy_light poison_cloud dark_magic big_explosion buff_aura level_up_burst "
                    "or a classic kind (what_can_be_made lists all). null = from the words.")] = None,
    theme: Annotated[str | None, Field(description="Colour theme (fire, ice, lightning, holy, nature, dark...).")] = None,
    size: Annotated[Literal["", "small", "medium", "large", "huge"] | None, Field(description="Size.")] = None,
    paint: Annotated[bool | None, Field(description="Hand-paint pass on the GPU (~0.6 min).")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Procedural many-frame effect (no GPU, seconds): sheet + glow map + per-frame timing, Godot
    additive material + demo scene."""
    return _kind_tool("effect", description, name, wait_s, project, effect=effect, theme=theme,
                      size=size, paint=paint)


@studio_tool(title="Make weather / water / a background")
def make_atmosphere(
    description: Annotated[str, Field(description="e.g. 'heavy rain at night', 'animated lagoon water', "
                                                  "'sunset mountains background', 'misty bamboo valley background'.")],
    kind: Annotated[Literal["weather", "water", "background"], Field(description="Which maker.")],
    weather: Annotated[str | None, Field(description="weather: rain snow fog god_rays leaves petals embers "
                                                     "fireflies dust sandstorm lightning cloud_shadows.")] = None,
    theme: Annotated[str | None, Field(description="water: ocean, lagoon, river, swamp, lava...")] = None,
    scene: Annotated[str | None, Field(description="background: mountains, forest, ... (procedural).")] = None,
    tod: Annotated[Literal["", "dawn", "noon", "sunset", "night"] | None, Field(description="Time of day.")] = None,
    intensity: Annotated[float | None, Field(ge=0.3, le=2, description="weather strength.")] = None,
    words: Annotated[str | None, Field(description="background: words for an SDXL-drawn scene (GPU); "
                                                   "empty = drawn by code.")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Looping, tiling weather overlays + particles; animated water with 47 shore tiles; parallax
    background layers (Godot Parallax2D). CPU, seconds (a background from words uses the GPU)."""
    return _kind_tool(kind, description, name, wait_s, project, weather=weather, theme=theme,
                      scene=scene, tod=tod, intensity=intensity, words=words)


@studio_tool(title="Make portraits (+ expressions)")
def make_portrait(
    description: Annotated[str, Field(description="Who, e.g. 'goblin shaman' or a library character's name, "
                                                  "plus expressions: 'Knight portrait, angry and happy'.")],
    character: Annotated[str | None, Field(description="Library character name to paint (null = from the words).")] = None,
    expressions: Annotated[str | None, Field(description="Comma list: neutral, happy, angry, sad, surprised, "
                                                         "determined, smug, hurt...")] = None,
    style: Annotated[Literal["painted", "pixel"] | None, Field(description="Shown as.")] = None,
    name: Annotated[str | None, Field(description="Name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Dialogue busts: a painted neutral portrait, then each expression by face-only inpaint
    (everything else identical), plus pixel versions and a 9-slice frame."""
    return _kind_tool("portrait", description, name, wait_s, project, character=character,
                      expressions=expressions, style=style)


@studio_tool(title="Make a card")
def make_card(
    description: Annotated[str, Field(description="e.g. 'fire dragon card, legendary' or 'Knight card'.")],
    type: Annotated[Literal["Unit", "Spell", "Item", "Relic"] | None, Field(description="Card type.")] = None,
    faction: Annotated[Literal["neutral", "fire", "nature", "water", "shadow", "holy"] | None, Field()] = None,
    rarity: Annotated[Literal["common", "rare", "epic", "legendary"] | None, Field()] = None,
    art_style: Annotated[Literal["painted", "pixel"] | None, Field()] = None,
    variations: Annotated[int | None, Field(ge=1, le=6, description="Art variations.")] = None,
    name: Annotated[str | None, Field(description="Card name (null = from the words).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A card: art (from a library asset named in the words, or drawn from the words), frame by
    rarity/faction, text. Godot Card.tscn / Unity CardView.cs on export."""
    return _kind_tool("card", description, name, wait_s, project, type=type, faction=faction,
                      rarity=rarity, art_style=art_style, variations=variations)


@studio_tool(title="Make a UI kit")
def make_ui_kit(
    theme: Annotated[Literal["wood", "stone", "parchment", "metal", "crystal", "sci-fi"], Field(description="Theme.")] = "wood",
    scale: Annotated[int, Field(ge=0, le=4, description="Whole-number scale (0 = from the tile size).")] = 0,
    name: Annotated[str | None, Field(description="Name (null = '<Theme> UI kit').")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """9-slice panels, window, dialogue box, buttons x4, tabs, slots, bars, slider, scrollbar,
    icons, cursors and a pixel font in the house style (no GPU). Godot theme.tres + demo scene /
    Unity sprites with borders on export."""
    return _kind_tool("ui", f"{theme} UI kit", name, wait_s, project, theme=theme, scale=scale)


@studio_tool(title="Make a map from a template")
def make_map(
    template: Annotated[Literal["flat", "harbour", "forest", "castle", "snowfield", "village"], Field()] = "flat",
    cols: Annotated[int | None, Field(ge=8, le=64)] = None,
    rows: Annotated[int | None, Field(ge=8, le=64)] = None,
    seed: Annotated[int, Field(ge=0, le=9999)] = 0,
    name: Annotated[str | None, Field(description="Name (null = '<Template> map').")] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A tactics map (heights, terrain, props, team spawns) drawn with the project's tilesets.
    Edit it with map_edit (terrain_grid shows the cells)."""
    return _kind_tool("map", f"{template} map", name, None, project, template=template, cols=cols,
                      rows=rows, seed=seed)


@studio_tool(title="Pick a variation")
def pick_variation(
    index: Annotated[int, Field(ge=1, le=12, description="1-based variation number (see the previews).")],
    ref: Annotated[str | None, Field(description=REF_DOC + " Items: icon; drawn props: drawing; cards: art.")] = None,
    job: Annotated[str | None, Field(description="Tilesets: the job id that made it.")] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Use another variation (no GPU): an item's icon, a drawn prop's drawing, a card's art, or a
    tileset's texture (rebuilt from it; give the job)."""
    return _reply(_A.pick_variation(ref, index, project, job))


# --------------------------------------------------------------------------- characters: front first

@studio_tool(title="Review a character (front drawing / sheet)", read_only=True)
def character_review(
    ref: Annotated[str, Field(description=REF_DOC)],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A draft character's front drawing + its idle (and earlier tries), the picture it was made
    from, the minutes left after an OK; or a finished character's directions + animations. Judge:
    does it match the words, read at game size, have the held things, no extra limbs? Then
    character_approve, character_redraw_front, character_change_words or character_pick_try."""
    return _reply(_A.review(ref, project))


@studio_tool(title="Approve: draw every direction")
def character_approve(
    ref: Annotated[str, Field(description=REF_DOC)],
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """The front looks good: draw the other directions paired to it (the front is reused byte
    for byte) and bend every animation. GPU ~0.75 min per direction + ~2 min bending (16
    directions ~13 min). Cancel keeps the drawings; approving again carries on."""
    return _reply(_A.approve(ref, wait_s, project))


@studio_tool(title="Redraw the front (new seed)")
def character_redraw_front(
    ref: Annotated[str, Field(description=REF_DOC)],
    seed: Annotated[int | None, Field(description="A specific seed (null = a new one).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Draw the front again with a new seed (~40 s GPU); earlier tries stay pickable."""
    return _reply(_A.redraw_front(ref, seed, wait_s, project))


@studio_tool(title="Change a character's words")
def character_change_words(
    ref: Annotated[str, Field(description=REF_DOC)],
    words: Annotated[str, Field(description="The new description.")],
    new_model: Annotated[bool, Field(description="Also a new picture + 3D shape (when the SHAPE is wrong: "
                                                 "missing weapon, wrong body; ~3 min).")] = False,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Redraw the front from new words (same seed, so only the words change); new_model for a
    new shape too."""
    return _reply(_A.change_words(ref, words, new_model, wait_s, project))


@studio_tool(title="Pick an earlier try of the front")
def character_pick_try(
    ref: Annotated[str, Field(description=REF_DOC)],
    n: Annotated[int, Field(ge=1, description="Try number (character_review lists them).")],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Use an earlier front drawing again (no GPU)."""
    return _reply(_A.pick_try(ref, n, project))


@studio_tool(title="Finish an interrupted drawing")
def character_finish(
    ref: Annotated[str, Field(description=REF_DOC)],
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A character whose drawing stopped (GPU crash: 3D-rendered sheets for now): draw only the
    missing directions, then bend every animation."""
    return _reply(_A.finish(ref, wait_s, project))


@studio_tool(title="Redraw some directions of a finished character")
def character_redraw(
    ref: Annotated[str, Field(description=REF_DOC)],
    directions: Annotated[list[str], Field(description="e.g. ['NE', 'NNE'] (empty = every direction).")],
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """New seeds for those directions (the rest stay identical, same palette); quality.drawn.worst
    in get_asset names weak ones."""
    return _reply(_A.redraw_directions(ref, directions, wait_s, project))


# --------------------------------------------------------------------------- animation

@studio_tool(title="List animations that can be added", read_only=True)
def list_motions(body: Annotated[Literal["person", "animal"], Field()] = "person") -> ToolOutput:
    """Built-in clips, clips add_animation can add (cast shoot block thrust jump wave kneel
    victory; animals bark howl jump eat) and the mocap library (downloaded ones usable)."""
    return _reply(_A.motions(body), False)


@studio_tool(title="Add animations to a character")
def add_animation(
    ref: Annotated[str, Field(description=REF_DOC)],
    names: Annotated[list[str], Field(description="Clips, e.g. ['cast', 'jump'] or ['attack=shoot'] with replace.")],
    replace: Annotated[bool, Field(description="Replace clips of the same name.")] = False,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """New clips on the rig; old frames stay identical (drawn characters bend the same drawings:
    no GPU). ~20-70 s CPU."""
    return _reply(_A.add_animation(ref, names, replace, wait_s, project))


@studio_tool(title="Add mocap clips to a character")
def add_mocap(
    ref: Annotated[str, Field(description=REF_DOC)],
    names: Annotated[list[str], Field(description="Mocap library names (list_motions) or .bvh paths.")],
    replace: Annotated[bool, Field(description="A clip with an existing name replaces it.")] = True,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Motion capture retargeted onto the character's rig (people only)."""
    return _reply(_A.add_mocap(ref, names, replace, wait_s, project))


@studio_tool(title="Capes / hair / tails swing (springs)")
def add_springs(
    ref: Annotated[str, Field(description=REF_DOC)],
    stiffness: Annotated[float | None, Field(ge=0, le=1)] = None,
    damping: Annotated[float | None, Field(ge=0, le=1)] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Spring bones baked into every clip, sheet rendered again."""
    opts = {k: v for k, v in {"stiffness": stiffness, "damping": damping}.items() if v is not None}
    return _reply(_A.add_springs(ref, opts, wait_s, project))


@studio_tool(title="Re-skin a character (new colours)")
def reskin_character(
    ref: Annotated[str, Field(description=REF_DOC)],
    name: Annotated[str, Field(description="The new character's name, e.g. 'Blue Knight'.")],
    colour: Annotated[str | None, Field(description="New colour for the main colour family: blue, red, green, "
                                                    "purple... or #rrggbb.")] = None,
    hue: Annotated[float | None, Field(ge=0, le=360, description="Or turn every hue by degrees.")] = None,
    source_colour: Annotated[str | None, Field(description="#rrggbb of the colour to change (null = the main one).")] = None,
    saturation: Annotated[float, Field(ge=0, le=3)] = 1.0,
    lightness: Annotated[float, Field(ge=0.2, le=3)] = 1.0,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A new library character with its sheet recoloured (pixels, frames, normal maps kept). No GPU."""
    return _reply(_A.reskin(ref, name, hue, colour, source_colour, saturation, lightness, project))


@studio_tool(title="Restyle the project's assets")
def restyle_project(
    dry_run: Annotated[bool, Field(description="Only say what would be redone.")] = True,
    kinds: Annotated[list[Literal["character", "item", "prop"]] | None, Field()] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Redo characters/items/props in the CURRENT house style (drawn characters are drawn again
    on the GPU: ~10 min each)."""
    return _reply(_A.restyle(kinds, dry_run, wait_s, project), False)


@studio_tool(title="Game-ready 3D model")
def game_model(
    ref: Annotated[str, Field(description=REF_DOC)],
    look: Annotated[Literal["smooth", "pixel"] | None, Field(description="smooth (1024 textures) or pixel "
                                                                         "(256, palette, nearest). null = project's.")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """LOD0-2 .glb with UVs, baked colour + normal maps, skin + every clip (characters), for 3D
    games. CPU ~1.5-2.5 min; drawn props/items first get a 3D shape on the GPU (~1.5 min).
    Export with export_to_game(models=true)."""
    return _reply(_A.game_model(ref, look, wait_s, project), True)


# --------------------------------------------------------------------------- maps

@studio_tool(title="Map cells (terrain / height grid)", read_only=True)
def map_info(
    ref: Annotated[str, Field(description=REF_DOC)],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """A map as text: terrain rows (one letter per cell, legend), height rows (levels 0-9),
    props, spawns; plus its preview. Row 0 is the top (north), col 0 the left."""
    return _reply(_A.map_info(ref, project))


@studio_tool(title="Edit a map")
def map_edit(
    ref: Annotated[str, Field(description=REF_DOC)],
    edits: Annotated[list[dict[str, Any]], Field(
        description="In order: {op: paint, col, row, terrain, brush} | {op: rect, col, row, col2, row2, "
                    "terrain and/or level} | {op: fill, col, row, terrain} | {op: raise|lower, col, row, "
                    "amount, brush} | {op: height, col, row, level} | {op: prop, col, row, kind} | "
                    "{op: erase_prop, col, row} | {op: spawn, col, row, team 0|1} | {op: clear_spawn, col, "
                    "row} | {op: resize, cols, rows}.")],
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Apply edits and save the map (preview re-rendered with the project's tilesets)."""
    return _reply(_A.map_edit(ref, edits, project))


# --------------------------------------------------------------------------- export into a game

@studio_tool(title="Export into a Godot / Unity project")
def export_to_game(
    project_path: Annotated[str, Field(description="The game project folder (has project.godot / Assets).")],
    engine: Annotated[Literal["godot", "unity"] | None, Field(description="null = the house style's engine.")] = None,
    refs: Annotated[list[str] | None, Field(description="Asset ids (null with no collection/kinds = every asset).")] = None,
    collection: Annotated[str | None, Field(description="Export a collection.")] = None,
    kinds: Annotated[list[AssetKind] | None, Field(description="Export every asset of these kinds.")] = None,
    subdir: Annotated[str, Field(description="Folder inside the project: Godot res://<subdir> (default art), "
                                             "Unity Assets/<subdir> (default Assets/PixelPipeline).")] = "",
    force: Annotated[bool, Field(description="Re-export unchanged assets too.")] = False,
    models: Annotated[bool, Field(description="Also the game-ready 3D models (made with game_model).")] = False,
    build_models: Annotated[bool, Field(description="Build missing 3D models during the export (slow).")] = False,
    verify: Annotated[bool, Field(description="Godot: import headless and load every exported resource.")] = False,
    preset: Annotated[str | None, Field(description="Sprite preset override (godot-2d, godot-hd2d, unity...).")] = None,
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
    project: Annotated[str | None, Field(description=PROJECT_DOC)] = None,
) -> ToolOutput:
    """Write assets into an existing game project with the engine's import settings (Godot
    .import: lossless, no mipmaps; Unity .meta: point filter, sliced sprites) and update
    <subdir>/pixel_pipeline_manifest.json: asset id -> files (res:// paths for Godot), frame
    size, animations (directions, frames, ms, loop), anchor (feet), ppu, tile, normal/glow maps;
    by_name maps engine-safe names to ids, so game code loads assets by name. Incremental:
    unchanged assets are skipped. Drafts are skipped (approve them first)."""
    return _reply(_A.export(refs, engine, project_path, subdir, collection, kinds, force, models,
                            build_models, verify, preset, wait_s, project), False)


@studio_tool(title="Verify the Godot import (headless)")
def verify_godot_import(
    project_path: Annotated[str, Field(description="The Godot project folder.")],
    subdir: Annotated[str, Field(description="Where the manifest is (default art).")] = "",
    wait_s: Annotated[float | None, Field(ge=0, le=3600, description=WAIT_DOC)] = None,
) -> ToolOutput:
    """Run Godot --headless --import, then load every resource in the manifest (SpriteFrames,
    textures, TileSets, themes, scenes instantiated, models). Needs GODOT_BIN or Godot on PATH."""
    return _reply(_A.verify_godot(project_path, subdir, wait_s), False)


# --------------------------------------------------------------------------- jobs

@studio_tool(title="Job status", read_only=True)
def job_status(
    job: Annotated[str, Field(description="Job id.")],
    wait_s: Annotated[float, Field(ge=0, le=600, description="Wait up to this long for it to finish first.")] = 0,
    images: Annotated[bool, Field(description="Attach the result's previews when done.")] = True,
) -> ToolOutput:
    """percent, stage, what it's doing, eta, minutes so far; when done the result (asset summary,
    previews, quality checks; attached as images); failed: the error and the log tail."""
    d = _A.job_status(job, wait_s)
    extra = [d["latest_drawing"]] if d.get("latest_drawing") else []
    return _reply(d, images, extra)


@studio_tool(title="List jobs", read_only=True)
def list_jobs(
    status: Annotated[Literal["queued", "running", "done", "failed", "cancelled", "interrupted"] | None, Field()] = None,
    project: Annotated[str | None, Field(description="Only this project's.")] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 20,
) -> ToolOutput:
    """Recent jobs (running/queued first) and who holds the GPU lock. 'interrupted' = its process
    stopped (crash/restart): resume_jobs."""
    return _reply(_A.list_jobs(status, project, limit), False)


@studio_tool(title="Cancel a job")
def cancel_job(job: Annotated[str, Field(description="Job id.")]) -> ToolOutput:
    """Queued: dropped. Running: stops at its next step (finished drawings/variations are kept;
    a character can be approved/finished again later)."""
    return _reply(_A.cancel_job(job), False)


@studio_tool(title="Resume interrupted jobs")
def resume_jobs(
    jobs: Annotated[list[str] | None, Field(description="Job ids (null = every interrupted job). Failed or "
                                                        "cancelled ones run again when named.")] = None,
) -> ToolOutput:
    """Queue interrupted jobs again (nothing restarts by itself after a crash). Drawn characters
    carry on from their kept drawings."""
    return _reply(_A.resume_jobs(jobs), False)


def main() -> None:
    """Serve over stdio (what Claude Code / Claude Desktop launch)."""
    server.run("stdio")


if __name__ == "__main__":
    main()
