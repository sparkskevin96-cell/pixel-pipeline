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
from .animate import IDLE_MS, IDLE_STYLES, idle_frames
from .export import load_rgba, load_sprite, write_animation, write_tileset
from .godot import write_tileset as write_godot_tileset
from .palettes import BUILTIN, rgb_to_hex
from .pipeline import convert_batch
from .quantize import DITHER_MODES
from .settings import PRESETS, SUBJECTS, Settings, settings_for
from .systems import SYSTEM_NAMES, SYSTEMS
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
SYSTEM_DOC = ("Retro console look: " + "; ".join(
    f"{k} = {v.name} ({v.sprite_colors} colours/sprite)" for k, v in SYSTEMS.items())
    + ". Sets the console's palette/colour depth, colours per sprite, typical sizes and outline "
      "style (explicit width/height/colors still win). off = free colours.")
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
        f"({Path.cwd()})."
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
        except CutoutUnavailable as e:
            raise ToolError(f"{e}. Pass remove_background=false for photos that already have "
                            "a transparent background, or for scenes.") from e
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


def _output_png(raw: str) -> Path:
    p = _path(raw)
    if p.suffix.lower() != ".png":
        raise ToolError(f"Output must be a .png path, got {p}")
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
           "system", "subject"}


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
                "remove_background": remove_background, "system": system}
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
                "colors": colors, "remove_background": remove_background, "system": system}
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
    sprite = sprite_from_rows(rows, palette, system, subject)
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


def main() -> None:
    """Serve over stdio (what Claude Code / Claude Desktop launch)."""
    server.run("stdio")


if __name__ == "__main__":
    main()
