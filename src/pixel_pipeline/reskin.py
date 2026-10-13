"""Re-skins: a library character in new colours, as a new library character.

    reskin(knight, "Blue Knight", target_colour="blue")   # the dominant colour (red cape) -> blue
    reskin(knight, "Dusk Knight", hue=200)                # every hue turned 200 degrees
    reskin(knight, "Pale Knight", saturation=0.4, lightness=1.1)
    reskin(knight, "Gold Trim", palette_map={"#c02020": "#d0a020"})   # exact colours
    make_team_variants(knight, ["blue", "red", "green"])  # "Knight Blue", "Knight Red", ...

Only colours change: alpha and every pixel's position stay exactly as they were, the normal map
is copied unchanged, the .aseprite is written again from the recoloured sheet (same frames,
durations and tags), the thumbnail is remade. The 3D model is copied as it was (not
recoloured), so `library.rebuild` of a re-skin renders the ORIGINAL colours again.

The colour family is found like team colours (teams.py): the saturated hue covering the most
pixels, recoloured in LCh keeping lightness, so the shading survives.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .teams import TEAMS, _lch, _rgb, recolour, team_family, team_palettes

COLOUR_NAMES = {
    "blue": "2f5fd8", "red": "d23a32", "green": "2fa04a", "purple": "a64fd0",
    "yellow": "e8c430", "orange": "e8802a", "pink": "e060a8", "cyan": "30b8d0",
    "teal": "20a090", "brown": "8a5a30", "white": "f0f0f0", "black": "202020",
    "grey": "808080", "gray": "808080", "gold": "d0a020", "silver": "b8c0c8",
    **TEAMS,
}


def colour_hex(colour) -> str:
    """'blue', 'enemy', '#2f5fd8', '2f5fd8' or (r, g, b) -> 'rrggbb'."""
    if isinstance(colour, (tuple, list, np.ndarray)):
        r, g, b = (int(c) for c in list(colour)[:3])
        return f"{r:02x}{g:02x}{b:02x}"
    s = str(colour).strip().lower()
    if s in COLOUR_NAMES:
        return COLOUR_NAMES[s]
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"Not a colour: {colour!r} (use a name like blue or a hex code)")
    return s


def _hex_rgb(h: str) -> np.ndarray:
    h = colour_hex(h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def image_colours(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every colour of a visible pixel (alpha > 0) and how many pixels use it."""
    px = rgba.reshape(-1, 4)
    px = px[px[:, 3] > 0][:, :3]
    if not len(px):
        return np.zeros((0, 3), np.uint8), np.zeros(0, int)
    return np.unique(px, axis=0, return_counts=True)


def _scale(lch: np.ndarray, saturation: float, lightness: float) -> np.ndarray:
    out = lch.copy()
    out[:, 1] = np.clip(out[:, 1] * saturation, 0, 150)
    out[:, 0] = np.clip(out[:, 0] * lightness, 0, 100)
    return out


def reskin_palette(colours: np.ndarray, counts: np.ndarray, *, hue: float | None = None,
                   palette_map: dict | None = None, target_colour=None, source_colour=None,
                   saturation: float = 1.0, lightness: float = 1.0) -> np.ndarray:
    """New colours row-aligned with `colours`.
    target_colour: the dominant colour family (or source_colour's family) -> that colour, then
    saturation/lightness on that family only. Else hue (degrees) + saturation/lightness on
    every colour. palette_map ({old: new}, exact colours) is applied last."""
    colours = np.asarray(colours, np.uint8).reshape(-1, 3)
    new = colours.copy()
    if len(colours) and target_colour is not None:
        family = team_family(colours, counts, colour_hex(source_colour) if source_colour else None)
        if not family.any():
            raise ValueError("This character has no strong colour to change; "
                             "use a hue shift instead.")
        new = team_palettes(colours, family, {"t": colour_hex(target_colour)}, spent=False)["t"]
        if saturation != 1.0 or lightness != 1.0:
            new[family] = _rgb(_scale(_lch(new[family]), saturation, lightness))
    elif len(colours) and (hue or saturation != 1.0 or lightness != 1.0):
        lch = _lch(colours)
        lch[:, 2] = (lch[:, 2] + float(hue or 0)) % 360
        new = _rgb(_scale(lch, saturation, lightness))
    for old, to in (palette_map or {}).items():
        hit = (colours == _hex_rgb(old)).all(1)
        new[hit] = _hex_rgb(to)
    return new


def recolour_image(rgba: np.ndarray, colours: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Same pixels and alpha, colours[i] -> new[i]."""
    return recolour(rgba, colours, new) if len(colours) else rgba.copy()


# --------------------------------------------------------------------------- characters

def _character(src):
    """A library Character, an assets.Asset or a character folder -> library.Character."""
    from .library import Character
    if isinstance(src, Character):
        return src
    folder = Path(src.folder if hasattr(src, "folder") else src)
    meta = folder / "character.json"
    if not meta.exists():
        raise ValueError(f"{folder} is not a library character (no character.json).")
    d = json.loads(meta.read_text(encoding="utf-8"))
    return Character(d.get("name", folder.name), d.get("kind", "person"), d.get("size", "normal"),
                     d.get("role", ""), d.get("created", 0.0), folder)


def _options(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None and not (k in ("saturation", "lightness")
                                                                  and v == 1.0)}


def preview(src, **kw) -> np.ndarray:
    """The character's thumbnail recoloured as reskin(**kw) would (nothing is saved)."""
    ch = _character(src)
    colours, counts = image_colours(np.asarray(Image.open(ch.sheet).convert("RGBA")))
    new = reskin_palette(colours, counts, **kw)
    thumb = ch.thumb if ch.thumb.exists() else ch.sheet
    return recolour_image(np.asarray(Image.open(thumb).convert("RGBA")), colours, new)


def _rewrite_aseprite(sheet_png: Path, rgba: np.ndarray, out: Path) -> Path:
    from .aseprite import write_aseprite
    from .exports import _frames, _tags
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    frames = _frames(meta)
    images = [rgba[f.y:f.y + f.h, f.x:f.x + f.w] for f in frames]
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    images = [np.pad(i, ((0, h - i.shape[0]), (0, w - i.shape[1]), (0, 0))) for i in images]
    cols, _ = image_colours(rgba)
    write_aseprite(out, images, cols if len(cols) <= 255 else None,
                   durations=[f.duration for f in frames], tags=_tags(meta, frames))
    return out


def reskin(src, name: str, *, hue: float | None = None, palette_map: dict | None = None,
           target_colour=None, source_colour=None, saturation: float = 1.0,
           lightness: float = 1.0, folder: Path | None = None):
    """A new library character: `src`'s folder copied with its sheet recoloured (see module
    doc). folder: the library folder to put it in (default: next to `src`). Its asset.json
    extra gets "reskin_of" (the source's name and folder) and "reskin" (the options)."""
    from .library import Character, _free_folder, make_thumb
    ch = _character(src)
    if not ch.sheet.exists():
        raise ValueError(f"{ch.name} has no sprite sheet to re-skin.")
    sheet = np.asarray(Image.open(ch.sheet).convert("RGBA"))
    colours, counts = image_colours(sheet)
    new = reskin_palette(colours, counts, hue=hue, palette_map=palette_map,
                         target_colour=target_colour, source_colour=source_colour,
                         saturation=saturation, lightness=lightness)
    base = Path(folder) if folder else ch.folder.parent
    dst = _free_folder(name, base)
    shutil.copytree(ch.folder, dst, ignore=shutil.ignore_patterns("thumb.png", "sheet.png",
                                                                  "sheet.aseprite"),
                    dirs_exist_ok=True)
    out_sheet = recolour_image(sheet, colours, new)
    Image.fromarray(out_sheet, "RGBA").save(dst / "sheet.png")
    if (ch.folder / "sheet.aseprite").exists():
        _rewrite_aseprite(dst / "sheet.png", out_sheet, dst / "sheet.aseprite")
    make_thumb(dst / "sheet.png", dst / "thumb.png")
    new_ch = Character(name, ch.kind, ch.size, ch.role, time.time(), dst)
    new_ch.save()
    meta_path = dst / "asset.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("extra", {})["reskin_of"] = {"name": ch.name, "folder": ch.folder.name}
    meta["extra"]["reskin"] = _options(
        hue=hue, target_colour=colour_hex(target_colour) if target_colour is not None else None,
        source_colour=colour_hex(source_colour) if source_colour else None,
        palette_map=palette_map, saturation=saturation, lightness=lightness)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return new_ch


def make_team_variants(src, colours=("blue", "red", "green"), *, source_colour=None,
                       folder: Path | None = None) -> list:
    """One re-skin per colour, named '<name> <Colour>' (hex codes: '<name> #rrggbb')."""
    ch = _character(src)
    out = []
    for c in colours:
        label = str(c).strip()
        label = label.title() if label.lower() in COLOUR_NAMES else "#" + colour_hex(c)
        out.append(reskin(ch, f"{ch.name} {label}", target_colour=c, source_colour=source_colour,
                          folder=folder))
    return out
