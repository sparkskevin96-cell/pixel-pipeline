"""Your characters, in one place: everything made in the Characters tab lands here, at one scale,
ready for the Game tab (and for Godot or Aseprite by hand).

    ~/Pixel Pipeline/Characters/<slug>/
        character.json   name, kind (person / animal), size, role (stats class), when
        model.glb        the coloured 3D model
        picture.png      the picture it was made from (if any)
        rigged.glb       with a skeleton and animations
        sheet.png/.json  every animation x 16 directions (+ sheet_n.png normal map,
                         sheet.aseprite, sheet.tres + sheet_facing.gd for Godot)
        thumb.png        a small portrait for lists

One scale for everyone: the frame is wide enough that the HEIGHT sets the scale, so a spear or
a bow never shrinks its owner, and animals are sized against a person (SIZES). Set
PIXEL_LIBRARY to keep the library somewhere else.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .diorama import CLASSES

# frame (width, height) and colours per kind and size; people 64 px tall at "normal"
SIZES = {
    "person": {"small": (80, 52), "normal": (96, 64), "large": (112, 76)},
    "animal": {"small": (64, 28), "medium": (72, 34), "large": (88, 44)},
}
DEFAULT_SIZE = {"person": "normal", "animal": "medium"}
COLOURS = {"person": 24, "animal": 16}
SHEET_FILES = (".png", ".json", "_n.png", ".aseprite", ".tres", "_facing.gd")


def root() -> Path:
    """The current project's characters (assets.py), or $PIXEL_LIBRARY."""
    env = os.environ.get("PIXEL_LIBRARY")
    if env:
        return Path(env)
    from .assets import current_project
    return current_project().kind_folder("character")


def style():
    from .assets import Style, current_project
    return Style() if os.environ.get("PIXEL_LIBRARY") else current_project().style


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "character"


@dataclass
class Character:
    name: str
    kind: str = "person"                 # person / animal
    size: str = "normal"
    role: str = ""                       # stats class (diorama.CLASSES); "" = by name or default
    created: float = 0.0
    folder: Path = field(default=Path("."), compare=False, repr=False)

    @property
    def model(self) -> Path:
        return self.folder / "model.glb"

    @property
    def rigged(self) -> Path:
        return self.folder / "rigged.glb"

    @property
    def picture(self) -> Path:
        return self.folder / "picture.png"

    @property
    def sheet(self) -> Path:
        return self.folder / "sheet.png"

    @property
    def thumb(self) -> Path:
        return self.folder / "thumb.png"

    @property
    def stats_class(self) -> str:
        if self.role in CLASSES:
            return self.role
        if slug(self.name) in CLASSES:
            return slug(self.name)
        return "dog" if self.kind == "animal" else "swordsman"

    def stats(self) -> tuple:
        return CLASSES[self.stats_class]

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "kind": self.kind, "size": self.size, "role": self.role,
                "created": self.created or time.time()}
        p = self.folder / "character.json"
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
        # the asset store's record too (keeps tags/collections set elsewhere)
        meta = self.folder / "asset.json"
        old = {}
        if meta.exists():
            try:
                old = json.loads(meta.read_text(encoding="utf-8"))
            except ValueError:
                old = {}
        meta.write_text(json.dumps({
            "kind": "character", "name": self.name, "tags": old.get("tags", []),
            "collections": old.get("collections", []), "created": data["created"],
            "extra": {"body": self.kind, "size": self.size, "role": self.role}}, indent=1),
            encoding="utf-8")
        return p


def list_characters(folder: Path | None = None) -> list[Character]:
    """Every character in the library, newest first."""
    base = Path(folder) if folder else root()
    out = []
    for meta in sorted(base.glob("*/character.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(Character(d.get("name", meta.parent.name), d.get("kind", "person"),
                             d.get("size", "normal"), d.get("role", ""), d.get("created", 0.0),
                             meta.parent))
    return sorted(out, key=lambda c: -c.created)


def get(name: str, folder: Path | None = None) -> Character | None:
    for c in list_characters(folder):
        if slug(c.name) == slug(name):
            return c
    return None


def _free_folder(name: str, base: Path) -> Path:
    folder = base / slug(name)
    k = 2
    while (folder / "character.json").exists():
        folder = base / f"{slug(name)}-{k}"
        k += 1
    return folder


def frame_size(kind: str, size: str, st=None) -> tuple[int, int]:
    """Frame for a kind and size in the project's scale (SIZES are for a 64 px person)."""
    sizes = SIZES.get(kind, SIZES["person"])
    w, h = sizes.get(size, sizes[DEFAULT_SIZE.get(kind, "normal")])
    st = st or style()
    return st.scale(w), st.scale(h)


def make_thumb(sheet_png: Path, out: Path, px: int = 96) -> Path:
    """The idle frame facing the viewer (SE or S), enlarged crisp to about px."""
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    frames = {f["filename"]: f["frame"] for f in meta["frames"]}
    key = next((k for k in ("idle_SE 0", "idle_S 0") if k in frames), meta["frames"][0]["filename"])
    r = frames[key]
    im = Image.open(sheet_png).convert("RGBA").crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]))
    box = im.getbbox()
    if box:
        im = im.crop(box)
    k = (px - 6) / max(im.size)                  # fill the box; nearest keeps pixels crisp
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((px - im.width) // 2, max(0, px - im.height - 4)))
    canvas.save(out)
    return out


def render_sheets(rigged: Path, out_png: Path, kind: str, size: str, frames: int = 8,
                  progress=None) -> list[Path]:
    """All animations x 16 directions at the library scale, with normal maps."""
    from .anim3d import animated_turnaround, write_animated
    from .settings import settings_for
    st = style()
    w, h = frame_size(kind, size, st)
    colours = st.colours if kind == "person" else max(8, round(st.colours * 2 / 3))
    s = settings_for(kind if kind == "animal" else "person", width=w, height=h, colors=colours)
    if st.outline == "none":
        s.outline = False
    elif st.outline in ("selout", "solid"):
        s.outline_style = st.outline
    if st.palette != "auto":
        s.palette = st.palette
    (progress or (lambda t: None))(f"Rendering every animation from {st.directions} directions…")
    anim, _ = animated_turnaround([rigged], s, st.directions, frames, elevation=st.elevation,
                                  light=st.light, normals=True)
    return write_animated(anim, out_png, aseprite=True, godot=True, normals=True)


def add_character(name: str, model, *, kind: str = "person", size: str | None = None,
                  picture=None, role: str = "", folder: Path | None = None,
                  progress=None) -> Character:
    """A coloured 3D model -> a library character: skeleton + animations (people: idle walk run
    attack hurt death; animals: idle walk run sit happy lie_down death), sheets, thumbnail."""
    say = progress or (lambda t: None)
    size = size or DEFAULT_SIZE.get(kind, "normal")
    base = Path(folder) if folder else root()
    ch = Character(name, kind, size, role, time.time(), _free_folder(name, base))
    ch.folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, ch.model)
    if picture is not None and Path(picture).exists():
        shutil.copyfile(picture, ch.picture)
    say("Adding a skeleton and animations…")
    if kind == "animal":
        from .quadruped import auto_rig_quadruped
        auto_rig_quadruped(ch.model, ch.rigged)
    else:
        from .autorig import auto_rig
        auto_rig(ch.model, ch.rigged)
    render_sheets(ch.rigged, ch.sheet, kind, size, progress=say)
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


def import_sheet(name: str, sheet_png, *, kind: str = "person", size: str | None = None,
                 model=None, rigged=None, picture=None, role: str = "",
                 folder: Path | None = None) -> Character:
    """Put an existing sheet (+ its .json, _n.png, .aseprite, .tres, _facing.gd) in the library."""
    sheet_png = Path(sheet_png)
    base = Path(folder) if folder else root()
    ch = Character(name, kind, size or DEFAULT_SIZE.get(kind, "normal"), role, time.time(),
                   _free_folder(name, base))
    ch.folder.mkdir(parents=True, exist_ok=True)
    for ext in SHEET_FILES:
        src = sheet_png.with_name(sheet_png.stem + ext)
        if src.exists():
            dst = ch.folder / f"sheet{ext}"
            if ext in (".tres", "_facing.gd"):
                text = src.read_text(encoding="utf-8").replace(sheet_png.name, "sheet.png")
                dst.write_text(text, encoding="utf-8")
            elif ext == ".json":
                meta = json.loads(src.read_text(encoding="utf-8"))
                meta.setdefault("meta", {})["image"] = "sheet.png"
                dst.write_text(json.dumps(meta), encoding="utf-8")
            else:
                shutil.copyfile(src, dst)
    for src, dst in ((model, ch.model), (rigged, ch.rigged), (picture, ch.picture)):
        if src is not None and Path(src).exists():
            shutil.copyfile(src, dst)
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


def remove(ch: Character) -> None:
    shutil.rmtree(ch.folder, ignore_errors=True)


def load_animated(ch: Character):
    """The character's sheet back as an AnimatedSet (to play it without rendering again)."""
    from .anim3d import Action, AnimatedSet
    from .pipeline import Sprite
    meta = json.loads(ch.sheet.with_suffix(".json").read_text(encoding="utf-8"))
    sheet = np.asarray(Image.open(ch.sheet).convert("RGBA"))
    nfile = ch.folder / "sheet_n.png"
    nsheet = np.asarray(Image.open(nfile).convert("RGBA")) if nfile.exists() else None
    groups: dict[str, dict[str, list]] = {}
    order: list[str] = []
    dirs: list[str] = []
    for f in meta["frames"]:
        key, _ = f["filename"].rsplit(" ", 1)
        action, direction = key.rsplit("_", 1)
        if action not in groups:
            groups[action] = {}
            order.append(action)
        if direction not in dirs:
            dirs.append(direction)
        r = f["frame"]
        rgba = sheet[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].copy()
        nrm = nsheet[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].copy() if nsheet is not None else None
        sprite = Sprite(rgba, np.where(rgba[..., 3] > 0, 0, -1), np.zeros((1, 3), np.uint8), nrm)
        groups[action].setdefault(direction, []).append((sprite, f.get("duration", 120)))
    once = {"attack", "hurt", "death", "sit", "lie_down"}
    actions = [Action(a, a not in once, int(groups[a][dirs[0]][0][1])) for a in order]
    nested = [[[s for s, _ in groups[a][d]] for d in dirs] for a in order]
    return AnimatedSet(actions, dirs, nested)


def rebuild(ch: Character, model, progress=None) -> Character:
    """Give a library character a new coloured model (e.g. repainted) and redo its skeleton,
    sheets and thumbnail, keeping its name, kind, size and role."""
    say = progress or (lambda t: None)
    if Path(model).resolve() != ch.model.resolve():
        shutil.copyfile(model, ch.model)
    say(f"{ch.name}: adding a skeleton and animations…")
    if ch.kind == "animal":
        from .quadruped import auto_rig_quadruped
        auto_rig_quadruped(ch.model, ch.rigged)
    else:
        from .autorig import auto_rig
        auto_rig(ch.model, ch.rigged)
    render_sheets(ch.rigged, ch.sheet, ch.kind, ch.size, progress=say)
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch
