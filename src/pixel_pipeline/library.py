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
SHEET_FILES = (".png", ".json", "_n.png", "_glow.png", ".aseprite", ".tres", "_facing.gd")


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
    prompt: str = ""                     # the words it was made from (portraits reuse them)

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
                "created": self.created or time.time(), "prompt": self.prompt}
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
            "extra": {**old.get("extra", {}),       # keeps e.g. reskin_of (reskin.py)
                      "body": self.kind, "size": self.size, "role": self.role}}, indent=1),
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
                             meta.parent, d.get("prompt", "")))
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
                  progress=None, clean=True, drawn: bool = False, prompt: str = "",
                  reuse: bool = True) -> list[Path]:
    """All animations x 16 directions at the library scale, with normal maps.
    drawn: the GPU draws the character from every direction and the skeleton bends the drawings
    (drawnrig.py; key drawings kept in <folder>/keys, reused unless reuse=False)."""
    if drawn:
        from .drawnrig import drawn_sheets
        return drawn_sheets(rigged, prompt, out_png=out_png, kind=kind, size=size, frames=frames,
                            keys_folder=Path(out_png).parent / "keys", reuse=reuse,
                            progress=progress)[2]
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
    if clean is True and st.palette != "auto":                          # hand-made look,
        clean = {"palette": st.palette}                                  # one palette for all
    anim, _ = animated_turnaround([rigged], s, st.directions, frames, elevation=st.elevation,
                                  light=st.light, normals=True, clean=clean)
    return write_animated(anim, out_png, aseprite=True, godot=True, normals=True)


def add_character(name: str, model, *, kind: str = "person", size: str | None = None,
                  picture=None, role: str = "", folder: Path | None = None,
                  progress=None, prompt: str = "", drawn: bool = False) -> Character:
    """A coloured 3D model -> a library character: skeleton + animations (people: idle walk run
    attack hurt death; animals: idle walk run sit happy lie_down death), sheets, thumbnail."""
    say = progress or (lambda t: None)
    size = size or DEFAULT_SIZE.get(kind, "normal")
    base = Path(folder) if folder else root()
    ch = Character(name, kind, size, role, time.time(), _free_folder(name, base), prompt)
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
    add_project_clips(ch.rigged, kind, progress=say)      # house style: the type's clip set
    render_sheets(ch.rigged, ch.sheet, kind, size, progress=say, drawn=drawn, prompt=prompt)
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


def is_drawn(ch: Character) -> bool:
    """Were its sheets drawn by the GPU and bent by the skeleton (drawnrig.py)?"""
    return (ch.folder / "keys" / "keys.json").exists()


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
    add_project_clips(ch.rigged, ch.kind, progress=say)   # house style: the type's clip set
    render_sheets(ch.rigged, ch.sheet, ch.kind, ch.size, progress=say, drawn=is_drawn(ch),
                  prompt=ch.prompt, reuse=False)          # a new model needs new drawings
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


# --------------------------------------------------------------------------- more clips later

def add_character_animations(ch: Character, names, progress=None, *, replace: bool = False,
                             clean: bool | None = None) -> Character:
    """Give a library character more animations later (motions.py: people cast, shoot, block,
    thrust, jump, wave, kneel, victory; animals bark, howl, jump, eat). 'attack=shoot' with
    replace=True changes what its attack looks like.

    rigged.glb keeps every old clip byte for byte (the new ones are appended), and the sheet is
    re-rendered with every clip but framed and coloured from the old clips only, so the old
    clips' frames come out exactly as before and the scale doesn't change (a new clip that
    reaches past the old frame, e.g. arms over the head, is cut at its edge). clean: pixel
    clean-up of the frames; None = whatever render_sheets does by default (keep them alike)."""
    import inspect

    from .anim3d import Gltf, animated_turnaround, write_animated
    from .motions import add_animations, parse_actions
    from .settings import settings_for
    say = progress or (lambda t: None)
    if not ch.rigged.exists():
        raise ValueError(f"{ch.name} has no rigged.glb (add it to the library again first)")
    old = [a.get("name") for a in Gltf(ch.rigged).doc.get("animations", [])]
    if clean is None:
        p = inspect.signature(render_sheets).parameters.get("clean")
        clean = bool(p.default) if p is not None and p.default is not inspect.Parameter.empty else False
    say(f"{ch.name}: adding {', '.join(str(n) for n in names)}…")
    tmp = ch.rigged.with_name("rigged.new.glb")
    add_animations(ch.rigged, names, tmp, replace=replace)
    redone = {n for n, _ in parse_actions(names, _known_motions(ch.kind))} if replace else set()
    keep = [n for n in old if n not in redone] or None
    try:
        st = style()
        w, h = frame_size(ch.kind, ch.size, st)
        colours = st.colours if ch.kind == "person" else max(8, round(st.colours * 2 / 3))
        s = settings_for(ch.kind if ch.kind == "animal" else "person", width=w, height=h,
                         colors=colours)
        if st.outline == "none":
            s.outline = False
        elif st.outline in ("selout", "solid"):
            s.outline_style = st.outline
        if st.palette != "auto":
            s.palette = st.palette
        if clean is True and st.palette != "auto":       # the same clean-up render_sheets did
            clean = {"palette": st.palette}
        say(f"Rendering every animation from {st.directions} directions…")
        if is_drawn(ch):                    # same drawings and scale, the new clips bent too
            tmp.replace(ch.rigged)
            render_sheets(ch.rigged, ch.sheet, ch.kind, ch.size, progress=say, drawn=True,
                          prompt=ch.prompt)
        else:
            anim, _ = animated_turnaround([tmp], s, st.directions, 8, elevation=st.elevation,
                                          light=st.light, normals=True, clean=clean,
                                          frame_with=keep)
            tmp.replace(ch.rigged)
            write_animated(anim, ch.sheet, aseprite=True, godot=True, normals=True)
    finally:
        if tmp.exists():
            tmp.unlink()
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


def _known_motions(kind: str):
    if kind == "animal":
        from .quadruped import all_actions
    else:
        from .autorig import all_actions
    return all_actions()


# --------------------------------------------------------------------------- project clip sets

def add_project_clips(rigged: Path, kind: str = "person", st=None, progress=None) -> list[str]:
    """Append the project type's extra clips (assets.Style.animations, e.g. a side-scroller's
    jump) that motions.py can make and the rig doesn't have yet. Clips no motion exists for
    (fall, land, climb, crouch until someone writes them) are skipped. Returns the added names."""
    from .anim3d import Gltf
    from .assets import extra_clips, missing_clips
    st = st or style()
    have = [a.get("name") for a in Gltf(rigged).doc.get("animations", [])]
    todo = extra_clips(st, kind, have)
    skipped = missing_clips(st, kind)
    say = progress or (lambda t: None)
    if skipped:
        say(f"No motion yet for {', '.join(skipped)}: skipped.")
    if todo:
        from .motions import add_animations
        say(f"Adding the project's clips: {', '.join(todo)}…")
        add_animations(rigged, todo)
    return todo


# --------------------------------------------------------------------------- mocap + springs

def _render_with_new_clips(ch: Character, tmp: Path, keep, progress=None) -> None:
    """Re-render ch's sheet from tmp (its rigged.glb plus new clips), framed and coloured from
    the clips in keep (None = all), then put tmp in place of rigged.glb."""
    import inspect

    from .anim3d import animated_turnaround, write_animated
    from .settings import settings_for
    say = progress or (lambda t: None)
    p = inspect.signature(render_sheets).parameters.get("clean")
    clean = bool(p.default) if p is not None and p.default is not inspect.Parameter.empty else False
    st = style()
    w, h = frame_size(ch.kind, ch.size, st)
    colours = st.colours if ch.kind == "person" else max(8, round(st.colours * 2 / 3))
    s = settings_for(ch.kind if ch.kind == "animal" else "person", width=w, height=h, colors=colours)
    if st.outline == "none":
        s.outline = False
    elif st.outline in ("selout", "solid"):
        s.outline_style = st.outline
    if st.palette != "auto":
        s.palette = st.palette
    if clean is True and st.palette != "auto":
        clean = {"palette": st.palette}
    say(f"Rendering every animation from {st.directions} directions…")
    if is_drawn(ch):
        tmp.replace(ch.rigged)
        render_sheets(ch.rigged, ch.sheet, ch.kind, ch.size, progress=say, drawn=True,
                      prompt=ch.prompt)
        return
    anim, _ = animated_turnaround([tmp], s, st.directions, 8, elevation=st.elevation,
                                  light=st.light, normals=True, clean=clean, frame_with=keep)
    tmp.replace(ch.rigged)
    write_animated(anim, ch.sheet, aseprite=True, godot=True, normals=True)


def add_mocap(ch: Character, bvh_files_or_library_names, progress=None, *, replace: bool = True,
              plant_feet: bool = True) -> Character:
    """Give a library person motion-captured clips (mocap.py): BVH paths, or friendly names from
    the mocap library index ("walk", "sword slash", ...; the files must be downloaded into
    mocap.mocap_root()). A file's clip is named after it; a library name after its entry. Old
    clips keep their frames and the scale (the sheet is framed by them); a mocap clip with an
    old clip's name (walk) replaces it when replace=True. Springs (capes, hair) are re-baked."""
    from .anim3d import Gltf
    from .mocap import resolve, retarget_bvh
    say = progress or (lambda t: None)
    if ch.kind == "animal":
        raise ValueError("Mocap clips are for people (the humanoid rig)")
    if not ch.rigged.exists():
        raise ValueError(f"{ch.name} has no rigged.glb (add it to the library again first)")
    items = [resolve(x) for x in bvh_files_or_library_names]           # fail before any work
    old = [a.get("name") for a in Gltf(ch.rigged).doc.get("animations", [])]
    tmp = ch.rigged.with_name("rigged.new.glb")
    shutil.copyfile(ch.rigged, tmp)
    try:
        for it in items:
            say(f"{ch.name}: retargeting {Path(it['path']).name} as {it['name']}…")
            retarget_bvh(it["path"], tmp, it["name"], tmp, start=it["start"], end=it["end"],
                         cycle=it.get("cycle"),
                         loop=it["loop"], plant_feet=plant_feet, replace=replace)
        from .springs import has_springs, rebake
        if has_springs(tmp):
            say("Swinging the capes, hair and tails…")
            rebake(tmp)
        redone = {it["name"] for it in items}
        keep = [n for n in old if n not in redone] or None
        if has_springs(tmp):
            keep = None                     # every clip's cape moves anew: frame them all
        _render_with_new_clips(ch, tmp, keep, say)
    finally:
        if tmp.exists():
            tmp.unlink()
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch


def add_character_springs(ch: Character, progress=None, **options) -> Character:
    """Capes, hair and tails swing and settle (springs.py): spring bones are added to the
    character's rig and baked into every clip, then the sheet is rendered again. options go to
    springs.add_springs (parts, stiffness, damping, gravity)."""
    from .springs import add_springs
    say = progress or (lambda t: None)
    if not ch.rigged.exists():
        raise ValueError(f"{ch.name} has no rigged.glb (add it to the library again first)")
    tmp = ch.rigged.with_name("rigged.new.glb")
    say(f"{ch.name}: finding capes, hair and tails…")
    try:
        add_springs(ch.rigged, tmp, **options)
        _render_with_new_clips(ch, tmp, None, say)
    finally:
        if tmp.exists():
            tmp.unlink()
    make_thumb(ch.sheet, ch.thumb)
    ch.save()
    return ch
