"""Projects and the asset store: where everything you make lives, so one game's assets match
and every game can reuse them.

    ~/Pixel Pipeline/                    (PIXEL_HOME to move it)
        settings.json                    {"project": "Default"}
        Projects/<project>/
            style.json                   the project's look (Style): palette, colours, outline,
                                         character height, camera tilt, directions, tile size...
            characters/<slug>/           character.json + model, rig, sheets (library.py)
            items/<slug>/                asset.json + icon.png, sprite sheet...
            props/<slug>/  maps/<slug>/  portraits/<slug>/  effects/<slug>/

Every asset folder has asset.json (characters keep character.json too): kind, name, tags,
collections, created, and anything the kind needs in "extra". list_assets() searches them.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("character", "item", "prop", "map", "portrait", "effect", "card", "tileset", "ui")
FOLDERS = {k: k + "s" for k in KINDS} | {"ui": "ui"}       # UI kits (uikit.py): <project>/ui


def home() -> Path:
    env = os.environ.get("PIXEL_HOME")
    return Path(env) if env else Path.home() / "Pixel Pipeline"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s or "asset"


# --------------------------------------------------------------------------- style

@dataclass
class Style:
    """One look for every asset in a project. Sizes are in sprite pixels.

    The HOUSE STYLE (style_prompt ... detail) makes every generator draw like one artist:
    style_words() turns it into prompt words that every generator appends."""
    palette: str = "auto"          # auto, a built-in palette name, lospec:<slug>, or a .gpl/.hex file
    colours: int = 24              # colours per sprite when the palette is auto
    outline: str = "selout"        # selout (shaded), solid, none
    person_height: int = 64        # a standing person's frame height; animals/props scale from it
    elevation: float = 30.0        # camera tilt for 3D renders (0 = side view, 30 = HD-2D/iso)
    directions: int = 16           # 2 (E, W: side-scrollers), 4, 8 or 16
    light: float = 0.2             # shading baked into renders (0 = flat)
    tile: int = 32                 # tile size for maps and tilesets
    ppu: float = 42.0              # sprite pixels per tile in 3D scenes
    icon: int = 32                 # inventory icon size
    # --- house style (2026-10-13); old style.json files load with these defaults
    style_prompt: str = ""         # e.g. "16-bit SNES JRPG, bold outlines, warm palette"
    style_negative: str = ""       # what this project never wants, e.g. "photo, gradients"
    shading: str = "soft"          # soft | cel | flat (prompt words + light + ramp steps)
    proportions: str = "normal"    # chibi | normal | tall (characters)
    detail: str = "normal"         # chunky | normal | fine
    project_type: str = "custom"   # a PROJECT_TYPES key
    camera: str = "perspective"    # perspective (3D scenes) | ortho (flat 2D / isometric tiles)
    tile_style: str = "hd2d"       # hd2d (3D maps) | topdown | iso | side | none
    engine: str = "godot"          # godot | unity
    export: str = "godot-2d"       # default exports.PRESETS key
    animations: list = field(default_factory=list)   # character clips ([] = the built-in six)
    style_lora: str = ""           # a LoRA trained on this project's look (styletrain.py); "" = off
    style_trigger: str = ""        # its trigger word, added to prompts when the LoRA is on
    style_strength: float = 0.8

    def person_frame(self) -> tuple[int, int]:
        h = self.person_height
        return round(h * 1.5), h

    def scale(self, px: int) -> int:
        """A size given for the default 64 px person, in this project's scale."""
        return max(4, round(px * self.person_height / 64))

    def heads_tall(self) -> float:
        """How many heads tall a character is in this style (chibi 2.5, normal 5.5, tall 7.5)."""
        return HEADS_TALL.get(self.proportions, HEADS_TALL["normal"])

    def ramp_steps(self) -> int:
        """Shades per material for the hand-pixelled finish (pixelclean.hand_pixel steps)."""
        return SHADING_STEPS.get(self.shading, SHADING_STEPS["soft"])


SHADINGS = ("soft", "cel", "flat")
PROPORTIONS = ("chibi", "normal", "tall")
DETAILS = ("chunky", "normal", "fine")
HEADS_TALL = {"chibi": 2.5, "normal": 5.5, "tall": 7.5}
SHADING_LIGHT = {"soft": 0.35, "cel": 0.25, "flat": 0.0}   # Style.light when you pick a shading
SHADING_STEPS = {"soft": 5, "cel": 3, "flat": 2}
PALETTE_FILE = "palette.hex"


def set_shading(style: Style, shading: str) -> Style:
    """Pick a shading and the baked-in light that goes with it."""
    if shading not in SHADINGS:
        raise ValueError(f"shading must be one of {', '.join(SHADINGS)}")
    style.shading = shading
    style.light = SHADING_LIGHT[shading]
    return style


# --------------------------------------------------------------------------- house style words
# Every generator builds its own prompt, then appends style_words(style, kind) so a project's
# characters, items, props and portraits read as one artist's work. Pixel kinds get every word;
# "clean" kinds (the concept pictures 3D models are built from, painted portraits) drop the
# pixel-art words, which would make Hunyuan3D's input / the painted look worse.

PIXEL_KINDS = ("sprite", "character", "icon", "prop", "tile", "effect")
CLEAN_KINDS = ("model", "model-object", "portrait")
STYLE_KINDS = PIXEL_KINDS + CLEAN_KINDS
_CHARACTER_KINDS = ("character", "model", "portrait")

SHADING_WORDS = {"soft": ("", ""),
                 "cel": ("cel shading, hard-edged two-tone shadows", "soft gradients"),
                 "flat": ("flat colours, minimal shading", "gradients, heavy shading")}
PROPORTION_WORDS = {"chibi": ("chibi proportions, big head, short small body",
                              "realistic proportions, long legs"),
                    "normal": ("", ""),
                    "tall": ("tall slender heroic proportions, long legs, small head",
                             "chibi, big head, stubby")}
DETAIL_WORDS = {"chunky": ("chunky pixels, low detail, bold simple shapes", "intricate details"),
                "normal": ("", ""),
                "fine": ("fine pixel detail, intricate details", "low detail")}
DETAIL_CLEAN = {"chunky": ("simple chunky shapes", "intricate details"), "normal": ("", ""),
                "fine": ("intricate details", "")}
OUTLINE_WORDS = {"solid": ("bold dark outline", ""), "selout": ("", ""),
                 "none": ("no outlines, lineless", "black outline")}
_PIXEL_TERMS = re.compile(
    r"\b(pixel[- ]?art|pixelated|pixels?|\d+[- ]?bits?|sprite ?sheets?|sprites?|"
    r"dither(ed|ing)?|snes|nes|game ?boy( colou?r)?|gbc|gba|sega genesis|genesis|"
    r"mega ?drive|pico-?8|master system|low[- ]?res)\b", re.IGNORECASE)


def _parts(text: str) -> list[str]:
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def _clean_terms(text: str) -> str:
    """The style words without pixel-art terms ("16-bit SNES JRPG" -> "JRPG")."""
    out = []
    for part in _parts(text):
        t = re.sub(r"\s{2,}", " ", _PIXEL_TERMS.sub("", part)).strip(" -")
        if t:
            out.append(t)
    return ", ".join(out)


def _join(text: str, extra: str) -> str:
    """text + the parts of extra it doesn't already say."""
    have = (text or "").lower()
    seen, add = set(), []
    for part in _parts(extra):
        key = part.lower()
        if key in seen or (have and key in have):
            continue
        seen.add(key)
        add.append(part)
    if not add:
        return text
    return ", ".join([text.rstrip(", ")] + add) if (text or "").strip() else ", ".join(add)


def style_words(style: Style | None, kind: str = "sprite") -> tuple[str, str]:
    """(positive words, negative words) of the project's house style for one kind of prompt.

    kind: sprite / character / icon / prop / tile / effect (pixel art drawn by SDXL + the pixel
    LoRA: every word), model / model-object (the clean picture a 3D model is built from) or
    portrait (painted bust): no pixel-art words, no outline words. Proportions only go to
    character / model / portrait. A default Style gives ("", "")."""
    if kind not in STYLE_KINDS:
        raise ValueError(f"Unknown style kind '{kind}' (use {', '.join(STYLE_KINDS)})")
    st = style or Style()
    clean = kind in CLEAN_KINDS
    pos, neg = [], []
    base = _clean_terms(st.style_prompt) if clean else ", ".join(_parts(st.style_prompt))
    if base:
        pos.append(base)
    if st.style_negative:
        neg.append(", ".join(_parts(st.style_negative)))
    pairs = []
    if kind in _CHARACTER_KINDS:
        pairs.append(PROPORTION_WORDS.get(st.proportions, ("", "")))
    pairs.append(SHADING_WORDS.get(st.shading, ("", "")))
    pairs.append((DETAIL_CLEAN if clean else DETAIL_WORDS).get(st.detail, ("", "")))
    if not clean:
        pairs.append(OUTLINE_WORDS.get(st.outline, ("", "")))
    for p, n in pairs:
        if p:
            pos.append(p)
        if n:
            neg.append(n)
    return _join("", ", ".join(pos)), _join("", ", ".join(neg))


def with_style(positive: str, negative: str, style: Style | None, kind: str = "sprite"
               ) -> tuple[str, str]:
    """A generator's own (positive, negative) prompts with the house style appended."""
    pos, neg = style_words(style, kind)
    return _join(positive, pos), _join(negative, neg)


def project_style(project: "Project | None" = None) -> Style:
    """The style a generator uses when its caller gave none: the (current) project's."""
    try:
        return (project or current_project()).style
    except Exception:  # noqa: BLE001 - a broken settings file mustn't stop a generator
        return Style()


# --------------------------------------------------------------------------- project types
# A project type sets everything up for one kind of game: camera, directions, sprite size,
# house-style words, animations, map tiles and exports. "custom" = the plain defaults.

PEOPLE_CLIPS = ("idle", "walk", "run", "attack", "hurt", "death")
PROJECT_TYPES = {
    "hd2d": {
        "label": "HD-2D tactics / RPG",
        "about": "Pixel-art characters standing in a 3D world (Octopath Traveler, Triangle "
                 "Strategy). The camera looks down at 30°; characters face 16 ways.",
        "sizes": {"small": 48, "normal": 64, "large": 80},
        "style": dict(elevation=30.0, directions=16, tile=32, ppu=42.0, icon=32, colours=24,
                      outline="selout", shading="soft", light=0.35, camera="perspective",
                      tile_style="hd2d", export="godot-hd2d",
                      style_prompt="HD-2D JRPG pixel art, detailed 16-bit sprites, "
                                   "rich warm colours"),
        "animations": PEOPLE_CLIPS + ("cast", "block", "victory"),
        "map": "harbour",
    },
    "topdown-rpg": {
        "label": "Top-down RPG / adventure",
        "about": "Seen from above at a slant (Zelda, Pokémon, Stardew Valley). Small characters "
                 "that walk 4 ways over a tile map.",
        "sizes": {"small": 24, "normal": 32, "large": 48},
        "style": dict(elevation=55.0, directions=4, tile=16, ppu=16.0, icon=16, colours=16,
                      outline="solid", shading="cel", light=0.25, camera="ortho",
                      tile_style="topdown", export="godot-2d",
                      style_prompt="16-bit SNES top-down RPG pixel art, clean readable sprites"),
        "animations": PEOPLE_CLIPS,
        "map": "village",
    },
    "side-scroller": {
        "label": "Side-scroller / platformer",
        "about": "Seen straight from the side (Mega Man, Celeste, Hollow Knight). Characters "
                 "face left and right, and jump, fall, climb and crouch.",
        "sizes": {"small": 32, "normal": 48, "large": 64},
        "style": dict(elevation=0.0, directions=2, tile=16, ppu=16.0, icon=16, colours=16,
                      outline="solid", shading="cel", light=0.25, camera="ortho",
                      tile_style="side", export="godot-2d",
                      style_prompt="16-bit platformer pixel art, side view, crisp readable "
                                   "sprites"),
        "animations": PEOPLE_CLIPS + ("jump", "fall", "land", "climb", "crouch"),
        "map": "flat",
    },
    "isometric": {
        "label": "Isometric tactics",
        "about": "Diamond tiles seen from a corner (Final Fantasy Tactics, Disgaea). Characters "
                 "face 8 ways; a flat (orthographic) camera.",
        "sizes": {"small": 32, "normal": 48, "large": 64},
        "style": dict(elevation=30.0, directions=8, tile=32, ppu=32.0, icon=24, colours=20,
                      outline="selout", shading="soft", light=0.35, camera="ortho",
                      tile_style="iso", export="godot-2d",
                      style_prompt="isometric tactics RPG pixel art, 16-bit, clean shapes"),
        "animations": PEOPLE_CLIPS + ("cast", "block", "victory"),
        "map": "castle",
    },
    "cards-ui": {
        "label": "Card game / UI-heavy",
        "about": "Cards, icons, portraits and menus (Slay the Spire, Balatro). Big icons and bold "
                 "outlines; characters are mostly art, not walking sprites.",
        "sizes": {"small": 48, "normal": 64, "large": 96},
        "style": dict(elevation=30.0, directions=4, tile=32, ppu=42.0, icon=64, colours=24,
                      outline="solid", shading="cel", light=0.25, camera="ortho",
                      tile_style="none", export="godot-2d",
                      style_prompt="pixel art card game illustration, bold readable shapes, "
                                   "strong silhouettes"),
        "animations": ("idle", "attack", "hurt"),
        "map": "flat",
    },
    "custom": {
        "label": "Custom",
        "about": "Start from the plain defaults and set everything yourself.",
        "sizes": {"small": 48, "normal": 64, "large": 80},
        "style": {},
        "animations": (),
        "map": "flat",
    },
}
SPRITE_SIZES = ("small", "normal", "large")


def style_for_type(project_type: str = "custom", size: str = "normal", engine: str = "godot",
                   **overrides) -> Style:
    """The Style a new project of this type starts with. size: small / normal / large (the
    type's person heights, PROJECT_TYPES[t]["sizes"]); engine: godot / unity (unity -> Unity
    exports). overrides (Style fields) win; None keeps the preset."""
    if project_type not in PROJECT_TYPES:
        raise ValueError(f"Unknown project type '{project_type}' (use {', '.join(PROJECT_TYPES)})")
    if size not in SPRITE_SIZES:
        raise ValueError(f"size must be one of {', '.join(SPRITE_SIZES)}")
    t = PROJECT_TYPES[project_type]
    data = dict(t["style"])
    data.update(project_type=project_type, person_height=t["sizes"][size],
                animations=list(t["animations"]), engine=engine)
    if engine == "unity":
        data["export"] = "unity"
    data.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(data) - set(Style.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Unknown style field(s): {', '.join(sorted(unknown))}")
    return Style(**data)


def _known_clips(kind: str) -> set[str]:
    try:
        if kind == "animal":
            from .quadruped import all_actions
        else:
            from .autorig import all_actions
        return set(all_actions())
    except Exception:  # noqa: BLE001
        return set()


def animation_set(style: Style | None = None, kind: str = "person") -> list[str]:
    """The character clips this project wants (its type's set; [] = the built-in ones).
    Animals keep their own built-in set."""
    st = style or Style()
    return [] if kind == "animal" else list(st.animations)


def missing_clips(style: Style | None = None, kind: str = "person") -> list[str]:
    """Clips the project wants that no motion exists for yet: they are skipped."""
    known = _known_clips(kind)
    return [n for n in animation_set(style, kind) if (n.partition("=")[2] or n) not in known]


def extra_clips(style: Style | None = None, kind: str = "person", have=()) -> list[str]:
    """The project's clips a rig doesn't have yet but motions exist for (to add_animations)."""
    known = _known_clips(kind)
    have = set(have)
    return [n for n in animation_set(style, kind)
            if (n.partition("=")[2] or n) in known and n.partition("=")[0] not in have]


# --------------------------------------------------------------------------- projects

@dataclass
class Project:
    name: str
    folder: Path

    @property
    def style(self) -> Style:
        p = self.folder / "style.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                known = {k: v for k, v in data.items() if k in Style.__dataclass_fields__}
                st = Style(**known)
                st.palette = self._palette_path(st.palette)
                return st
            except (OSError, ValueError, TypeError):
                pass
        return Style()

    def _palette_path(self, spec) -> str:
        """A palette file kept in the project (palette.hex) is stored relative; give it back as
        a full path so every generator loads it wherever the project lives."""
        spec = str(spec or "auto")
        p = Path(spec)
        if p.suffix.lower() in (".hex", ".gpl", ".txt", ".png", ".gif") and not p.is_absolute() \
                and (self.folder / p).exists():
            return str(self.folder / p)
        return spec

    def save_style(self, style: Style) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        p = self.folder / "style.json"
        data = asdict(style)
        pal = Path(str(data.get("palette") or "auto"))
        try:
            if pal.is_absolute() and pal.resolve().parent == self.folder.resolve():
                data["palette"] = pal.name                   # moves with the project
        except OSError:
            pass
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return p

    @property
    def palette_file(self) -> Path:
        """Where lock_palette writes the project's own palette."""
        return self.folder / PALETTE_FILE

    def kind_folder(self, kind: str) -> Path:
        if kind not in FOLDERS:
            raise ValueError(f"Unknown asset kind '{kind}' (use {', '.join(KINDS)})")
        return self.folder / FOLDERS[kind]


def projects_folder() -> Path:
    return home() / "Projects"


def _settings() -> dict:
    p = home() / "settings.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(data: dict) -> None:
    home().mkdir(parents=True, exist_ok=True)
    (home() / "settings.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


def _migrate() -> None:
    """The first Character Library lived in ~/Pixel Pipeline/Characters: move it into the
    Default project once."""
    old = home() / "Characters"
    new = projects_folder() / "Default" / "characters"
    if old.exists() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))


def list_projects() -> list[Project]:
    _migrate()
    base = projects_folder()
    out = [Project(p.name, p) for p in sorted(base.iterdir()) if p.is_dir()] if base.exists() else []
    if not out:
        out = [create_project("Default")]
    return out


def create_project(name: str, style: Style | None = None, type: str | None = None, *,
                   size: str = "normal", engine: str = "godot") -> Project:
    """A project folder with its style.json. type (a PROJECT_TYPES key) applies that type's
    preset (style_for_type with size / engine); a style you give wins over the preset."""
    if not name or not name.strip():
        raise ValueError("Give the project a name.")
    if type is not None and type not in PROJECT_TYPES:
        raise ValueError(f"Unknown project type '{type}' (use {', '.join(PROJECT_TYPES)})")
    folder = projects_folder() / name.strip()
    folder.mkdir(parents=True, exist_ok=True)
    proj = Project(name.strip(), folder)
    if style is None and type is not None:
        style = style_for_type(type, size, engine)
    elif style is not None and type is not None and style.project_type == "custom":
        style.project_type = type
    if not (folder / "style.json").exists() or style is not None:
        proj.save_style(style or Style())
    return proj


def current_project() -> Project:
    _migrate()
    name = _settings().get("project", "Default")
    folder = projects_folder() / name
    if not (folder / "style.json").exists():
        return create_project(name)
    return Project(name, folder)


def set_current_project(name: str) -> Project:
    data = _settings()
    data["project"] = name
    _save_settings(data)
    return current_project()


# --------------------------------------------------------------------------- assets

@dataclass
class Asset:
    kind: str
    name: str
    folder: Path = field(compare=False, repr=False)
    tags: list = field(default_factory=list)
    collections: list = field(default_factory=list)
    created: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def thumb(self) -> Path:
        return self.folder / "thumb.png"

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {"kind": self.kind, "name": self.name, "tags": self.tags,
                "collections": self.collections, "created": self.created or time.time(),
                "extra": self.extra}
        p = self.folder / "asset.json"
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return p

    def matches(self, query: str = "", tag: str | None = None, collection: str | None = None) -> bool:
        if tag and tag not in self.tags:
            return False
        if collection and collection not in self.collections:
            return False
        q = query.strip().lower()
        if q:
            hay = " ".join([self.name, self.kind, *self.tags, *self.collections,
                            str(self.extra.get("prompt", ""))]).lower()
            return all(word in hay for word in q.split())
        return True


def _load(folder: Path, kind: str) -> Asset | None:
    meta = folder / "asset.json"
    if not meta.exists() and kind == "character" and (folder / "character.json").exists():
        meta = folder / "character.json"          # characters made before assets.py
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    extra = d.get("extra", {})
    if meta.name == "character.json":
        extra = {k: d[k] for k in ("kind", "size", "role") if k in d}
        extra["body"] = extra.pop("kind", "person")
    return Asset(kind, d.get("name", folder.name), folder, d.get("tags", []),
                 d.get("collections", []), d.get("created", 0.0), extra)


def list_assets(kind: str | None = None, query: str = "", tag: str | None = None,
                collection: str | None = None, project: Project | None = None) -> list[Asset]:
    """Assets of a project (default: the current one), newest first, filtered."""
    proj = project or current_project()
    kinds = [kind] if kind else list(KINDS)
    out = []
    for k in kinds:
        base = proj.kind_folder(k)
        if k == "character" and os.environ.get("PIXEL_LIBRARY"):
            base = Path(os.environ["PIXEL_LIBRARY"])
        if not base.exists():
            continue
        for folder in base.iterdir():
            if folder.is_dir():
                a = _load(folder, k)
                if a is not None and a.matches(query, tag, collection):
                    out.append(a)
    return sorted(out, key=lambda a: -a.created)


def new_asset(kind: str, name: str, project: Project | None = None, **extra) -> Asset:
    """A fresh folder for a new asset (name-2, name-3… if taken), with asset.json written."""
    proj = project or current_project()
    base = proj.kind_folder(kind)
    folder = base / slug(name)
    k = 2
    while (folder / "asset.json").exists() or (folder / "character.json").exists():
        folder = base / f"{slug(name)}-{k}"
        k += 1
    a = Asset(kind, name, folder, created=time.time(), extra=dict(extra))
    a.save()
    return a


def all_tags(project: Project | None = None) -> list[str]:
    return sorted({t for a in list_assets(project=project) for t in a.tags})


def all_collections(project: Project | None = None) -> list[str]:
    return sorted({c for a in list_assets(project=project) for c in a.collections})


def set_tags(asset: Asset, tags=None, collections=None) -> Asset:
    """Change tags/collections (characters: also written into asset.json beside character.json)."""
    if tags is not None:
        asset.tags = sorted({t.strip() for t in tags if t.strip()})
    if collections is not None:
        asset.collections = sorted({c.strip() for c in collections if c.strip()})
    asset.save()
    return asset


# --------------------------------------------------------------------------- palette lock

APPROVED_TAG = "approved"


def approved_assets(project: Project | None = None) -> list[Asset]:
    """The assets a palette is locked from: those tagged "approved" if any are, else all."""
    every = list_assets(project=project)
    tagged = [a for a in every if APPROVED_TAG in a.tags]
    return tagged or every


def asset_pixels(asset: Asset, limit: int = 20000, seed: int = 0):
    """Opaque RGB pixels (N, 3) of an asset's finished picture (sheet, icon or thumbnail),
    at most `limit` of them so every asset counts about the same; None if it has none."""
    import numpy as np
    from PIL import Image
    for name in ("sheet.png", "icon.png", "thumb.png"):
        p = Path(asset.folder) / name
        if p.exists():
            try:
                rgba = np.asarray(Image.open(p).convert("RGBA")).reshape(-1, 4)
            except OSError:
                continue
            rgb = rgba[rgba[:, 3] >= 128][:, :3]
            if not len(rgb):
                continue
            if len(rgb) > limit:
                rgb = rgb[np.random.default_rng(seed).choice(len(rgb), limit, replace=False)]
            return rgb
    return None


def lock_palette(project: Project | None = None, colours: int = 32, *,
                 source: Project | None = None, assets=None) -> Path:
    """Build ONE palette (k-means in Lab, `colours` of them, 24-32 suits most games) from the
    approved assets of `source` (default: the project itself; or the given assets), write it
    as <project>/palette.hex and point the project's Style.palette at it, so every asset made
    from now on snaps to the same colours."""
    import numpy as np
    from .palettes import auto_palette, rgb_to_hex
    proj = project or current_project()
    pool = list(assets) if assets is not None else approved_assets(source or proj)
    pixels = [px for px in (asset_pixels(a, seed=i) for i, a in enumerate(pool)) if px is not None]
    if not pixels:
        raise ValueError("No finished assets with pictures to take colours from yet: make a few "
                         "characters or items first (tag the good ones 'approved').")
    pal = auto_palette(np.concatenate(pixels), max(2, int(colours)))
    path = proj.palette_file
    path.write_text("\n".join(rgb_to_hex(c).lstrip("#") for c in pal) + "\n", encoding="utf-8")
    st = proj.style
    st.palette = str(path)
    proj.save_style(st)
    return path


# --------------------------------------------------------------------------- restyle

_LOOK_FIELDS = ("palette", "colours", "outline", "person_height", "elevation", "directions",
                "light", "icon", "shading", "detail")
_WORD_FIELDS = ("style_prompt", "style_negative", "proportions")


def restyle_plan(project: Project | None = None, old: Style | None = None,
                 new: Style | None = None) -> list[str]:
    """What restyling the project's assets to `new` (default: its saved style) would change,
    in plain lines. Changes to the house-style WORDS can't be applied to finished art without
    the GPU making it again; the look fields (palette, outline, sizes, camera...) can."""
    proj = project or current_project()
    new = new or proj.style
    old = old or Style()
    changed = [f for f in _LOOK_FIELDS if getattr(old, f) != getattr(new, f)]
    words = [f for f in _WORD_FIELDS if getattr(old, f) != getattr(new, f)]
    lines = []
    if changed:
        lines.append("Changes: " + ", ".join(f"{f} {getattr(old, f)} -> {getattr(new, f)}"
                                             for f in changed))
    every = list_assets(project=proj)
    for a in every:
        what = _restyle_how(a)
        if what:
            lines.append(f"{a.kind} {a.name}: {what}")
    if words:
        lines.append("Style words changed (" + ", ".join(words) + "): new assets use them; "
                     "existing art keeps its drawing until you make it again on the GPU.")
    if not lines:
        lines.append("Nothing to restyle.")
    return lines


def _restyle_how(a: Asset) -> str:
    f = Path(a.folder)
    if a.kind == "character":
        return ("re-render every animation from its rigged model (CPU, a few minutes)"
                if (f / "model.glb").exists() else "")
    if a.kind == "item":
        bits = []
        if list(f.glob("picture_v*.png")):
            bits.append("redo icons from the saved pictures")
        if a.extra.get("world"):
            bits.append("re-render the world sprite")
        return ", ".join(bits)
    if a.kind == "prop":
        if a.extra.get("look") == "drawn":
            n = a.extra.get("directions", 1)
            return "re-convert the chosen drawing" if n == 1 and (f / "drawn_v1.png").exists() \
                else "drawn from 4 sides: make it again on the GPU"
        return "re-render from its 3D model" if (f / "model.glb").exists() else ""
    return ""


def restyle(project: Project | None = None, kinds=("character", "item", "prop"),
            progress=None) -> list[str]:
    """Re-render the project's assets in its CURRENT style with no GPU: characters from their
    model (library.rebuild; only for the current project's Character Library), items (icons
    from the saved pictures + world sprite), props (3D: render_prop; drawn: the chosen drawing
    re-converted). Returns what was done / skipped, one line each."""
    say = progress or (lambda t: None)
    proj = project or current_project()
    done = []
    for a in list_assets(project=proj):
        if a.kind not in kinds or not _restyle_how(a):
            continue
        try:
            if a.kind == "character":
                from . import library
                ch = library.get(a.name, Path(a.folder).parent)
                if ch is None or not ch.model.exists():
                    continue
                say(f"Restyling {a.name}…")
                library.rebuild(ch, ch.model, progress=say)
            elif a.kind == "item":
                from . import objects
                say(f"Restyling {a.name}…")
                item = objects.open_asset(a.folder)
                if list(Path(a.folder).glob("picture_v*.png")):
                    objects.redo_icons(item, proj)
                if item.extra.get("world"):
                    objects.render_item(item, project=proj)
            elif a.kind == "prop":
                from . import objects
                prop = objects.open_asset(a.folder)
                say(f"Restyling {a.name}…")
                if prop.extra.get("look") == "drawn":
                    if int(prop.extra.get("directions", 1)) != 1:
                        done.append(f"{a.name}: skipped (drawn from 4 sides needs the GPU)")
                        continue
                    objects.choose_drawn(prop, int(prop.extra.get("chosen", 1)), project=proj)
                else:
                    objects.render_prop(prop, project=proj)
            done.append(f"{a.name}: restyled")
        except Exception as e:  # noqa: BLE001 - one bad asset mustn't stop the rest
            done.append(f"{a.name}: failed ({e})")
    return done
