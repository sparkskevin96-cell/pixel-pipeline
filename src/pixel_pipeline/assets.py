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

KINDS = ("character", "item", "prop", "map", "portrait", "effect")
FOLDERS = {k: k + "s" for k in KINDS}


def home() -> Path:
    env = os.environ.get("PIXEL_HOME")
    return Path(env) if env else Path.home() / "Pixel Pipeline"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s or "asset"


# --------------------------------------------------------------------------- style

@dataclass
class Style:
    """One look for every asset in a project. Sizes are in sprite pixels."""
    palette: str = "auto"          # auto, a built-in palette name, lospec:<slug>, or a .gpl/.hex file
    colours: int = 24              # colours per sprite when the palette is auto
    outline: str = "selout"        # selout (shaded), solid, none
    person_height: int = 64        # a standing person's frame height; animals/props scale from it
    elevation: float = 30.0        # camera tilt for 3D renders (0 = side view, 30 = HD-2D/iso)
    directions: int = 16           # 4, 8 or 16
    light: float = 0.2             # shading baked into renders (0 = flat)
    tile: int = 32                 # tile size for maps and tilesets
    ppu: float = 42.0              # sprite pixels per tile in 3D scenes
    icon: int = 32                 # inventory icon size

    def person_frame(self) -> tuple[int, int]:
        h = self.person_height
        return round(h * 1.5), h

    def scale(self, px: int) -> int:
        """A size given for the default 64 px person, in this project's scale."""
        return max(4, round(px * self.person_height / 64))


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
                return Style(**known)
            except (OSError, ValueError, TypeError):
                pass
        return Style()

    def save_style(self, style: Style) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        p = self.folder / "style.json"
        p.write_text(json.dumps(asdict(style), indent=1), encoding="utf-8")
        return p

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


def create_project(name: str, style: Style | None = None) -> Project:
    folder = projects_folder() / name.strip()
    folder.mkdir(parents=True, exist_ok=True)
    proj = Project(name.strip(), folder)
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
