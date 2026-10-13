"""Export presets: one asset (or a whole collection) out of the studio, ready for an engine.

    export_asset(asset, "unity", "C:/MyGame/Assets/Sprites")      -> [written files]
    export_collection("starter kit", "godot-2d", "C:/MyGame/art")  -> one folder per asset + index.json

An asset is a library Character, an assets.Asset or just its folder. Sprite-sheet assets
(characters, and items/props made with a sheet) have sheet.png + sheet.json (a row per
action_direction, frames named "{action}_{dir} {i}"), maybe sheet_n.png (normal map),
sheet.aseprite, sheet.tres + sheet_facing.gd. Items may have only icon.png; that is exported
as a one-frame sprite.

Presets (PRESETS): godot-2d, godot-hd2d, unity, gamemaker, aseprite, generic.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PRESETS = {
    "godot-2d": "Godot 2D (SpriteFrames .tres + facing script)",
    "godot-hd2d": "Godot HD-2D unit (for the tactics project)",
    "unity": "Unity (sliced sprite sheet + .meta)",
    "gamemaker": "GameMaker (one _stripN PNG per animation)",
    "aseprite": "Aseprite (.aseprite with tags)",
    "generic": "Generic (PNG + JSON + single frames)",
}


# --------------------------------------------------------------------------- reading an asset

@dataclass
class Frame:
    name: str
    x: int
    y: int
    w: int
    h: int
    duration: int = 100


@dataclass
class Source:
    """What an asset folder has to export."""
    name: str          # display name
    kind: str
    folder: Path
    sheet: Path | None     # sheet.png (with sheet.json), or None
    frames: list           # [Frame] in sheet order (one frame = the whole image for icons)
    tags: list             # [(name, first, last)] in frame order
    normals: Path | None
    icon: Path | None
    image: Path | None     # the main picture: sheet, else icon, else any png

    @property
    def file_name(self) -> str:
        """Engine-safe base name (letters, digits, underscores)."""
        return file_name(self.name)

    def groups(self) -> list[tuple[str, list]]:
        """[(action_dir, [Frame...])] in sheet order: frames named '<key> <i>' share a group;
        other names ('SE', 'default') are a group each."""
        out: dict[str, list] = {}
        for f in self.frames:
            out.setdefault(group_key(f.name), []).append(f)
        return list(out.items())


def group_key(frame_name: str) -> str:
    """'walk_SE 3' -> 'walk_SE'; 'SE' -> 'SE'."""
    parts = frame_name.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else frame_name


def file_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s or "asset"


def _frames(meta: dict) -> list[Frame]:
    """Frames of an Aseprite/TexturePacker JSON, array or hash format."""
    raw = meta.get("frames", [])
    items = raw.items() if isinstance(raw, dict) else ((f.get("filename", f"frame {i}"), f)
                                                       for i, f in enumerate(raw))
    out = []
    for name, f in items:
        r = f["frame"]
        out.append(Frame(str(name), int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]),
                         int(f.get("duration", 100))))
    return out


def _tags(meta: dict, frames: list[Frame]) -> list[tuple[str, int, int]]:
    tags = [(t["name"], int(t["from"]), int(t["to"])) for t in meta.get("meta", {}).get("frameTags", [])]
    if tags or not frames:
        return tags
    out: list = []
    for i, f in enumerate(frames):
        key = group_key(f.name)
        if out and out[-1][0] == key:
            out[-1] = (key, out[-1][1], i)
        else:
            out.append((key, i, i))
    return out


def source(asset) -> Source:
    """A library Character, an assets.Asset, a folder or a sheet .png -> Source."""
    if isinstance(asset, (str, Path)):
        p = Path(asset)
        folder = p.parent if p.is_file() else p
        name, kind = folder.name, "asset"
        for meta in ("asset.json", "character.json"):
            if (folder / meta).exists():
                try:
                    d = json.loads((folder / meta).read_text(encoding="utf-8"))
                    name, kind = d.get("name", name), d.get("kind", kind)
                    if meta == "character.json":
                        kind = "character"
                    break
                except (OSError, ValueError):
                    pass
        sheet = p if p.is_file() and p.suffix == ".png" else folder / "sheet.png"
    else:
        folder = Path(asset.folder)
        name = asset.name
        kind = getattr(asset, "kind", "character")
        if kind in ("person", "animal"):          # a library Character's body kind
            kind = "character"
        sheet = folder / "sheet.png"
    icon = folder / "icon.png" if (folder / "icon.png").exists() else None
    frames, tags, normals = [], [], None
    if sheet.exists() and sheet.with_suffix(".json").exists():
        meta = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
        frames = _frames(meta)
        tags = _tags(meta, frames)
        n = sheet.with_name(sheet.stem + "_n.png")
        normals = n if n.exists() else None
    else:
        sheet = None
    image = sheet or icon
    if image is None:
        pngs = sorted(p for p in folder.glob("*.png") if p.name != "thumb.png") if folder.exists() else []
        image = pngs[0] if pngs else (folder / "thumb.png" if (folder / "thumb.png").exists() else None)
    if image is None:
        raise ValueError(f"'{name}' has no picture to export ({folder}).")
    if not frames:
        w, h = Image.open(image).size
        frames = [Frame(file_name(name), 0, 0, w, h)]
    return Source(name, kind, folder, sheet, frames, tags, normals, icon, image)


def _rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def _save(rgba: np.ndarray, path: Path) -> Path:
    Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(path)
    return path


def _crop(img: np.ndarray, f: Frame) -> np.ndarray:
    return img[f.y:f.y + f.h, f.x:f.x + f.w]


def _foot_row(img: np.ndarray, frames: list[Frame]) -> int:
    lows = []
    for f in frames:
        ys = np.nonzero(_crop(img, f)[..., 3].max(1) >= 128)[0]
        if len(ys):
            lows.append(ys.max())
    return int(np.median(lows)) + 1 if lows else frames[0].h


def _copy(src: Path, dst: Path, written: list) -> Path:
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    written.append(dst)
    return dst


# --------------------------------------------------------------------------- presets

def _godot_2d(src: Source, out: Path, res_path: str | None = None, **_) -> list[Path]:
    """<name>.png + <name>.tres (SpriteFrames; texture path relative to the .tres, or
    res_path + name.png when given) + <name>_facing.gd + <name>_n.png."""
    from .godot import write_sprite_frames
    n = src.file_name
    written: list[Path] = []
    png = _copy(src.image, out / f"{n}.png", written)
    tex = f"{res_path.rstrip('/')}/{png.name}" if res_path else png.name
    old_tres = src.sheet.with_suffix(".tres") if src.sheet else None
    old_gd = src.sheet.with_name(src.sheet.stem + "_facing.gd") if src.sheet else None
    tres = out / f"{n}.tres"
    if old_tres is not None and old_tres.exists():
        text = old_tres.read_text(encoding="utf-8")
        text = re.sub(r'(\[ext_resource type="Texture2D" path=")[^"]*(")',
                      lambda m: m.group(1) + tex + m.group(2), text)
        tres.write_text(text, encoding="utf-8")
    else:
        rects = [(f.x, f.y, f.w, f.h) for f in src.frames]
        anims = {}
        for key, fs in src.groups():
            idx = [src.frames.index(f) for f in fs]
            anims[key] = {"frames": idx, "duration_ms": fs[0].duration,
                          "loop": key.rsplit("_", 1)[0] not in ONCE}
        write_sprite_frames(tres, png.name, rects, animations=anims)
        if res_path:
            text = tres.read_text(encoding="utf-8").replace(f'path="{png.name}"', f'path="{tex}"')
            tres.write_text(text, encoding="utf-8")
    written.append(tres)
    if old_gd is not None and old_gd.exists():
        gd = out / f"{n}_facing.gd"
        text = re.sub(r"sprite_frames to \S+?\.tres", f"sprite_frames to {tres.name}",
                      old_gd.read_text(encoding="utf-8"))
        gd.write_text(text, encoding="utf-8")
        written.append(gd)
    else:
        dirs, acts = _directions_actions(src)
        if dirs and "S" in dirs:
            from .anim3d import godot_action_script
            gd = out / f"{n}_facing.gd"
            gd.write_text(godot_action_script(dirs, acts, tres.name), encoding="utf-8")
            written.append(gd)
    if src.normals:
        _copy(src.normals, out / f"{n}_n.png", written)
    if src.icon and src.icon != src.image:
        _copy(src.icon, out / f"{n}_icon.png", written)
    return written


ONCE = {"attack", "hurt", "death", "sit", "lie_down"}


def _directions_actions(src: Source) -> tuple[list[str], list[str]]:
    from .turnaround import COMPASS16
    dirs, acts = [], []
    for key, _ in src.groups():
        if "_" not in key:
            return [], []
        a, d = key.rsplit("_", 1)
        if d not in COMPASS16:
            return [], []
        if d not in dirs:
            dirs.append(d)
        if a not in acts:
            acts.append(a)
    return dirs, acts


def _godot_hd2d(src: Source, out: Path, ppu: float | None = None, **_) -> list[Path]:
    """out = the Godot project folder: writes res://units/<name>/ like hd2d.add_unit."""
    from .hd2d import add_unit
    if src.sheet is None:
        raise ValueError(f"'{src.name}' has no sprite sheet; HD-2D units need one.")
    if ppu is None:
        ppu = _project_style_value("ppu", 42.0)
    folder = add_unit(out, src.file_name, src.sheet, ppu)
    return sorted(p for p in folder.iterdir() if p.is_file())


def _project_style_value(field_name: str, default):
    try:
        from .assets import current_project
        return getattr(current_project().style, field_name, default)
    except Exception:  # noqa: BLE001 - no project yet: the default
        return default


def unity_guid(text: str) -> str:
    """Stable 32-hex GUID (same asset -> same GUID, so re-exports keep Unity references)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, "pixel-pipeline:" + text).hex


def _unity_platform(compression: int = 0) -> str:
    return "".join(f"""  - serializedVersion: 3
    buildTarget: {target}
    maxTextureSize: 8192
    resizeAlgorithm: 0
    textureFormat: -1
    textureCompression: {compression}
    compressionQuality: 50
    crunchedCompression: 0
    allowsAlphaSplitting: 0
    overridden: 0
    ignorePlatformSupport: 0
    androidETC2FallbackOverride: 0
    forceMaximumCompressionQuality_BC6H_BC7: 0
""" for target in ("DefaultTexturePlatform", "Standalone"))


def unity_meta(guid: str, size: tuple[int, int], frames: list[Frame] | None, *, ppu: float = 32,
               normal_map: bool = False, pivot: tuple[float, float] = (0.5, 0.0)) -> str:
    """Unity TextureImporter .meta: point filter, no compression, no mipmaps. frames given ->
    Sprite (Multiple) with one sprite per frame (Unity rects start bottom-left, so y is
    flipped); None -> a single sprite; normal_map -> texture type Normal map (linear)."""
    w, h = size
    sprites = ""
    names = ""
    if frames and not normal_map:
        for i, f in enumerate(frames):
            sid = uuid.uuid5(uuid.NAMESPACE_URL, f"{guid}/{f.name}").hex
            internal = 21300000 + 2 * i
            sprites += f"""    - serializedVersion: 2
      name: {_yaml_name(f.name)}
      rect:
        serializedVersion: 2
        x: {f.x}
        y: {h - f.y - f.h}
        width: {f.w}
        height: {f.h}
      alignment: 7
      pivot: {{x: {pivot[0]}, y: {pivot[1]}}}
      border: {{x: 0, y: 0, z: 0, w: 0}}
      outline: []
      physicsShape: []
      tessellationDetail: 0
      bones: []
      spriteID: {sid}
      internalID: {internal}
      vertices: []
      indices:
      edges: []
      weights: []
"""
            names += f"      {_yaml_name(f.name)}: {internal}\n"
    sprite_mode = 0 if normal_map else (2 if frames else 1)
    texture_type = 1 if normal_map else 8
    return f"""fileFormatVersion: 2
guid: {guid}
TextureImporter:
  internalIDToNameTable: []
  externalObjects: {{}}
  serializedVersion: 12
  mipmaps:
    mipMapMode: 0
    enableMipMap: 0
    sRGBTexture: {0 if normal_map else 1}
    linearTexture: 0
    fadeOut: 0
    borderMipMap: 0
    mipMapsPreserveCoverage: 0
    alphaTestReferenceValue: 0.5
    mipMapFadeDistanceStart: 1
    mipMapFadeDistanceEnd: 3
  bumpmap:
    convertToNormalMap: 0
    externalNormalMap: 0
    heightScale: 0.25
    normalMapFilter: 0
    flipGreenChannel: 0
  isReadable: 0
  streamingMipmaps: 0
  streamingMipmapsPriority: 0
  vTOnly: 0
  ignoreMipmapLimit: 0
  grayScaleToAlpha: 0
  generateCubemap: 6
  cubemapConvolution: 0
  seamlessCubemap: 0
  textureFormat: 1
  maxTextureSize: 8192
  textureSettings:
    serializedVersion: 2
    filterMode: 0
    aniso: 1
    mipBias: 0
    wrapU: 1
    wrapV: 1
    wrapW: 1
  nPOTScale: 0
  lightmap: 0
  compressionQuality: 50
  spriteMode: {sprite_mode}
  spriteExtrude: 1
  spriteMeshType: 0
  alignment: 7
  spritePivot: {{x: {pivot[0]}, y: {pivot[1]}}}
  spritePixelsToUnits: {ppu:g}
  spriteBorder: {{x: 0, y: 0, z: 0, w: 0}}
  spriteGenerateFallbackPhysicsShape: 0
  alphaUsage: 1
  alphaIsTransparency: {0 if normal_map else 1}
  spriteTessellationDetail: -1
  textureType: {texture_type}
  textureShape: 1
  singleChannelComponent: 0
  flipbookRows: 1
  flipbookColumns: 1
  maxTextureSizeSet: 0
  compressionQualitySet: 0
  textureFormatSet: 0
  ignorePngGamma: 0
  applyGammaDecoding: 0
  swizzle: 50462976
  cookieLightType: 0
  platformSettings:
{_unity_platform(0)}  spriteSheet:
    serializedVersion: 2
    sprites:
{sprites or "    []" + chr(10)}    outline: []
    physicsShape: []
    bones: []
    spriteID:
    internalID: 0
    vertices: []
    indices:
    edges: []
    weights: []
    secondaryTextures: []
    nameFileIdTable:
{names or "      {}" + chr(10)}  mipmapLimitGroupName:
  pSDRemoveMatte: 0
  userData:
  assetBundleName:
  assetBundleVariant:
"""


def _yaml_name(name: str) -> str:
    """Sprite names as plain YAML scalars: 'walk_SE 3' -> walk_SE_3."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "sprite"


UNITY_README = """{name} — exported from pixel-pipeline for Unity
=============================================

Files
  {png}            the sprite sheet ({count} sprites, already sliced by its .meta)
  {png}.meta       import settings: Sprite (Multiple), Point filter, no compression,
                   no mipmaps, pivot bottom-centre (the feet), {ppu:g} pixels per unit
{normal_line}
How to use
  1. Copy every file (the .meta files too) into your project's Assets folder. Unity reads
     the .meta and slices the sheet; expand {png} in the Project window to see the sprites.
  2. Sprite names are <action>_<direction>_<frame>, e.g. walk_SE_0. Select one animation's
     frames and drag them into the Scene: Unity makes an Animation clip from them.
     Frame times (ms) are in {name}_frames.json.
  3. Pixel-perfect: add a Pixel Perfect Camera (URP 2D) with Assets Pixels Per Unit = {ppu:g}.
  4. Normal map: on a Sprite-Lit material (URP 2D) add {npng} as the sheet's Secondary
     Texture named _NormalMap (Sprite Editor > Secondary Textures).
"""


def _unity(src: Source, out: Path, ppu: float | None = None, **_) -> list[Path]:
    n = src.file_name
    written: list[Path] = []
    if ppu is None:
        ppu = float(_project_style_value("tile", 32))
    png = _copy(src.image, out / f"{n}.png", written)
    size = Image.open(png).size
    sheet_frames = src.frames if src.sheet is not None else None
    guid = unity_guid(f"{src.kind}/{src.name}")
    meta = out / f"{n}.png.meta"
    meta.write_text(unity_meta(guid, size, sheet_frames, ppu=ppu), encoding="utf-8")
    written.append(meta)
    normal_line = ""
    npng = f"{n}_n.png"
    if src.normals:
        _copy(src.normals, out / npng, written)
        nmeta = out / f"{npng}.meta"
        nmeta.write_text(unity_meta(unity_guid(f"{src.kind}/{src.name}/normals"), size, None,
                                    ppu=ppu, normal_map=True), encoding="utf-8")
        written.append(nmeta)
        normal_line = (f"  {npng}          normal map (OpenGL, green up), imported as a Normal map\n")
    if src.icon and src.icon != src.image:
        icon = _copy(src.icon, out / f"{n}_icon.png", written)
        imeta = out / f"{icon.name}.meta"
        imeta.write_text(unity_meta(unity_guid(f"{src.kind}/{src.name}/icon"), Image.open(icon).size,
                                    None, ppu=ppu), encoding="utf-8")
        written.append(imeta)
    frames_json = out / f"{n}_frames.json"
    frames_json.write_text(json.dumps({
        "sprites": [{"name": _yaml_name(f.name), "duration_ms": f.duration} for f in src.frames],
        "animations": [{"name": k, "sprites": [_yaml_name(f.name) for f in fs],
                        "frame_ms": fs[0].duration} for k, fs in src.groups()]}, indent=1),
        encoding="utf-8")
    written.append(frames_json)
    readme = out / f"{n}_README.txt"
    readme.write_text(UNITY_README.format(name=n, png=png.name, count=len(src.frames), ppu=ppu,
                                          normal_line=normal_line, npng=npng), encoding="utf-8")
    written.append(readme)
    return written


def _gamemaker(src: Source, out: Path, normals: bool = True, **_) -> list[Path]:
    """One horizontal strip per action_direction: <name>_<action>_<dir>_strip<N>.png (GameMaker
    slices _stripN itself on import; direction keeps its case: knight_walk_SE_strip8.png), the
    normal map as <name>_<action>_<dir>_normal_strip<N>.png, + <name>_gamemaker.json."""
    n = src.file_name
    img = _rgba(src.image)
    nimg = _rgba(src.normals) if (normals and src.normals) else None
    written: list[Path] = []
    summary = {"name": n, "origin": "bottom centre", "sprites": []}
    groups = src.groups()
    single = len(groups) == 1 and len(groups[0][1]) == 1
    for key, fs in groups:
        w, h = max(f.w for f in fs), max(f.h for f in fs)
        base = n if single else f"{n}_{_yaml_name(key)}"
        files = []
        for layer, suffix in ((img, ""), (nimg, "_normal")):
            if layer is None:
                continue
            strip = np.zeros((h, w * len(fs), 4), np.uint8)
            for i, f in enumerate(fs):
                c = _crop(layer, f)
                strip[:c.shape[0], i * w:i * w + c.shape[1]] = c
            name = f"{base}{suffix}.png" if len(fs) == 1 else f"{base}{suffix}_strip{len(fs)}.png"
            written.append(_save(strip, out / name))
            files.append(name)
        summary["sprites"].append({
            "sprite": base, "file": files[0], "normal_map": files[1] if len(files) > 1 else None,
            "frames": len(fs), "frame_w": w, "frame_h": h,
            "fps": round(1000 / max(1, fs[0].duration), 3),
            "origin": [w // 2, _foot_row(img, fs)]})
    if src.icon and src.icon != src.image:
        _copy(src.icon, out / f"{n}_icon.png", written)
        summary["icon"] = f"{n}_icon.png"
    js = out / f"{n}_gamemaker.json"
    js.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    written.append(js)
    return written


def _aseprite(src: Source, out: Path, **_) -> list[Path]:
    """The .aseprite (its tags = one per action_direction); written from the sheet if the
    asset has none."""
    from .aseprite import write_aseprite
    n = src.file_name
    dst = out / f"{n}.aseprite"
    have = src.sheet.with_suffix(".aseprite") if src.sheet else src.folder / "sheet.aseprite"
    written: list[Path] = []
    if have.exists():
        _copy(have, dst, written)
        return written
    img = _rgba(src.image)
    frames = [_crop(img, f) for f in src.frames]
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    frames = [np.pad(f, ((0, h - f.shape[0]), (0, w - f.shape[1]), (0, 0))) for f in frames]
    write_aseprite(dst, frames, durations=[f.duration for f in src.frames], tags=src.tags)
    written.append(dst)
    return written


def hash_json(src: Source, image_name: str) -> dict:
    """Aseprite / TexturePacker 'hash' format (frames keyed by name)."""
    img_w, img_h = Image.open(src.image).size
    return {
        "frames": {f.name: {"frame": {"x": f.x, "y": f.y, "w": f.w, "h": f.h}, "rotated": False,
                            "trimmed": False,
                            "spriteSourceSize": {"x": 0, "y": 0, "w": f.w, "h": f.h},
                            "sourceSize": {"w": f.w, "h": f.h}, "duration": f.duration}
                   for f in src.frames},
        "meta": {"app": "pixel-pipeline", "image": image_name, "format": "RGBA8888",
                 "size": {"w": img_w, "h": img_h}, "scale": "1",
                 "frameTags": [{"name": t, "from": a, "to": b, "direction": "forward"}
                               for t, a, b in src.tags]},
    }


def _generic(src: Source, out: Path, single_frames: bool = True, **_) -> list[Path]:
    n = src.file_name
    written: list[Path] = []
    png = _copy(src.image, out / f"{n}.png", written)
    js = out / f"{n}.json"
    js.write_text(json.dumps(hash_json(src, png.name), indent=1), encoding="utf-8")
    written.append(js)
    if src.normals:
        _copy(src.normals, out / f"{n}_n.png", written)
    if src.icon and src.icon != src.image:
        _copy(src.icon, out / f"{n}_icon.png", written)
    if single_frames and src.sheet is not None:
        img = _rgba(src.image)
        fdir = out / "frames"
        fdir.mkdir(exist_ok=True)
        for f in src.frames:
            written.append(_save(_crop(img, f), fdir / f"{n}_{_yaml_name(f.name)}.png"))
    return written


_WRITERS = {"godot-2d": _godot_2d, "godot-hd2d": _godot_hd2d, "unity": _unity,
            "gamemaker": _gamemaker, "aseprite": _aseprite, "generic": _generic}


# --------------------------------------------------------------------------- public API

def export_asset(asset, preset: str, out_dir, **opts) -> list[Path]:
    """Write one asset for an engine into out_dir; returns the files written.

    opts: godot-2d res_path ("res://units/knight" -> absolute texture path, default relative);
    godot-hd2d ppu (out_dir = the Godot project; files go to units/<name>/); unity ppu (pixels
    per unit, default the project's tile size); gamemaker normals (also _n strips, default
    True); generic single_frames (default True)."""
    if preset not in _WRITERS:
        raise ValueError(f"Unknown export preset '{preset}' (use {', '.join(PRESETS)})")
    src = source(asset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return _WRITERS[preset](src, out, **opts)


def export_collection(collection, preset: str, out_dir, project=None, **opts) -> list[Path]:
    """A collection's name (assets.list_assets(collection=...)) or a list of assets -> one
    folder per asset inside out_dir + index.json. godot-hd2d: out_dir is the Godot project
    and each asset is its own units/<name>/ folder."""
    if isinstance(collection, str):
        from .assets import list_assets
        name = collection
        items = list_assets(collection=collection, project=project)
    else:
        name = None
        items = list(collection)
    if not items:
        raise ValueError(f"Nothing to export{f' in collection {name!r}' if name else ''}.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"collection": name, "preset": preset, "assets": []}
    written: list[Path] = []
    used: set[str] = set()
    for a in items:
        src = source(a)
        sub = src.file_name
        k = 2
        while sub in used:
            sub = f"{src.file_name}_{k}"
            k += 1
        used.add(sub)
        target = out if preset == "godot-hd2d" else out / sub
        files = export_asset(a, preset, target, **opts)
        written += files
        index["assets"].append({"name": src.name, "kind": src.kind,
                                "folder": sub if preset != "godot-hd2d" else f"units/{src.file_name}",
                                "files": [p.relative_to(out).as_posix() for p in files]})
    idx = out / "index.json"
    idx.write_text(json.dumps(index, indent=1), encoding="utf-8")
    written.append(idx)
    return written
