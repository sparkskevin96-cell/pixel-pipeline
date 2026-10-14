"""Export presets: one asset (or a whole collection) out of the studio, ready for an engine.

    export_asset(asset, "unity", "C:/MyGame/Assets/Sprites")      -> [written files]
    export_collection("starter kit", "godot-2d", "C:/MyGame/art")  -> one folder per asset + index.json

An asset is a library Character, an assets.Asset or just its folder. Sprite-sheet assets
(characters, and items/props made with a sheet) have sheet.png + sheet.json (a row per
action_direction, frames named "{action}_{dir} {i}"), maybe sheet_n.png (normal map) and
sheet_glow.png (emissive mask, glow.py; made on the fly for the export when missing),
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
    "gamemaker": "GameMaker (one _stripN PNG per animation + sprite_add loader)",
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
    glow: Path | None = None   # <sheet>_glow.png (emissive mask), if the asset has one
    loops: dict | None = None  # {animation: loops?} from sheet.json meta "loops" (effects)

    def is_loop(self, key: str) -> bool:
        """Does animation `key` loop: the sheet's own flag, else by action name."""
        if self.loops and key in self.loops:
            return bool(self.loops[key])
        return key.rsplit("_", 1)[0] not in ONCE

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
    frames, tags, normals, loops = [], [], None, None
    if sheet.exists() and sheet.with_suffix(".json").exists():
        meta = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
        loops = meta.get("meta", {}).get("loops")
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
    glow = sheet.with_name(sheet.stem + "_glow.png") if sheet is not None else None
    return Source(name, kind, folder, sheet, frames, tags, normals, icon, image,
                  glow if glow is not None and glow.exists() else None, loops)


def _glow_into(src: Source, dst: Path, written: list) -> Path | None:
    """The asset's glow map at dst: copied, or made by glow.auto_glow from the sheet (effects:
    all of it glows) when it has none. None when nothing glows."""
    if src.glow is not None:
        return _copy(src.glow, dst, written)
    if src.sheet is None:
        return None
    from .glow import write_glow_sheet
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + "_src.png")          # frames from the sheet's own .json
    shutil.copyfile(src.sheet, tmp)
    shutil.copyfile(src.sheet.with_suffix(".json"), tmp.with_suffix(".json"))
    try:
        out = write_glow_sheet(tmp, out=dst, force=True, effect=src.kind == "effect")
    finally:
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".json").unlink(missing_ok=True)
    if out is not None:
        written.append(out)
    return out


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
                          "loop": src.is_loop(key)}
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
    _glow_into(src, out / f"{n}_glow.png", written)
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
    """out = the Godot project folder: writes res://units/<name>/ like hd2d.add_unit (sheet,
    normal map, glow map) + <name>_material.tres (the HD-2D sprite ShaderMaterial with the
    three textures; the shader goes to res://shaders/sprite.gdshader if the project has none)."""
    from .hd2d import SPRITE_SHADER, add_unit
    if src.sheet is None:
        raise ValueError(f"'{src.name}' has no sprite sheet; HD-2D units need one.")
    if ppu is None:
        ppu = _project_style_value("ppu", 42.0)
    n = src.file_name
    folder = add_unit(out, n, src.sheet, ppu, glow=src.kind != "effect")
    if src.kind == "effect" and not (folder / f"{n}_glow.png").exists():
        _glow_into(src, folder / f"{n}_glow.png", [])
    shader = out / "shaders" / "sprite.gdshader"
    if not shader.exists():
        shader.parent.mkdir(parents=True, exist_ok=True)
        shader.write_text(SPRITE_SHADER, encoding="utf-8")
    (folder / f"{n}_material.tres").write_text(hd2d_material(n, (folder / f"{n}_n.png").exists(),
                                                             (folder / f"{n}_glow.png").exists()),
                                               encoding="utf-8")
    return sorted(p for p in folder.iterdir() if p.is_file())


def hd2d_material(name: str, normals: bool, glow: bool) -> str:
    """A ShaderMaterial .tres for res://units/<name>/: the sheet, its normal map and glow map
    (EMISSION: only those pixels bloom with WorldEnvironment glow). Set `frame` per frame."""
    base = f"res://units/{name}/{name}"
    ext = [("sh", "Shader", "res://shaders/sprite.gdshader"), ("sheet", "Texture2D", f"{base}.png")]
    if normals:
        ext.append(("nrm", "Texture2D", f"{base}_n.png"))
    if glow:
        ext.append(("glw", "Texture2D", f"{base}_glow.png"))
    lines = [f'[gd_resource type="ShaderMaterial" load_steps={len(ext) + 1} format=3]', ""]
    lines += [f'[ext_resource type="{t}" path="{p}" id="{i}"]' for i, t, p in ext]
    lines += ["", "[resource]", 'shader = ExtResource("sh")', 'shader_parameter/sheet = ExtResource("sheet")']
    if normals:
        lines += ['shader_parameter/normals = ExtResource("nrm")', "shader_parameter/has_normals = true"]
    if glow:
        lines += ['shader_parameter/glow = ExtResource("glw")', "shader_parameter/has_glow = true"]
    return "\n".join(lines) + "\n"


def _project_style_value(field_name: str, default):
    try:
        from .assets import current_project
        return getattr(current_project().style, field_name, default)
    except Exception:  # noqa: BLE001 - no project yet: the default
        return default


def unity_guid(text: str) -> str:
    """Stable 32-hex GUID (same asset -> same GUID, so re-exports keep Unity references)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, "pixel-pipeline:" + text).hex


def _unity_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Unity's own files use LF line ends (also on Windows)."""
    with open(path, "w", encoding=encoding, newline="\n") as fh:
        fh.write(text)


def _unity_platform(compression: int = 0, max_size: int = 8192) -> str:
    return "".join(f"""  - serializedVersion: 4
    buildTarget: {target}
    maxTextureSize: {max_size}
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


# Unity 6.3 refuses to import a texture over 8192 px on a side ("out of range height (got
# 9216 max supported 8192)", seen in batch mode), so taller sheets are repacked for Unity.
UNITY_MAX_TEXTURE = 8192


def unity_layout(frames: list[Frame], size: tuple[int, int]) -> tuple[list[Frame], tuple[int, int]] | None:
    """None when the sheet fits Unity as it is; else the frames repacked: the sheet's rows
    (frames sharing a y) stacked into columns at most UNITY_MAX_TEXTURE tall, side by side."""
    if max(size) <= UNITY_MAX_TEXTURE:
        return None
    rows: dict[int, list[Frame]] = {}
    for f in frames:
        rows.setdefault(f.y, []).append(f)
    moved: dict[int, Frame] = {}
    col_x = col_w = y = 0
    for ry in sorted(rows):
        fs = rows[ry]
        rh = max(f.y + f.h for f in fs) - ry
        rw = max(f.x + f.w for f in fs)
        if y and y + rh > UNITY_MAX_TEXTURE:
            col_x, col_w, y = col_x + col_w, 0, 0
        for f in fs:
            moved[id(f)] = Frame(f.name, col_x + f.x, y + (f.y - ry), f.w, f.h, f.duration)
        col_w, y = max(col_w, rw), y + rh
    out = [moved[id(f)] for f in frames]
    new_size = (max(f.x + f.w for f in out), max(f.y + f.h for f in out))
    if max(new_size) > UNITY_MAX_TEXTURE:
        raise ValueError(f"The sheet is {size[0]}x{size[1]} and does not fit Unity's "
                         f"{UNITY_MAX_TEXTURE} px texture limit even repacked. Export fewer "
                         "animations or directions.")
    return out, new_size


def _repack(path: Path, frames: list[Frame], moved: list[Frame], size: tuple[int, int], dst: Path) -> Path:
    img = _rgba(path)
    canvas = np.zeros((size[1], size[0], 4), np.uint8)
    for f, m in zip(frames, moved):
        canvas[m.y:m.y + m.h, m.x:m.x + m.w] = _crop(img, f)
    return _save(canvas, dst)


def unity_sprite_names(frames: list[Frame]) -> list[str]:
    """Unique Unity sprite names in frame order: 'walk_SE 3' -> walk_SE_3 (a clash gets _2...)."""
    out, used = [], set()
    for f in frames:
        base = _yaml_name(f.name)
        n, k = base, 2
        while n in used:
            n, k = f"{base}_{k}", k + 1
        used.add(n)
        out.append(n)
    return out


def unity_sprite_file_id(i: int) -> int:
    """The internalID (= fileID in references, e.g. from .anim clips) of the i-th sprite."""
    return 21300000 + 2 * i


def _unity_max_size(size: tuple[int, int]) -> int:
    big = max(size)
    if big > UNITY_MAX_TEXTURE:
        raise ValueError(f"The texture is {size[0]}x{size[1]}; Unity imports at most "
                         f"{UNITY_MAX_TEXTURE} px a side.")
    return 8192


def unity_meta(guid: str, size: tuple[int, int], frames: list[Frame] | None, *, ppu: float = 32,
               normal_map: bool = False, pivot: tuple[float, float] = (0.5, 0.0),
               normal_guid: str | None = None, emission_guid: str | None = None) -> str:
    """Unity TextureImporter .meta: point filter, no compression, no mipmaps. frames given ->
    Sprite (Multiple) with one sprite per frame (Unity rects start bottom-left, so y is
    flipped); None -> a single sprite; normal_map -> texture type Normal map (linear);
    normal_guid -> that texture is the sprites' _NormalMap secondary texture (URP 2D lights);
    emission_guid -> the glow map as their _EmissionMap secondary texture."""
    w, h = size
    max_size = _unity_max_size(size)
    sprites = ""
    table = []
    if frames and not normal_map:
        for i, (f, name) in enumerate(zip(frames, unity_sprite_names(frames))):
            sid = uuid.uuid5(uuid.NAMESPACE_URL, f"{guid}/{name}").hex
            internal = unity_sprite_file_id(i)
            sprites += f"""    - serializedVersion: 2
      name: {name}
      rect:
        serializedVersion: 2
        x: {f.x}
        y: {h - f.y - f.h}
        width: {f.w}
        height: {f.h}
      alignment: 7
      pivot: {{x: {pivot[0]:g}, y: {pivot[1]:g}}}
      border: {{x: 0, y: 0, z: 0, w: 0}}
      customData:
      outline: []
      physicsShape: []
      tessellationDetail: -1
      bones: []
      spriteID: {sid}
      internalID: {internal}
      vertices: []
      indices:
      edges: []
      weights: []
"""
            table.append((name, internal))
    # Unity writes the name table sorted by name (ordinal)
    names = "".join(f"      {nm}: {i}\n" for nm, i in sorted(table))
    secondary = "    secondaryTextures: []\n"
    extra = [(g, nm) for g, nm in ((normal_guid, "_NormalMap"), (emission_guid, "_EmissionMap")) if g]
    if extra and not normal_map:
        secondary = "    secondaryTextures:\n" + "".join(
            f"    - texture: {{fileID: 2800000, guid: {g}, type: 3}}\n      name: {nm}\n" for g, nm in extra)
    sprite_mode = 0 if normal_map else (2 if frames else 1)
    texture_type = 1 if normal_map else 8
    return f"""fileFormatVersion: 2
guid: {guid}
TextureImporter:
  internalIDToNameTable: []
  externalObjects: {{}}
  serializedVersion: 13
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
  maxTextureSize: {max_size}
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
  spritePivot: {{x: {pivot[0]:g}, y: {pivot[1]:g}}}
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
{_unity_platform(0, max_size)}  spriteSheet:
    serializedVersion: 2
    sprites:{(chr(10) + sprites) if sprites else " []" + chr(10)}    outline: []
    customData:
    physicsShape: []
    bones: []
    spriteID:
    internalID: 0
    vertices: []
    indices:
    edges: []
    weights: []
{secondary}    spriteCustomMetadata:
      entries: []
    nameFileIdTable:{(chr(10) + names) if names else " {}" + chr(10)}  mipmapLimitGroupName:
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
  2. Sprite names are <action>_<direction>_<frame>, e.g. walk_SE_0. The clips/ folder has
     an Animation clip per <action>_<direction> (drag one onto a GameObject with a
     SpriteRenderer; Unity adds an Animator). Frame times (ms) are in {name}_frames.json.
  3. Pixel-perfect: add a Pixel Perfect Camera (URP 2D) with Assets Pixels Per Unit = {ppu:g}.
  4. Normal map: the sheet's sprites already carry {npng} as their _NormalMap secondary
     texture, so a Sprite-Lit material (URP 2D) lights them with it.{glow_line}
"""


def _gcd_ms(frames: list[Frame]) -> int:
    """The clip's sample step: every frame time is a whole number of samples."""
    import math
    g = 0
    for f in frames:
        g = math.gcd(g, max(1, int(f.duration)))
    return max(1, g)


def unity_clip_plan(src: Source) -> list[dict]:
    """One clip per animation of 2+ frames: {name, sprites, ids (sprite fileIDs), times (s,
    from each frame's duration), length, fps, loop}."""
    if src.sheet is None:
        return []
    names = unity_sprite_names(src.frames)
    index = {id(f): i for i, f in enumerate(src.frames)}
    plans = []
    for key, fs in src.groups():
        if len(fs) < 2:
            continue
        idx = [index[id(f)] for f in fs]
        times, t = [], 0
        for f in fs:
            times.append(round(t / 1000, 6))
            t += f.duration
        plans.append({"name": _yaml_name(key), "sprites": [names[i] for i in idx],
                      "ids": [unity_sprite_file_id(i) for i in idx], "times": times,
                      "length": round(t / 1000, 6), "fps": round(1000 / _gcd_ms(fs), 6),
                      "loop": src.is_loop(key)})
    return plans


def unity_anim(plan: dict, sheet_guid: str) -> str:
    """A sprite AnimationClip (.anim, text YAML) that swaps SpriteRenderer.sprite through the
    sheet's sprites at the sheet's frame times (the format Unity itself saves)."""
    refs = [f"{{fileID: {i}, guid: {sheet_guid}, type: 3}}" for i in plan["ids"]]
    times = list(plan["times"])
    step = 1 / plan["fps"]
    if plan["length"] - times[-1] > step + 1e-6:
        # Unity holds the last key for one sample (clip length = last key + 1/fps), so a longer
        # last frame gets the same sprite keyed again one sample before the end
        times.append(round(plan["length"] - step, 6))
        refs = refs + refs[-1:]
    keys = "".join(f"    - time: {t:g}\n      value: {r}\n" for t, r in zip(times, refs))
    mapping = "".join(f"    - {r}\n" for r in refs)
    return f"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {{fileID: 0}}
  m_PrefabInstance: {{fileID: 0}}
  m_PrefabAsset: {{fileID: 0}}
  m_Name: {plan["name"]}
  serializedVersion: 7
  m_Legacy: 0
  m_Compressed: 0
  m_UseHighQualityCurve: 1
  m_RotationCurves: []
  m_CompressedRotationCurves: []
  m_EulerCurves: []
  m_PositionCurves: []
  m_ScaleCurves: []
  m_FloatCurves: []
  m_PPtrCurves:
  - serializedVersion: 2
    curve:
{keys}    attribute: m_Sprite
    path: 
    classID: 212
    script: {{fileID: 0}}
    flags: 2
  m_SampleRate: {plan["fps"]:.9g}
  m_WrapMode: 0
  m_Bounds:
    m_Center: {{x: 0, y: 0, z: 0}}
    m_Extent: {{x: 0, y: 0, z: 0}}
  m_ClipBindingConstant:
    genericBindings:
    - serializedVersion: 2
      path: 0
      attribute: 0
      script: {{fileID: 0}}
      typeID: 212
      customType: 23
      isPPtrCurve: 1
      isIntCurve: 0
      isSerializeReferenceCurve: 0
    pptrCurveMapping:
{mapping}  m_AnimationClipSettings:
    serializedVersion: 2
    m_AdditiveReferencePoseClip: {{fileID: 0}}
    m_AdditiveReferencePoseTime: 0
    m_StartTime: 0
    m_StopTime: {plan["length"]:.9g}
    m_OrientationOffsetY: 0
    m_Level: 0
    m_CycleOffset: 0
    m_HasAdditiveReferencePose: 0
    m_LoopTime: {1 if plan["loop"] else 0}
    m_LoopBlend: 0
    m_LoopBlendOrientation: 0
    m_LoopBlendPositionY: 0
    m_LoopBlendPositionXZ: 0
    m_KeepOriginalOrientation: 0
    m_KeepOriginalPositionY: 1
    m_KeepOriginalPositionXZ: 0
    m_HeightFromFeet: 0
    m_Mirror: 0
  m_EditorCurves: []
  m_EulerEditorCurves: []
  m_HasGenericRootTransform: 0
  m_HasMotionFloatCurves: 0
  m_Events: []
"""


def _unity(src: Source, out: Path, ppu: float | None = None, clips: bool = True, **_) -> list[Path]:
    """<name>.png + .meta (sliced: Sprite Multiple), <name>_n.png + .meta (Normal map, linked to
    the sprites as their _NormalMap secondary texture), clips/<action_dir>.anim (+ .meta) when
    clips, <name>_frames.json, <name>_README.txt."""
    n = src.file_name
    written: list[Path] = []
    if ppu is None:
        ppu = float(_project_style_value("tile", 32))
    sheet_frames = src.frames if src.sheet is not None else None
    size = Image.open(src.image).size
    layout = unity_layout(sheet_frames, size) if sheet_frames else None
    npng = f"{n}_n.png"
    gpng = f"{n}_glow.png"
    glow = _glow_into(src, out / gpng, []) if src.sheet is not None else None
    if layout:                                   # too tall for Unity: rows into columns
        sheet_frames, size = layout
        png = _repack(src.image, src.frames, sheet_frames, size, out / f"{n}.png")
        written.append(png)
        if src.normals:
            written.append(_repack(src.normals, src.frames, sheet_frames, size, out / npng))
        if glow is not None:
            written.append(_repack(glow, src.frames, sheet_frames, size, out / gpng))
    else:
        png = _copy(src.image, out / f"{n}.png", written)
        if src.normals:
            _copy(src.normals, out / npng, written)
        if glow is not None:
            written.append(glow)
    guid = unity_guid(f"{src.kind}/{src.name}")
    normal_guid = unity_guid(f"{src.kind}/{src.name}/normals") if src.normals else None
    emission_guid = unity_guid(f"{src.kind}/{src.name}/emission") if glow is not None else None
    meta = out / f"{n}.png.meta"
    _unity_write(meta, unity_meta(guid, size, sheet_frames, ppu=ppu, normal_guid=normal_guid,
                                  emission_guid=emission_guid))
    written.append(meta)
    glow_line = ""
    if glow is not None:
        gmeta = out / f"{gpng}.meta"
        _unity_write(gmeta, unity_meta(emission_guid, Image.open(out / gpng).size, None, ppu=ppu))
        written.append(gmeta)
        glow_line = (f"\n  5. Glow: {gpng} (white where the sprite gives off light, alpha = how much) is the\n"
                     "     sprites' _EmissionMap secondary texture: in a Shader Graph sprite shader add a\n"
                     "     Texture2D property named _EmissionMap, multiply the sprite colour by its alpha and\n"
                     "     feed that (x an intensity) to Emission, then Bloom in a URP Volume blooms only those\n"
                     "     pixels (fire, crystals, lanterns, eyes).")
    normal_line = ""
    if src.normals:
        nmeta = out / f"{npng}.meta"
        _unity_write(nmeta, unity_meta(normal_guid, Image.open(out / npng).size, None,
                                       ppu=ppu, normal_map=True))
        written.append(nmeta)
        normal_line = (f"  {npng}          normal map (OpenGL, green up), imported as a Normal map\n"
                       f"                   and already the sprites' _NormalMap secondary texture\n")
    if src.icon and src.icon != src.image:
        icon = _copy(src.icon, out / f"{n}_icon.png", written)
        imeta = out / f"{icon.name}.meta"
        _unity_write(imeta, unity_meta(unity_guid(f"{src.kind}/{src.name}/icon"), Image.open(icon).size,
                                    None, ppu=ppu), encoding="utf-8")
        written.append(imeta)
    plans = unity_clip_plan(src) if clips else []
    clip_line = ""
    if plans:
        cdir = out / "clips"
        cdir.mkdir(exist_ok=True)
        for plan in plans:
            anim = cdir / f"{plan['name']}.anim"
            _unity_write(anim, unity_anim(plan, guid), encoding="utf-8")
            am = cdir / f"{anim.name}.meta"
            clip_guid = unity_guid(f"{src.kind}/{src.name}/clip/{plan['name']}")
            _unity_write(am, f"fileFormatVersion: 2\nguid: {clip_guid}\n"
                          "NativeFormatImporter:\n  externalObjects: {}\n  mainObjectFileID: 7400000\n"
                          "  userData:\n  assetBundleName:\n  assetBundleVariant:\n", encoding="utf-8")
            written += [anim, am]
        clip_line = (f"  clips/           {len(plans)} Animation clips, one per <action>_<direction>"
                     " (frame times from the sheet)\n")
    names = unity_sprite_names(src.frames) if src.sheet is not None else [png.stem]
    by_frame = {id(f): nm for f, nm in zip(src.frames, names)}
    frames_json = out / f"{n}_frames.json"
    frames_json.write_text(json.dumps({
        "sprites": [{"name": nm, "duration_ms": f.duration} for f, nm in zip(src.frames, names)],
        "animations": [{"name": k, "sprites": [by_frame[id(f)] for f in fs],
                        "frame_ms": fs[0].duration} for k, fs in src.groups()]}, indent=1),
        encoding="utf-8")
    written.append(frames_json)
    readme = out / f"{n}_README.txt"
    readme.write_text(UNITY_README.format(name=n, png=png.name, count=len(src.frames), ppu=ppu,
                                          normal_line=normal_line + clip_line, npng=npng,
                                          glow_line=glow_line),
                      encoding="utf-8")
    written.append(readme)
    return written


def gamemaker_name(name: str) -> str:
    """A GameMaker resource name: letters, digits, underscores, not starting with a digit,
    with the usual spr_ prefix ('Health Potion' -> spr_health_potion)."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "asset"
    return s if s.startswith("spr_") else f"spr_{s}"


GAMEMAKER_README = """{name} - exported from pixel-pipeline for GameMaker
=============================================

Every animation is one horizontal strip, named <sprite>_strip<frames>.png. GameMaker cuts a
file whose name ends in _strip<N> into N frames by itself.

In the IDE (any GameMaker since 2.3)
  1. Create a Sprite, name it as the file without _strip<N> (e.g. {example}), click Import
     and pick {example_file}: it arrives as {example_frames} frames.
  2. Origin: set it to Custom and type the "origin" x, y from {json}
     (bottom centre: the feet; x = half the frame width, y = the row under the feet).
  3. Speed: Frames per second = the sprite's "fps" in {json}.
  4. Collision mask: Automatic is fine (frames are tight around the character).
  _normal_strip files are the matching normal maps (OpenGL, green up) for your own lighting
  shader; GameMaker does not use them by itself.

Without importing (sprite_add at run time)
  Put the PNGs in Included Files (datafiles) and call {gml_func}() from {gml}: it adds every
  strip with sprite_add (frame count + origin) and sets its speed, and returns a struct
  keyed by animation name, e.g. sprites[$ "{example_key}"]. (On HTML5 sprite_add loads
  asynchronously.)

Not included: a ready .yy sprite resource. GameMaker's project format changes between
releases and was not available to test, so import the strips as above.
"""


def _gamemaker(src: Source, out: Path, normals: bool = True, **_) -> list[Path]:
    """One horizontal strip per action_direction: spr_<name>_<action>_<dir>_strip<N>.png
    (GameMaker slices _stripN itself on import; direction keeps its case:
    spr_knight_walk_SE_strip8.png), frames bottom-centre aligned; the normal map as
    spr_<name>_<action>_<dir>_normal_strip<N>.png; <name>_gamemaker.json (frames, fps,
    origin = feet, bbox, per-frame ms), <name>_load_sprites.gml (sprite_add loader) and
    <name>_GameMaker_README.txt."""
    n = src.file_name
    spr = gamemaker_name(n)
    img = _rgba(src.image)
    nimg = _rgba(src.normals) if (normals and src.normals) else None
    written: list[Path] = []
    summary = {"name": n, "origin": "bottom centre (x = frame width // 2, y = row under the feet)",
               "sprites": []}
    groups = src.groups()
    single = len(groups) == 1 and len(groups[0][1]) == 1
    gml = []
    for key, fs in groups:
        w, h = max(f.w for f in fs), max(f.h for f in fs)
        base = spr if single else f"{spr}_{_yaml_name(key)}"
        files = []
        colour_strip = None
        for layer, suffix in ((img, ""), (nimg, "_normal")):
            if layer is None:
                continue
            strip = np.zeros((h, w * len(fs), 4), np.uint8)
            for i, f in enumerate(fs):
                c = _crop(layer, f)
                x0 = i * w + (w - c.shape[1]) // 2          # bottom-centre: feet stay put
                strip[h - c.shape[0]:h, x0:x0 + c.shape[1]] = c
            if colour_strip is None:
                colour_strip = strip
            name = f"{base}{suffix}.png" if len(fs) == 1 else f"{base}{suffix}_strip{len(fs)}.png"
            written.append(_save(strip, out / name))
            files.append(name)
        cells = [Frame(f.name, i * w, 0, w, h) for i, f in enumerate(fs)]
        foot = _foot_row(colour_strip, cells)
        alpha = colour_strip[..., 3] >= 128
        bbox = None
        if alpha.any():
            per = alpha.reshape(h, len(fs), w).any(1)            # union of all frames
            ys, xs = np.nonzero(per)
            bbox = {"left": int(xs.min()), "top": int(ys.min()),
                    "right": int(xs.max()), "bottom": int(ys.max())}
        total = sum(f.duration for f in fs)
        fps = round(1000 * len(fs) / max(1, total), 3)
        summary["sprites"].append({
            "sprite": base, "animation": key, "file": files[0],
            "normal_map": files[1] if len(files) > 1 else None,
            "frames": len(fs), "frame_w": w, "frame_h": h, "fps": fps,
            "frame_ms": [f.duration for f in fs], "origin": [w // 2, foot], "bbox": bbox})
        gml.append(f'    _s[$ "{key}"] = sprite_add(_dir + "{files[0]}", {len(fs)}, false, false, '
                   f'{w // 2}, {foot});\n'
                   f'    sprite_set_speed(_s[$ "{key}"], {fps:g}, spritespeed_framespersecond);\n')
    if src.icon and src.icon != src.image:
        _copy(src.icon, out / f"{spr}_icon.png", written)
        summary["icon"] = f"{spr}_icon.png"
    js = out / f"{n}_gamemaker.json"
    js.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    written.append(js)
    func = f"{n}_load_sprites" if not n[0].isdigit() else f"load_{n}_sprites"
    script = out / f"{n}_load_sprites.gml"
    script.write_text(
        f"/// {func}([dir]) -> struct of sprites for {src.name}, keyed by animation name.\n"
        f"/// The PNGs must be in Included Files; dir is their folder inside it (\"\" = top).\n"
        f"function {func}(_dir = \"\") {{\n"
        f"    if (_dir != \"\" && string_char_at(_dir, string_length(_dir)) != \"/\") _dir += \"/\";\n"
        f"    var _s = {{}};\n" + "".join(gml) + "    return _s;\n}\n", encoding="utf-8")
    written.append(script)
    first = summary["sprites"][0]
    readme = out / f"{n}_GameMaker_README.txt"
    readme.write_text(GAMEMAKER_README.format(
        name=n, example=first["sprite"], example_file=first["file"], example_frames=first["frames"],
        example_key=first["animation"], json=js.name, gml=script.name, gml_func=func), encoding="utf-8")
    written.append(readme)
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
    _glow_into(src, out / f"{n}_glow.png", written)
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
    per unit, default the project's tile size), clips (an .anim per animation, default True);
    gamemaker normals (also _normal strips, default True); generic single_frames (default
    True)."""
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
