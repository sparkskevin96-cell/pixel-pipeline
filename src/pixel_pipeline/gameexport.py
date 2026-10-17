"""Export assets straight into a game project (Godot 4 or Unity) + a manifest the game loads by name.

    export_to_game(["character/knight", "tileset/grass-tiles"], "godot", "C:/Games/MyGame",
                   subdir="art")
    -> C:/Games/MyGame/art/characters/knight/knight.png/.tres/_n.png/_facing.gd (+ .import)
       C:/Games/MyGame/art/pixel_pipeline_manifest.json

Per kind, the engine's own files (the same writers as the Library's Export):
    character, item, prop, effect, portrait   godot: SpriteFrames .tres + sheet (+ normal/glow maps,
                                              facing script); unity: sliced sheet + .meta + .anim
    weather, water, background                exports.KIND_EXPORTS (overlay / particle scenes,
                                              shore TileSet, Parallax2D; Unity layers + .meta)
    tileset                                   godot: tileset.png + tileset.tres (+ .tsx, example
                                              map); unity: the sliced sheet
    ui                                        uikit.export_kit (theme.tres / Unity 9-slices)
    card                                      cards.export_cards (Card.tscn / CardView.cs)
    map                                       maplib.export_map (the project's map target)
    3D models (models=True)                   godot-3d-model / unity-3d-model presets (model_game)

Godot: every PNG we write gets a pixel-art .import (lossless, no mipmaps, no VRAM compression
when used in 3D) unless one exists (Godot's own, with its uid, is kept); project.godot gets the
nearest-neighbour canvas filter if it has none (pixel_filter=True). Unity: subdir lives under
Assets/; .meta files come from the writers (point filter, no compression) and the manifest gets
one (TextAsset).

The manifest (pixel_pipeline_manifest.json in the subdir) maps asset id ("kind/folder") -> name,
files, frame size, animations (directions, frames, ms, loop), anchor (feet), ppu, tile size,
normal / glow maps. Re-export is incremental: an asset whose source files and options didn't
change (fingerprint) and whose exported files are all there is skipped. Drafts (characters
waiting for an OK on their front) are skipped with a reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

MANIFEST = "pixel_pipeline_manifest.json"
VERSION = 1                      # bump when the export layout changes (forces re-export)
ENGINES = ("godot", "unity")
SKIP_DIRS = {"keys", "attempts", "agent_preview", "__pycache__"}
SHEET_KINDS = ("character", "item", "prop", "effect", "portrait")
MODEL_KINDS = ("character", "item", "prop")

GODOT_TEXTURE_IMPORT = """[remap]

importer="texture"
type="CompressedTexture2D"

[deps]

source_file="{res}"

[params]

compress/mode=0
compress/high_quality=false
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=0
"""


# --------------------------------------------------------------------------- engine projects

def find_godot() -> str | None:
    """Godot 4 for headless checks: $GODOT_BIN, PATH, or winget's (the console exe first)."""
    env = os.environ.get("GODOT_BIN")
    if env and Path(env).exists():
        return env
    for name in ("godot", "godot4", "Godot"):
        hit = shutil.which(name)
        if hit:
            return hit
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.exists():
        found = sorted(winget.glob("GodotEngine.GodotEngine*/Godot_v4*_win64*.exe"))
        console = [p for p in found if "console" in p.name]
        if console or found:
            return str((console or found)[-1])
    return None


def check_project(engine: str, project_path) -> Path:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}")
    root = Path(project_path).expanduser()
    if not root.is_absolute():
        raise ValueError("project_path must be an absolute folder path")
    if engine == "godot" and not (root / "project.godot").exists():
        raise ValueError(f"{root} is not a Godot project (no project.godot). Create the project in "
                         "Godot first, or pass the folder that has project.godot.")
    if engine == "unity" and not (root / "Assets").is_dir():
        raise ValueError(f"{root} is not a Unity project (no Assets folder).")
    return root


def _subdir(engine: str, subdir: str) -> str:
    s = str(subdir or "").replace("\\", "/").strip("/")
    if s.startswith("res://"):
        s = s[6:].strip("/")
    if engine == "unity":
        if not s:
            s = "Assets/PixelPipeline"
        elif s.split("/")[0] != "Assets":
            s = "Assets/" + s
    elif not s:
        s = "art"
    if ".." in s.split("/"):
        raise ValueError("subdir must stay inside the project")
    return s


def ensure_godot_pixel_filter(root: Path) -> bool:
    """Add rendering/textures/canvas_textures/default_texture_filter=0 (Nearest) to project.godot
    if it doesn't set one. True when it was added."""
    p = root / "project.godot"
    text = p.read_text(encoding="utf-8")
    if "default_texture_filter" in text:
        return False
    line = "textures/canvas_textures/default_texture_filter=0"
    m = re.search(r"^\[rendering\]\s*$", text, re.M)
    if m:
        text = text[:m.end()] + "\n\n" + line + text[m.end():]
    else:
        text = text.rstrip("\n") + "\n\n[rendering]\n\n" + line + "\n"
    p.write_text(text, encoding="utf-8")
    return True


# --------------------------------------------------------------------------- assets

def asset_id(asset) -> str:
    kind = getattr(asset, "kind", "asset")
    if kind in ("person", "animal"):
        kind = "character"
    return f"{kind}/{Path(asset.folder).name}"


def fingerprint(asset, engine: str, options: dict) -> str:
    """Source files (path, size, mtime) + export options: what makes an export stale."""
    folder = Path(asset.folder)
    h = hashlib.sha1(f"{VERSION}|{engine}|{json.dumps(options, sort_keys=True, default=str)}".encode())
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(part in SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        if not options.get("models") and rel.parts[0] == "model_game":
            continue
        st = p.stat()
        h.update(f"{rel.as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _map_target(engine: str, project) -> str:
    from .maplib import defaults_for
    try:
        target = defaults_for(project.style)[1]
    except Exception:  # noqa: BLE001
        target = "godot-tilemap"
    if engine == "godot":
        return target if target.startswith("godot") else "godot-tilemap"
    return target if target in ("tiled", "json") else "tiled"


def _write_engine_files(asset, kind: str, engine: str, target: Path, res_dir: str,
                        options: dict, project) -> list[Path]:
    from . import exports
    preset = options.get("preset")
    if kind == "ui":
        from .uikit import export_kit
        return export_kit(asset, target, engine)
    if kind == "card":
        from .cards import export_cards
        return export_cards([asset], target, engine, res_path=res_dir)
    if kind == "map":
        from .maplib import export_map
        return export_map(asset, options.get("map_target") or _map_target(engine, project), target)
    if kind == "tileset" and engine == "godot":
        folder = Path(asset.folder)
        target.mkdir(parents=True, exist_ok=True)
        out = []
        for name in ("tileset.png", "tileset.tres", "tileset.tsx", "tileset.json", "example.tmj",
                     "example.png", "texture.png"):
            if (folder / name).exists():
                shutil.copyfile(folder / name, target / name)
                out.append(target / name)
        if not out:
            raise ValueError(f"{asset.name} has no tileset files")
        return out
    if engine == "godot":
        return exports.export_asset(asset, preset or "godot-2d", target)
    return exports.export_asset(asset, preset or "unity", target,
                                **({"ppu": options["ppu"]} if options.get("ppu") else {}))


def _sheet_info(src_json: Path, image: Path, kind: str = "character") -> dict:
    """frame size, animations {action: {directions, frames, ms, loop}}, directions, anchor."""
    from .exports import ONCE, _foot_row, _frames, group_key
    from .turnaround import COMPASS16
    meta = json.loads(src_json.read_text(encoding="utf-8"))
    frames = _frames(meta)
    loops = (meta.get("meta") or {}).get("loops") or {}
    info: dict = {}
    if not frames:
        return info
    info["frame"] = {"w": frames[0].w, "h": frames[0].h}
    info["frame_count"] = len(frames)
    groups: dict[str, list] = {}
    for f in frames:
        groups.setdefault(group_key(f.name), []).append(f)
    anims: dict[str, dict] = {}
    dirs: list[str] = []
    for key, fs in groups.items():
        action, _, d = key.rpartition("_")
        if key in COMPASS16:                    # a still sprite per direction (items, props)
            action, d = "default", key
        elif not action or d not in COMPASS16:
            action, d = key, ""
        a = anims.setdefault(action, {"directions": [], "frames": len(fs),
                                      "ms": [f.duration for f in fs]})
        if d and d not in a["directions"]:
            a["directions"].append(d)
        if d and d not in dirs:
            dirs.append(d)
        loop = loops.get(key, loops.get(action))
        a["loop"] = bool(loop) if loop is not None else action not in ONCE
    for a in anims.values():
        if len(set(a["ms"])) == 1:
            a["ms"] = a["ms"][0]
        if not a["directions"]:
            del a["directions"]
    info["animations"] = anims
    if dirs:
        info["directions"] = dirs
    if kind not in ("character", "item", "prop"):
        return info
    try:
        import numpy as np
        img = np.asarray(Image.open(image).convert("RGBA"))
        info["anchor"] = {"x": frames[0].w // 2, "y": _foot_row(img, frames[:64]),
                          "note": "feet: bottom-centre of the frame's standing pixels"}
    except (OSError, ValueError):
        pass
    return info


def _describe(asset, kind: str, files: list[Path], base: Path, root: Path, engine: str,
              project, target: Path) -> dict:
    """The manifest entry's files + facts for one exported asset."""
    from .exports import file_name
    key = file_name(asset.name)
    rel = [p.relative_to(base).as_posix() for p in files if p.exists()]
    inside = {p.relative_to(base).as_posix(): p.relative_to(target).as_posix()
              for p in files if p.exists() and target in p.parents}

    def res(p: str) -> str:
        full = (base / p).relative_to(root).as_posix()
        return f"res://{full}" if engine == "godot" else full

    def pick(test) -> str | None:
        hits = [r for r, own in inside.items() if test(own.lower())]
        hits.sort(key=lambda r: (inside[r].count("/"), len(inside[r])))
        return hits[0] if hits else None

    roles: dict[str, str | None] = {
        "texture": pick(lambda n: n in (f"{key}.png", "tileset.png", "sheet.png"))
        or pick(lambda n: n.endswith(".png") and "/" not in n and not n.endswith(("_n.png", "_glow.png", "_icon.png"))
                and not n.startswith(("example", "preview"))),
        "normal_map": pick(lambda n: n.endswith(("_n.png", "_normalmap.png"))),
        "glow_map": pick(lambda n: n.endswith("_glow.png")),
        "icon": pick(lambda n: n.endswith("_icon.png")),
        "facing_script": pick(lambda n: n.endswith("_facing.gd")),
        "tileset": pick(lambda n: n == "tileset.tres") if kind == "tileset" else None,
        "theme": pick(lambda n: n == "theme.tres"),
        "material": pick(lambda n: n.endswith("_material.tres")),
        "sprite_frames": pick(lambda n: n == f"{key}.tres" or (n.endswith(".tres") and "/" not in n
                                                               and n not in ("theme.tres", "tileset.tres")
                                                               and not n.endswith(("_material.tres", "_add.tres", "_glow.tres"))))
        if kind in SHEET_KINDS + ("weather", "water") else None,
        "scene": pick(lambda n: n.endswith(".tscn")),
        "model": pick(lambda n: n.endswith((".glb", ".ppmodel"))),
        "data": pick(lambda n: n.endswith(".json") and n != MANIFEST),
        "unity_controller": pick(lambda n: n.endswith(".controller")),
    }
    roles = {k: v for k, v in roles.items() if v}
    entry: dict = {"files": roles, "all_files": rel}
    if engine == "godot":
        entry["res"] = {k: res(v) for k, v in roles.items()}
    folder = Path(asset.folder)
    sheet = folder / "sheet.png"
    if kind not in ("tileset", "ui", "map", "card", "background") and sheet.exists() and \
            sheet.with_suffix(".json").exists():
        entry.update(_sheet_info(sheet.with_suffix(".json"), sheet, kind))
    st = project.style
    from .assets import pixel_grid
    entry["ppu"] = float(pixel_grid(st)["ppu"])      # one pixel grid: sprite texels = tile texels
    entry["tile"] = int(st.tile)
    extra = getattr(asset, "extra", {}) or {}
    if kind == "tileset":
        entry["tileset"] = {k: extra.get(k) for k in ("terrain", "base", "camera", "set", "tile", "role")
                            if extra.get(k) is not None}
    if kind == "map":
        try:
            m = json.loads((folder / "map.json").read_text(encoding="utf-8"))
            entry["map"] = {k: m.get(k) for k in ("cols", "rows", "template") if k in m}
        except (OSError, ValueError):
            pass
    if kind == "character":
        entry["body"] = extra.get("body", "person")
        entry.update(_motion_info(asset, sheet, entry, entry["ppu"]))
        q = extra.get("quality") or {}
        if q:
            entry["quality"] = {k: q.get(k) for k in ("passed", "total", "failed") if k in q}
            entry["weak_directions"] = list(q.get("failed") or [])
    if kind == "effect":
        entry["effect"] = {k: extra.get(k) for k in ("effect", "theme", "blend") if extra.get(k)}
    entry["has_normal_map"] = "normal_map" in roles
    entry["has_glow_map"] = "glow_map" in roles
    return entry


def _motion_info(asset, sheet: Path, entry: dict, ppu: float) -> dict:
    """World scale + stride for a character (motion.py): px_scale (how much bigger than the
    library's 64 px-person scale its sheet is drawn), world_ppu (= ppu x px_scale: sprite px
    per tile so every character shares one world scale), body_px, and per locomotion clip
    animations[clip].stride {px, units (tiles), cycle_ms}: play it at speed / units cycles/s."""
    from .motion import px_scale, sheet_stride
    out: dict = {}
    fh = (entry.get("frame") or {}).get("h")
    if fh:
        k = px_scale(int(fh), getattr(asset, "kind", "person") if getattr(asset, "kind", "") in
                     ("person", "animal") else "person", getattr(asset, "size", None))
        out["px_scale"] = round(k, 4)
        out["world_ppu"] = round(ppu * k, 3)
    stride, body = sheet_stride(sheet.with_suffix(".json")) if sheet.exists() else ({}, None)
    if body:
        out["body_px"] = body
    anims = entry.get("animations") or {}
    wp = out.get("world_ppu", ppu)
    for clip, v in stride.items():
        if clip in anims and v.get("px"):
            anims[clip]["stride"] = {"px": v["px"], "units": round(float(v["px"]) / wp, 4),
                                     "cycle_ms": v.get("cycle_ms"), "source": v.get("source", "")}
    return out


RUNTIME_FILES = ("pp_motion.gd", "pp_manifest.gd", "pp_sprite3d.gdshader")


def install_runtime(engine: str, base: Path) -> list[Path]:
    """The game runtime that ships with an export: Godot gets res://<subdir>/pp_runtime/ (pp_motion.gd:
    acceleration, turning through the 16 facings, stride-matched walk/run, settle, hops, hit-stop,
    shake, critically damped camera; pp_manifest.gd; pp_sprite3d.gdshader: billboard sprites whose
    depth is their feet) + motion.json (the tunables); Unity gets motion.json (the same keys) for
    a C# port of docs/FOR_CLAUDE.md's algorithms."""
    from . import godot_runtime
    from .motion import write_motion_json
    out = base / "pp_runtime"
    files = []
    if engine == "godot":
        files += godot_runtime.install(base, [n for n in RUNTIME_FILES if (godot_runtime.HERE / n).exists()],
                                       subdir="pp_runtime")
    files.append(write_motion_json(out))
    return files


def _godot_imports(files: list[Path], root: Path) -> list[Path]:
    """A pixel-art .import beside every PNG that has none (Godot's own are kept: they carry uids).
    BMFont pages are left alone: the .fnt importer claims them (importer "skip")."""
    out = []
    pages = set()
    for f in files:
        if f.suffix.lower() == ".fnt" and f.exists():
            for m in re.finditer(r'file="([^"]+)"', f.read_text(encoding="utf-8", errors="replace")):
                pages.add((f.parent / m.group(1)).resolve())
    for p in files:
        if p.suffix.lower() != ".png" or not p.exists() or p.resolve() in pages:
            continue
        imp = p.with_name(p.name + ".import")
        if imp.exists():
            continue
        res = "res://" + p.relative_to(root).as_posix()
        imp.write_text(GODOT_TEXTURE_IMPORT.format(res=res), encoding="utf-8")
        out.append(imp)
    return out


TEXT_SUFFIXES = (".tscn", ".tres", ".gd", ".gdshader", ".cfg")


def embed_standalone(target: Path, root: Path, files: list[Path]) -> list[Path]:
    """Some writers (map exports) make a whole Godot project. Inside a game, a nested
    project.godot makes Godot skip the folder, and their res:// paths point at the project
    root: drop the project.godot and move every res:// path under the target folder."""
    pg = target / "project.godot"
    if not pg.exists() or pg == root / "project.godot":
        return files
    pg.unlink()
    rel = target.relative_to(root).as_posix()
    rx = re.compile(r"res://(?!" + re.escape(rel) + r"/)")
    for p in target.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            text = p.read_text(encoding="utf-8", errors="replace")
            new = rx.sub(f"res://{rel}/", text)
            if new != text:
                p.write_text(new, encoding="utf-8")
    return [f for f in files if Path(f) != pg]


def _unity_text_meta(path: Path, key: str) -> Path:
    from .exports import unity_guid
    meta = path.with_name(path.name + ".meta")
    if not meta.exists():
        meta.write_text(f"fileFormatVersion: 2\nguid: {unity_guid(key)}\nTextScriptImporter:\n"
                        "  externalObjects: {}\n  userData: \n  assetBundleName: \n"
                        "  assetBundleVariant: \n", encoding="utf-8")
    return meta


def _remove_old(entry: dict, base: Path) -> None:
    """Files this exporter wrote last time (engines' .import / .uid / .meta stay)."""
    for rel in entry.get("all_files", []):
        p = base / rel
        if p.suffix in (".import", ".uid", ".meta"):
            continue
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def export_to_game(items, engine: str, project_path, subdir: str = "", *, project=None,
                   force: bool = False, models: bool = False, build_models: bool = False,
                   pixel_filter: bool = True, preset: str | None = None, ppu: float | None = None,
                   map_target: str | None = None, runtime: bool = True, progress=None) -> dict:
    """Write assets (assets.Asset / library Characters) into a Godot or Unity project; returns a
    report dict (exported / unchanged / skipped, manifest path, warnings). runtime: also ship the
    game runtime (install_runtime: <subdir>/pp_runtime/ + motion.json, listed as manifest
    "runtime")."""
    from . import assets as A
    from .exports import file_name
    say = progress or (lambda t: None)
    root = check_project(engine, project_path)
    sub = _subdir(engine, subdir)
    base = root / sub
    base.mkdir(parents=True, exist_ok=True)
    proj = project or A.current_project()
    man_path = base / MANIFEST
    man = load_manifest(man_path)
    if man.get("engine") not in (None, engine):
        raise ValueError(f"{man_path} was written for {man.get('engine')}; use another subdir")
    entries: dict = man.get("assets", {})
    report = {"engine": engine, "project_path": str(root), "subdir": sub,
              "manifest": str(man_path), "exported": [], "unchanged": [], "skipped": [],
              "warnings": []}
    options = {k: v for k, v in {"preset": preset, "ppu": ppu, "map_target": map_target,
                                 "models": bool(models)}.items() if v}
    items = list(items)
    for k, asset in enumerate(items):
        aid = asset_id(asset)
        kind = aid.split("/", 1)[0]
        if kind == "character":
            from .library import is_draft_folder
            if is_draft_folder(asset.folder):
                report["skipped"].append({"id": aid, "why": "draft: its front waits for your OK "
                                                            "(character_approve), no sprites yet"})
                continue
        say(f"Exporting {asset.name} ({k + 1} of {len(items)})…")
        fp = fingerprint(asset, engine, options)
        old = entries.get(aid)
        target = base / A.FOLDERS.get(kind, kind + "s") / Path(asset.folder).name
        if old and not force and old.get("source_hash") == fp and \
                all((base / f).exists() for f in old.get("all_files", [])):
            report["unchanged"].append(aid)
            continue
        if old:
            _remove_old(old, base)
        res_dir = "res://" + target.relative_to(root).as_posix()
        try:
            files = list(_write_engine_files(asset, kind, engine, target, res_dir, options, proj))
            if models and kind in MODEL_KINDS:
                from . import gamemodel
                if not gamemodel.has_game_model(asset):
                    if not build_models:
                        report["warnings"].append(f"{aid}: no game-ready 3D model yet (game_model "
                                                  "builds one); sprites exported only")
                    else:
                        say(f"Building the game-ready 3D model of {asset.name}…")
                        gamemodel.game_model_for(asset, progress=say)
                if gamemodel.has_game_model(asset):
                    from .exports import export_asset
                    files += export_asset(asset, f"{engine}-3d-model", target / "model")
        except Exception as e:  # noqa: BLE001 - one asset mustn't stop the rest
            report["skipped"].append({"id": aid, "why": f"{type(e).__name__}: {e}"})
            entries.pop(aid, None)
            continue
        if engine == "godot":
            files = embed_standalone(target, root, [Path(f) for f in files])
        files = [Path(f) for f in files if Path(f).exists() and Path(f).is_file()]
        files = [f for f in files if base in f.parents]
        if engine == "godot":
            files += _godot_imports(files, root)
        entry = {"id": aid, "name": asset.name, "key": file_name(asset.name), "kind": kind,
                 "folder": target.relative_to(base).as_posix(), "source": str(asset.folder),
                 "source_hash": fp, "exported": round(time.time()), "engine": engine,
                 "preset": options.get("preset") or ("godot-2d" if engine == "godot" else "unity")}
        entry.update(_describe(asset, kind, files, base, root, engine, proj, target))
        tags = list(getattr(asset, "tags", []) or [])
        if tags:
            entry["tags"] = tags
        entries[aid] = entry
        report["exported"].append({"id": aid, "files": len(entry["all_files"]),
                                   "folder": entry["folder"]})
    by_name: dict[str, str] = {}
    for aid, e in sorted(entries.items(), key=lambda kv: kv[1].get("exported", 0)):
        by_name[e.get("key") or aid] = aid        # the newest export of a name wins
    man = {"generator": "pixel-pipeline", "version": VERSION, "engine": engine, "subdir": sub,
           "res_root": f"res://{sub}" if engine == "godot" else sub, "project": proj.name,
           "updated": round(time.time()), "assets": dict(sorted(entries.items())),
           "by_name": dict(sorted(by_name.items()))}
    from .motion import MOTION
    man["motion"] = dict(MOTION)            # incl. the feet collider + corner correction (on)
    from .assets import pixel_grid
    man["pixel_grid"] = pixel_grid(proj.style)   # tile, ppu, whole scale, base, render, pitch
    if runtime:
        rt = install_runtime(engine, base)
        man["runtime"] = {Path(f).name: (f"res://{sub}/pp_runtime/{Path(f).name}" if engine == "godot"
                                         else f"{sub}/pp_runtime/{Path(f).name}") for f in rt}
        report["runtime"] = [str(f) for f in rt]
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    if engine == "unity":
        _unity_text_meta(man_path, f"pixel-pipeline/manifest/{sub}")
    if engine == "godot" and pixel_filter:
        if ensure_godot_pixel_filter(root):
            report["warnings"].append("project.godot: set the canvas texture filter to Nearest "
                                      "(rendering/textures/canvas_textures/default_texture_filter=0)")
    report["counts"] = {"exported": len(report["exported"]), "unchanged": len(report["unchanged"]),
                        "skipped": len(report["skipped"]), "in_manifest": len(entries)}
    return report


# --------------------------------------------------------------------------- Godot check

VERIFY_GD = r'''extends SceneTree
# Written by pixel-pipeline (verify_godot); deleted after the run.
func _init():
	var path = "%MANIFEST%"
	var f = FileAccess.open(path, FileAccess.READ)
	if f == null:
		print("PPVERIFY " + JSON.stringify({"error": "no manifest at " + path}))
		quit(1)
		return
	var man = JSON.parse_string(f.get_as_text())
	var out = {}
	for id in man["assets"]:
		var e = man["assets"][id]
		var res_paths = e.get("res", {})
		var r = {"ok": true, "loaded": [], "failed": []}
		for role in res_paths:
			var p = res_paths[role]
			if p.ends_with(".gd") or p.ends_with(".json") or p.ends_with(".tmj") or p.ends_with(".tsx"):
				continue
			var res = ResourceLoader.load(p)
			if res == null:
				r["ok"] = false
				r["failed"].append(p)
				continue
			var what = res.get_class()
			if res is SpriteFrames:
				what += " anims=%d" % res.get_animation_names().size()
			elif res is Texture2D:
				what += " %dx%d" % [res.get_width(), res.get_height()]
			elif res is TileSet:
				what += " sources=%d" % res.get_source_count()
			elif res is PackedScene:
				var node = res.instantiate()
				if node == null:
					r["ok"] = false
					r["failed"].append(p + " (instantiate)")
					continue
				what += " root=" + node.get_class()
				node.free()
			r["loaded"].append(role + ": " + what)
		out[id] = r
	print("PPVERIFY " + JSON.stringify(out))
	quit(0)
'''


def verify_godot(project_path, subdir: str = "", timeout: float = 600.0, godot: str | None = None) -> dict:
    """Import the project headless (godot --headless --import) and load every resource the
    manifest lists (SpriteFrames, textures, TileSets, themes, scenes instantiated, models).
    Returns {ok, godot, assets: {id: {ok, loaded, failed}}, errors: [import log lines]}."""
    root = check_project("godot", project_path)
    sub = _subdir("godot", subdir)
    man = root / sub / MANIFEST
    if not man.exists():
        raise ValueError(f"No {MANIFEST} in {root / sub}: export into it first")
    exe = godot or find_godot()
    if not exe:
        raise ValueError("Godot 4 not found: set GODOT_BIN to the Godot (console) exe")
    run = [exe, "--headless", "--path", str(root)]
    imp = subprocess.run(run + ["--import"], capture_output=True, text=True, timeout=timeout,
                         encoding="utf-8", errors="replace")
    if "ERROR" in (imp.stdout or "") + (imp.stderr or ""):    # dependent imports (fonts on their
        imp = subprocess.run(run + ["--import"], capture_output=True, text=True,  # pages) settle
                             timeout=timeout, encoding="utf-8", errors="replace")  # on a 2nd pass
    script = root / "pixel_pipeline_verify.gd"
    script.write_text(VERIFY_GD.replace("%MANIFEST%", f"res://{sub}/{MANIFEST}"), encoding="utf-8")
    try:
        res = subprocess.run(run + ["-s", "res://pixel_pipeline_verify.gd"], capture_output=True,
                             text=True, timeout=timeout, encoding="utf-8", errors="replace")
    finally:
        for p in (script, script.with_name(script.name + ".uid")):
            p.unlink(missing_ok=True)
    line = next((ln for ln in (res.stdout or "").splitlines() if ln.startswith("PPVERIFY ")), None)
    errors = [ln.strip() for ln in ((imp.stdout or "") + (imp.stderr or "") + (res.stderr or "")).splitlines()
              if re.search(r"\b(ERROR|SCRIPT ERROR)\b", ln)]
    if line is None:
        return {"ok": False, "godot": exe, "assets": {}, "errors": errors[-30:] or
                [(res.stdout or "")[-2000:], (res.stderr or "")[-2000:]]}
    data = json.loads(line[len("PPVERIFY "):])
    if "error" in data:
        return {"ok": False, "godot": exe, "assets": {}, "errors": [data["error"]] + errors[-20:]}
    ok = all(v.get("ok") for v in data.values())
    return {"ok": ok, "godot": exe, "assets": data, "errors": errors[-30:]}
