"""Pixel Pipeline for Claude (MCP tools, the CLI's --json, scripts): every app operation as a plain
function that returns compact JSON-able dicts.

    plan("a knight in a red cape")            # what the router would make + minutes, no work
    job = make("a knight in a red cape")      # a background job (GPU): {"job": id, "status": ...}
    job_status(job["job"], wait_s=30)         # percent, stage, eta; result when done
    review("character/knight")                # the front drawing (draft) + idle, quality report
    approve("character/knight")               # draw the other directions (job)
    export(["character/knight"], "godot", "C:/Games/MyGame", subdir="art")

Assets are named by id "kind/folder" (e.g. "item/rusty-iron-sword"), a folder path, or a name.
Long work runs through agentjobs (one GPU job at a time, records under the project, resumable).
Results carry previews (PNG paths under <project>/.agent/previews) so Claude can look at them,
and quality checks (subject check, drawn-key IoU / colour scores, empty frames).
"""
from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import agentjobs as J
from . import assets as A
from . import makeany as M

PREVIEW_PX = 512
PREVIEW_MAX = 1024
DEFAULT_WAIT_CPU = 300.0          # CPU-only jobs (UI kits, effects, maps...) take seconds


# --------------------------------------------------------------------------- projects

def _project(name: str | None = None) -> A.Project:
    if not name:
        return A.current_project()
    for p in A.list_projects():
        if p.name.lower() == str(name).strip().lower():
            return p
    raise ValueError(f"No project {name!r} (projects: {', '.join(p.name for p in A.list_projects())})")


def _style_dict(st: A.Style) -> dict:
    from dataclasses import asdict
    return asdict(st)


def projects() -> dict:
    cur = A.current_project().name
    out = []
    for p in A.list_projects():
        st = p.style
        out.append({"name": p.name, "current": p.name == cur, "folder": str(p.folder),
                    "type": st.project_type, "engine": st.engine,
                    "assets": len(A.list_assets(project=p))})
    return {"current": cur, "projects": out, "types": {k: v.get("label", k) if isinstance(v, dict) else k
                                                       for k, v in A.PROJECT_TYPES.items()}}


def create_project(name: str, type: str = "custom", size: str = "normal", engine: str = "godot",
                   switch: bool = True, style: dict | None = None) -> dict:
    if (A.projects_folder() / name.strip() / "style.json").exists():
        raise ValueError(f"Project {name!r} exists already (switch_project, or another name)")
    proj = A.create_project(name, type=type, size=size, engine=engine)
    if style:
        set_style(style, project=proj.name)
    if switch:
        A.set_current_project(proj.name)
    return {"project": proj.name, "folder": str(proj.folder), "current": switch,
            "style": _style_dict(proj.style)}


def switch_project(name: str) -> dict:
    proj = _project(name)
    A.set_current_project(proj.name)
    return {"current": proj.name, "folder": str(proj.folder), "style": _style_dict(proj.style)}


def get_style(project: str | None = None) -> dict:
    proj = _project(project)
    st = proj.style
    d = _style_dict(st)
    d["_words"] = {"sprite": A.style_words(st, "sprite")[0], "tile": A.style_words(st, "tile")[0]}
    return {"project": proj.name, "style": d,
            "choices": {"shading": A.SHADINGS, "proportions": A.PROPORTIONS, "detail": A.DETAILS,
                        "outline": ("selout", "solid", "none"), "directions": (2, 4, 8, 16),
                        "engine": ("godot", "unity"), "project_type": tuple(A.PROJECT_TYPES)}}


def set_style(changes: dict, project: str | None = None) -> dict:
    """Change house-style fields (style_prompt, palette, outline, person_height, directions...).
    Returns the new style and what restyling existing assets would do (nothing is redone)."""
    proj = _project(project)
    old = proj.style
    st = proj.style
    unknown = sorted(set(changes) - set(A.Style.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown style field(s) {', '.join(unknown)} "
                         f"(fields: {', '.join(A.Style.__dataclass_fields__)})")
    for k, v in changes.items():
        if k == "shading":
            A.set_shading(st, v)
            continue
        cur = getattr(st, k)
        if isinstance(cur, bool):
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        elif isinstance(cur, list):
            v = list(v) if not isinstance(v, str) else [s.strip() for s in v.split(",") if s.strip()]
        setattr(st, k, v)
    if st.directions not in (2, 4, 8, 16):
        raise ValueError("directions must be 2, 4, 8 or 16")
    proj.save_style(st)
    return {"project": proj.name, "style": _style_dict(proj.style),
            "restyle_plan": A.restyle_plan(proj, old, proj.style)}


def lock_palette(colours: int = 32, project: str | None = None) -> dict:
    proj = _project(project)
    path = A.lock_palette(proj, colours)
    cols = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return {"project": proj.name, "palette_file": str(path), "colours": len(cols), "hex": cols}


# --------------------------------------------------------------------------- asset references

def _all_assets(project: A.Project | None = None) -> list[A.Asset]:
    return A.list_assets(project=project)


def asset_ref(a: A.Asset) -> str:
    return f"{a.kind}/{Path(a.folder).name}"


def resolve(ref, kind: str | None = None, project: str | A.Project | None = None) -> A.Asset:
    """An asset by id ("kind/folder"), folder path, or (unique) name."""
    if isinstance(ref, A.Asset):
        return ref
    proj = project if isinstance(project, A.Project) else _project(project)
    s = str(ref).strip()
    p = Path(s)
    if p.is_absolute() and p.is_dir():
        a = A._load(p, kind or _kind_of_folder(p))
        if a is None:
            raise ValueError(f"{p} is not an asset folder")
        return a
    every = _all_assets(proj)
    if "/" in s:
        k, _, folder = s.partition("/")
        hit = [a for a in every if a.kind == k and Path(a.folder).name == folder]
        if hit:
            return hit[0]
    low = A.slug(s)
    hits = [a for a in every if (kind is None or a.kind == kind)
            and (A.slug(a.name) == low or Path(a.folder).name == low)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"{s!r} matches {len(hits)} assets: "
                         + ", ".join(asset_ref(a) for a in hits) + " (use the id)")
    near = [asset_ref(a) for a in every if low in A.slug(a.name)][:8]
    raise ValueError(f"No asset {s!r} in project {proj.name}"
                     + (f" (did you mean {', '.join(near)}?)" if near else ""))


def _kind_of_folder(p: Path) -> str:
    if (p / "character.json").exists():
        return "character"
    try:
        return json.loads((p / "asset.json").read_text(encoding="utf-8")).get("kind", "prop")
    except (OSError, ValueError):
        return "prop"


def character(ref, project=None):
    from . import library
    a = resolve(ref, "character", project)
    if a.kind != "character":
        raise ValueError(f"{asset_ref(a)} is not a character")
    ch = library.from_folder(a.folder)
    if ch is None:
        raise ValueError(f"{asset_ref(a)} has no character.json")
    return ch


# --------------------------------------------------------------------------- previews

def preview_folder(project: A.Project | None = None) -> Path:
    proj = project or A.current_project()
    f = Path(proj.folder) / ".agent" / "previews"
    f.mkdir(parents=True, exist_ok=True)
    return f


def _checker(w: int, h: int, cell: int) -> Image.Image:
    light = (np.arange(h)[:, None] // cell + np.arange(w)[None, :] // cell) % 2 == 1
    return Image.fromarray(np.where(light, np.uint8(204), np.uint8(153)), "L").convert("RGBA")


def _flat(img: Image.Image, cell: int = 8) -> Image.Image:
    img = img.convert("RGBA")
    if img.getextrema()[3][0] == 255:
        return img.convert("RGB")
    return Image.alpha_composite(_checker(img.width, img.height, max(4, cell)), img).convert("RGB")


def enlarged(img: Image.Image, target: int = PREVIEW_PX) -> Image.Image:
    """Pixel art enlarged crisp (nearest) to ~target px, big pictures shrunk to PREVIEW_MAX."""
    w, h = img.size
    k = max(1, min(target // max(w, h, 1), PREVIEW_MAX // max(w, h, 1)))
    out = img.convert("RGBA")
    if k > 1:
        out = out.resize((w * k, h * k), Image.NEAREST)
    elif max(w, h) > PREVIEW_MAX:
        out.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return _flat(out, 8 * k)


def grid(images: list, labels: list[str] | None = None, cols: int | None = None,
         target: int = 160, crop: bool = True) -> Image.Image:
    """Images (PIL / RGBA arrays / paths) in a labelled grid, each enlarged alike (whole
    pixels, as big as fits PREVIEW_MAX). crop: cut all to their common opaque box first."""
    ims = []
    for im in images:
        if isinstance(im, (str, Path)):
            im = Image.open(im)
        elif isinstance(im, np.ndarray):
            im = Image.fromarray(np.ascontiguousarray(im))
        ims.append(im.convert("RGBA"))
    if not ims:
        return Image.new("RGB", (8, 8), (48, 48, 48))
    if crop and len({i.size for i in ims}) == 1:
        boxes = [i.getbbox() for i in ims]
        boxes = [b for b in boxes if b]
        if boxes:
            box = (max(0, min(b[0] for b in boxes) - 1), max(0, min(b[1] for b in boxes) - 1),
                   min(ims[0].width, max(b[2] for b in boxes) + 1),
                   min(ims[0].height, max(b[3] for b in boxes) + 1))
            ims = [i.crop(box) for i in ims]
    labels = labels or [""] * len(ims)
    n = len(ims)
    cols = max(1, min(cols or math.ceil(math.sqrt(n)), n))
    rows = math.ceil(n / cols)
    cw = max(i.width for i in ims)
    ch = max(i.height for i in ims)
    gap, lab = 6, 12 if any(labels) else 0
    k = max(1, min(max(1, target // max(cw, ch)),
                   (PREVIEW_MAX - gap * (cols + 1)) // max(1, cols * cw),
                   (PREVIEW_MAX - (gap + lab) * rows - gap) // max(1, rows * ch)))
    W = gap + cols * (cw * k + gap)
    H = gap + rows * (ch * k + lab + gap)
    sheet = Image.new("RGB", (W, H), (40, 40, 44))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (im, label) in enumerate(zip(ims, labels)):
        x = gap + (i % cols) * (cw * k + gap)
        y = gap + (i // cols) * (ch * k + lab + gap)
        if label:
            draw.text((x, y), str(label)[: max(3, cw * k // 6)], fill=(230, 230, 230), font=font)
        big = im.resize((im.width * k, im.height * k), Image.NEAREST) if k > 1 else im
        cell = Image.new("RGBA", (cw * k, ch * k), (0, 0, 0, 0))
        cell.paste(big, ((cw * k - big.width) // 2, ch * k - big.height))
        sheet.paste(_flat(cell, 4 * k), (x, y + lab))
    if max(sheet.size) > PREVIEW_MAX:
        sheet.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return sheet


def _sheet(folder: Path, name: str = "sheet"):
    """(image RGBA array, frames, {group: [Frame]}) of a sheet.png + .json, or None."""
    from .exports import _frames, group_key
    png, js = folder / f"{name}.png", folder / f"{name}.json"
    if not (png.exists() and js.exists()):
        return None
    try:
        meta = json.loads(js.read_text(encoding="utf-8"))
        frames = _frames(meta)
        img = np.asarray(Image.open(png).convert("RGBA"))
    except (OSError, ValueError, KeyError):
        return None
    groups: dict[str, list] = {}
    for f in frames:
        groups.setdefault(group_key(f.name), []).append(f)
    return img, frames, groups, meta


def _crop(img, f):
    return img[f.y:f.y + f.h, f.x:f.x + f.w]


def _facing(dirs: list[str]) -> str | None:
    for d in ("S", "SE", "SSE", "E", "SW"):
        if d in dirs:
            return d
    return dirs[0] if dirs else None


def _group(groups: dict, action: str, d: str):
    return groups.get(f"{action}_{d}") or (groups.get(d) if action == "default" else None)


def _actions_dirs(groups: dict) -> tuple[list[str], list[str]]:
    from .turnaround import COMPASS16
    acts, dirs = [], []
    for key in groups:
        a, _, d = key.rpartition("_")
        if key in COMPASS16:                    # still sprites per direction (items, props)
            a, d = "default", key
        if a and d in COMPASS16:
            if a not in acts:
                acts.append(a)
            if d not in dirs:
                dirs.append(d)
    return acts, dirs


def write_previews(asset: A.Asset, project: A.Project | None = None) -> dict:
    """Preview pictures of an asset for Claude to look at: {"preview": png, "animation": gif,
    "variations": png} (paths that exist)."""
    folder = Path(asset.folder)
    out_dir = preview_folder(project)
    stem = f"{asset.kind}-{folder.name}"
    out: dict[str, str] = {}
    sh = _sheet(folder)
    if asset.kind == "character" and sh is None:
        from . import library
        ch = library.from_folder(folder)
        info = library.preview_info(ch) if ch is not None else None
        if info:                                   # a draft: its front drawing + idle
            key = Image.open(info["key"])
            frames = [Image.open(f) for f in info["frames"]]
            p = out_dir / f"{stem}.png"
            grid([key] + frames[:8], [f"front {info['direction']}"] + [str(i) for i in range(len(frames[:8]))],
                 cols=min(5, 1 + len(frames[:8])), target=192).save(p)
            out["preview"] = str(p)
            if frames:
                g = out_dir / f"{stem}_idle.gif"
                _gif(frames, g, int(info.get("frame_ms", 120)))
                out["animation"] = str(g)
    if sh is not None and asset.kind not in ("tileset", "ui", "map", "card", "background"):
        img, frames, groups, meta = sh
        acts, dirs = _actions_dirs(groups)
        if dirs:
            act = "idle" if "idle" in acts else acts[0]
            cells, labels = [], []
            for d in dirs:
                fs = _group(groups, act, d)
                if fs:
                    cells.append(_crop(img, fs[0]))
                    labels.append(d)
            p = out_dir / f"{stem}.png"
            grid(cells, labels, cols=min(8, len(cells)), target=256).save(p)
            out["preview"] = str(p)
            face = _facing(dirs)
            if len(acts) > 1 or len(_group(groups, act, face) or []) > 1:
                cells, labels = [], []
                shown = acts[:10]
                width = max(len(_group(groups, a, face) or []) for a in shown)
                width = min(8, width)
                for a in shown:
                    fs = (_group(groups, a, face) or [])[:width]
                    for i in range(width):
                        f = fs[min(i, len(fs) - 1)] if fs else None
                        cells.append(_crop(img, f) if f is not None else np.zeros((1, 1, 4), np.uint8))
                        labels.append(f"{a}" if i == 0 else (str(i) if i < len(fs) else ""))
                q = out_dir / f"{stem}_animations.png"
                grid(cells, labels, cols=width, target=192).save(q)
                out["animations"] = str(q)
            main = _group(groups, act, face) or frames[:8]
        else:
            first = list(groups.values())[0]
            cells = [_crop(img, f) for f in frames[:32]]
            labels = [str(i) for i in range(len(cells))]
            p = out_dir / f"{stem}.png"
            grid(cells, labels, cols=min(8, len(cells)), target=192).save(p)
            out["preview"] = str(p)
            main = first
        if len(main) > 1:
            g = out_dir / f"{stem}_anim.gif"
            _gif([Image.fromarray(np.ascontiguousarray(_crop(img, f))) for f in main], g,
                 int(main[0].duration or 100), [int(f.duration or 100) for f in main])
            out["animation"] = str(g)
    if "preview" not in out:
        for name in ("preview.png", "card.png", "example.png", "icon.png", "portrait_neutral.png",
                     "tileset.png", "thumb.png"):
            p = folder / name
            if p.exists():
                q = out_dir / f"{stem}.png"
                enlarged(Image.open(p)).save(q)
                out["preview"] = str(q)
                break
    if asset.kind == "item" and (folder / "icon.png").exists() and sh is not None:
        q = out_dir / f"{stem}_icon.png"
        enlarged(Image.open(folder / "icon.png"), 256).save(q)
        out["icon"] = str(q)
    gif = folder / "preview.gif"
    if "animation" not in out and gif.exists():
        out["animation"] = str(gif)
    vars_ = variation_files(asset)
    if len(vars_) > 1:
        q = out_dir / f"{stem}_variations.png"
        grid(vars_, [f"{i + 1}" + (" (chosen)" if i + 1 == chosen_variation(asset) else "")
                     for i in range(len(vars_))], cols=len(vars_), target=192).save(q)
        out["variations"] = str(q)
    return out


def _gif(frames, path: Path, ms: int = 100, durations=None, target: int = 256) -> Path:
    ims = [f.convert("RGBA") for f in frames]
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    k = max(1, target // max(w, h))
    out = []
    for im in ims:
        cell = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cell.paste(im, ((w - im.width) // 2, h - im.height))
        out.append(_flat(cell.resize((w * k, h * k), Image.NEAREST), 4 * k).convert("P", palette=Image.ADAPTIVE))
    out[0].save(path, save_all=True, append_images=out[1:], duration=durations or ms, loop=0,
                disposal=2)
    return path


# --------------------------------------------------------------------------- variations

def variation_files(asset: A.Asset) -> list[Path]:
    f = Path(asset.folder)
    num = lambda q: int(re.sub(r"\D", "", q.stem) or 0)          # noqa: E731
    if asset.kind == "item":
        return sorted(f.glob("icon_v*.png"), key=num)
    if asset.kind == "prop":
        hits = sorted(f.glob("sprite_v*.png"), key=num) or sorted(f.glob("drawn_v*.png"), key=num)
        return hits if len(hits) > 1 else []
    if asset.kind == "card":
        return sorted(f.glob("art_v*.png"), key=num)
    if asset.kind == "tileset" and (f / "swatch_variants.png").exists():
        return []
    return []


def chosen_variation(asset: A.Asset) -> int:
    e = asset.extra or {}
    return int(e.get("icon_choice") or e.get("chosen") or e.get("art_choice") or 1)


def pick_variation(ref, index: int, project: str | None = None, job: str | None = None) -> dict:
    """Use variation `index` (1-based) of an item's icon, a drawn prop's drawing or a card's art;
    tilesets: give the job id that made it (the set is rebuilt from that texture)."""
    proj = _project(project)
    index = int(index)
    if job:
        rec = J.manager().get(job)
        made = _made_for(rec)
        if made is None:
            raise ValueError("That job's variations aren't available any more (make it again)")
        if made.plan.kind == "tileset":
            old = made.asset
            made.extra["project"] = proj
            if index != made.extra.get("kept"):
                made.extra.pop("kept", None)
                made.asset = None
                M.keep(made, index, proj)
                if old is not None and Path(old.folder).exists():
                    _trash(old, proj)
            return {"picked": index, "asset": asset_summary(made.asset, proj)}
        ref = ref or (made.asset and asset_ref(made.asset))
    a = resolve(ref, project=proj)
    n = len(variation_files(a))
    if not n:
        raise ValueError(f"{asset_ref(a)} has no variations to pick from")
    if not 1 <= index <= n:
        raise ValueError(f"index must be 1..{n}")
    if a.kind == "item":
        from .objects import choose_icon, open_asset
        choose_icon(open_asset(a.folder), index, proj)
    elif a.kind == "prop":
        from .objects import choose_drawn, open_asset
        choose_drawn(open_asset(a.folder), index, proj)
    elif a.kind == "card":
        from .cards import choose_art, open_card
        choose_art(open_card(a.folder), index, proj)
    return {"picked": index, "asset": asset_summary(resolve(asset_ref(a), project=proj), proj)}


def _made_path(rec) -> Path:
    return Path(rec.project_folder) / J.FOLDER / f"{rec.id}.made.pkl"


def _made_for(rec):
    made = J.manager().extras.get(rec.id)
    if made is not None:
        return made
    p = _made_path(rec)
    if p.exists():
        try:
            with open(p, "rb") as fh:
                made = pickle.load(fh)
        except Exception:  # noqa: BLE001 - an old pickle: gone
            return None
        J.manager().extras[rec.id] = made
        return made
    return None


# --------------------------------------------------------------------------- quality

def quality(asset: A.Asset) -> dict:
    """What there is to judge without looking: subject check of the chosen picture, the
    generator's own checks, drawn-character key scores, frame/colour stats; `warnings` lists
    what to look at."""
    folder = Path(asset.folder)
    extra = asset.extra or {}
    q: dict = {}
    warn: list[str] = []
    if extra.get("checks"):
        q["picture_checks"] = extra["checks"]
        bad = [c for c in (extra["checks"] if isinstance(extra["checks"], list) else [])
               if isinstance(c, dict) and not c.get("ok", True)]
        if bad:
            warn.append(f"{len(bad)} picture(s) failed the one-subject check")
    pic = _main_picture(asset)
    if pic is not None:
        try:
            from .subjectcheck import assess
            v = assess(np.asarray(Image.open(pic).convert("RGB")))
            q["subject_check"] = {"picture": pic.name, **v.summary()}
            if not v.ok:
                warn.append(f"subject check of {pic.name}: {v.reason()}")
        except Exception:  # noqa: BLE001 - checks are a bonus
            pass
    for k in ("colour_miss", "pattern_miss", "method"):
        if k in extra:
            q[k] = extra[k]
    sh = _sheet(folder)
    if sh is not None and asset.kind not in ("tileset", "ui", "map", "card", "background"):
        img, frames, groups, _ = sh
        empty = [f.name for f in frames if not _crop(img, f)[..., 3].any()]
        opaque = img[img[..., 3] > 0][:, :3]
        q["frames"] = len(frames)
        q["colours"] = int(len(np.unique(opaque, axis=0))) if len(opaque) else 0
        if empty:
            q["empty_frames"] = empty[:12] + (["…"] if len(empty) > 12 else [])
            if asset.kind not in ("effect", "weather", "water"):    # effects fade in/out on purpose
                warn.append(f"{len(empty)} empty frame(s)")
    if asset.kind == "character":
        keys = folder / "keys" / "keys.json"
        if keys.exists():
            try:
                meta = json.loads(keys.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            rep = meta.get("keys_report") or {}
            if rep:
                rows = []
                for d, r in rep.items():
                    rows.append((d, float(r.get("iou", 0)), float(r.get("colour_distance", 0)),
                                 float(r.get("sharp", 0))))
                ious = [r[1] for r in rows]
                worst = sorted(rows, key=lambda r: (r[1] - r[2]))[:3]
                q["drawn"] = {"directions": len(rows), "iou_min": round(min(ious), 3),
                              "iou_mean": round(sum(ious) / len(ious), 3),
                              "worst": [{"dir": d, "iou": round(i, 3), "colour_distance": round(c, 3),
                                         "sharp": round(s, 2)} for d, i, c, s in worst],
                              "drawings": meta.get("drawings"), "model_frames": meta.get("model_frames")}
                if min(ious) < 0.8:
                    warn.append(f"drawn direction {worst[0][0]} fits the model poorly (IoU "
                                f"{worst[0][1]:.2f}): character_redraw it")
            prev = (meta.get("preview") or {}).get("report")
            if prev:
                q["front_report"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in prev.items() if not isinstance(v, list)}
            if meta.get("partial") and not meta.get("preview"):
                warn.append("drawing unfinished (the GPU stopped): character_finish carries on")
    q["warnings"] = warn
    return q


def _main_picture(asset: A.Asset) -> Path | None:
    f = Path(asset.folder)
    e = asset.extra or {}
    names = []
    if asset.kind == "item":
        names = [f"picture_v{int(e.get('icon_choice', 1) or 1)}.png", "picture_v1.png"]
    elif asset.kind == "prop":
        names = [f"drawn_v{int(e.get('chosen', 1) or 1)}.png", "drawn_v1.png", "drawn_S.png", "model.front.png"]
    elif asset.kind == "character":
        names = ["picture.png"]
    # card art is a scene and portraits are busts: the one-subject check doesn't apply
    return next((f / n for n in names if (f / n).exists()), None)


# --------------------------------------------------------------------------- summaries

def asset_files(asset: A.Asset) -> dict:
    f = Path(asset.folder)
    roles = {"sheet": "sheet.png", "sheet_json": "sheet.json", "normal_map": "sheet_n.png",
             "glow_map": "sheet_glow.png", "aseprite": "sheet.aseprite", "godot_sprite_frames": "sheet.tres",
             "facing_script": "sheet_facing.gd", "icon": "icon.png", "model": "model.glb",
             "rigged": "rigged.glb", "picture": "picture.png", "tileset": "tileset.png",
             "tileset_godot": "tileset.tres", "tileset_tiled": "tileset.tsx", "card": "card.png",
             "map": "map.json", "kit": "kit.json", "theme_preview": "preview.png",
             "weather": "weather.json", "water": "water.json", "parallax": "parallax.json",
             "game_model": "model_game/lod0.glb", "thumb": "thumb.png"}
    return {k: str(f / v) for k, v in roles.items() if (f / v).exists()}


def _state(asset: A.Asset) -> str:
    if asset.kind != "character":
        return "ready"
    from . import library
    ch = library.from_folder(asset.folder)
    if ch is None:
        return "unknown"
    if library.in_preview(ch):
        return "draft: front drawn, waiting for your OK (character_review / character_approve)"
    if library.drawing_unfinished(ch):
        return "unfinished drawing: 3D-rendered sheets for now (character_finish)"
    if not ch.sheet.exists():
        return "no sprites"
    return "ready (drawn)" if library.is_drawn(ch) else "ready (3D render)"


def asset_summary(asset: A.Asset, project: A.Project | None = None, previews: bool = True,
                  checks: bool = True) -> dict:
    f = Path(asset.folder)
    d = {"id": asset_ref(asset), "kind": asset.kind, "name": asset.name, "folder": str(f),
         "state": _state(asset)}
    if asset.tags:
        d["tags"] = asset.tags
    if asset.collections:
        d["collections"] = asset.collections
    prompt = (asset.extra or {}).get("prompt")
    if asset.kind == "character":
        from . import library
        ch = library.from_folder(f)
        prompt = ch.prompt if ch is not None else prompt
        if ch is not None:
            d["body"], d["size"] = ch.kind, ch.size
            from . import metrics as MT
            t = MT.summary(f)
            if t:
                d["timing"] = t                            # minutes per stage, GPU, wall
            q = library.quality_of(ch)
            if q:
                d["direction_quality"] = {k: q.get(k) for k in ("text", "passed", "total", "failed",
                                                                 "redrawn", "reverted")}
    if prompt:
        d["prompt"] = prompt
    d["files"] = asset_files(asset)
    sh = _sheet(f)
    if sh is not None and asset.kind not in ("tileset", "ui", "map", "card", "background"):
        _, frames, groups, _ = sh
        acts, dirs = _actions_dirs(groups)
        d["frame"] = {"w": frames[0].w, "h": frames[0].h} if frames else None
        if acts:
            d["animations"] = acts
            d["directions"] = dirs
        else:
            d["animations"] = list(groups)
            d["frames"] = len(frames)
    vars_ = variation_files(asset)
    if vars_:
        d["variations"] = [str(v) for v in vars_]
        d["chosen"] = chosen_variation(asset)
    keep = ("category", "terrain", "effect", "theme", "template", "weather", "tod", "scene",
            "look", "directions", "height_people", "expressions", "world", "body")
    small = {k: v for k, v in (asset.extra or {}).items() if k in keep and v not in (None, "", [])}
    if small:
        d["settings"] = small
    if previews:
        try:
            d["previews"] = write_previews(asset, project)
        except Exception as e:  # noqa: BLE001 - a summary must not fail on a preview
            d["previews"] = {"error": str(e)}
    if checks:
        d["quality"] = quality(asset)
    return d


# --------------------------------------------------------------------------- library

def list_assets(kind: str | None = None, query: str = "", tag: str | None = None,
                collection: str | None = None, project: str | None = None, limit: int = 50) -> dict:
    proj = _project(project)
    if kind and kind not in A.KINDS:
        raise ValueError(f"kind must be one of {', '.join(A.KINDS)}")
    found = A.list_assets(kind, query, tag, collection, project=proj)
    rows = []
    for a in found[:limit]:
        r = {"id": asset_ref(a), "name": a.name, "kind": a.kind}
        if a.tags:
            r["tags"] = a.tags
        if a.collections:
            r["collections"] = a.collections
        if a.kind == "character":
            r["state"] = _state(a).split(":")[0]
        rows.append(r)
    return {"project": proj.name, "count": len(found), "assets": rows,
            "tags": A.all_tags(proj), "collections": A.all_collections(proj)}


def get_asset(ref, project: str | None = None) -> dict:
    proj = _project(project)
    return asset_summary(resolve(ref, project=proj), proj)


def _trash(asset: A.Asset, proj: A.Project) -> Path:
    t = Path(proj.folder) / ".trash" / f"{time.strftime('%Y%m%d-%H%M%S')}-{asset.kind}-{Path(asset.folder).name}"
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(asset.folder), str(t))
    return t


def delete_asset(ref, project: str | None = None) -> dict:
    """Move an asset into <project>/.trash (recoverable by moving the folder back)."""
    proj = _project(project)
    a = resolve(ref, project=proj)
    where = _trash(a, proj)
    return {"deleted": asset_ref(a), "name": a.name, "moved_to": str(where)}


def tag_asset(ref, add_tags=(), remove_tags=(), add_collections=(), remove_collections=(),
              project: str | None = None) -> dict:
    proj = _project(project)
    a = resolve(ref, project=proj)
    tags = (set(a.tags) | set(add_tags or ())) - set(remove_tags or ())
    cols = (set(a.collections) | set(add_collections or ())) - set(remove_collections or ())
    A.set_tags(a, tags, cols)
    return {"id": asset_ref(a), "tags": a.tags, "collections": a.collections}


def collections(project: str | None = None) -> dict:
    proj = _project(project)
    out = {}
    for c in A.all_collections(proj):
        out[c] = [asset_ref(a) for a in A.list_assets(collection=c, project=proj)]
    return {"project": proj.name, "collections": out}


# --------------------------------------------------------------------------- plans + make

def _fields(kind: str) -> list[dict]:
    out = []
    for key, label, choice in M.fields_for(kind):
        f = {"key": key, "label": label}
        if isinstance(choice, tuple):
            f["choices"] = list(choice)
        elif choice is bool:
            f["type"] = "bool"
        elif isinstance(choice, str):
            f["type"] = choice
        out.append(f)
    return out


def _apply(plan: M.Plan, params: dict | None, name: str | None) -> M.Plan:
    if name:
        plan.name = name.strip()
    for k, v in (params or {}).items():
        if k == "prompt":
            plan.prompt = str(v)
        elif k == "name":
            plan.name = str(v)
        else:
            plan.params[k] = v
    return plan


def _plan_dict(plan: M.Plan, style=None) -> dict:
    gpu = M.gpu_minutes(plan, style)
    return {"kind": plan.kind, "name": plan.name, "prompt": plan.prompt, "params": plan.params,
            "label": plan.label, "why": plan.why, "summary": M.summary(plan, style),
            "gpu_minutes": round(gpu, 2), "cpu_minutes": round(M.cpu_minutes(plan), 2),
            "after_ok_gpu_minutes": round(M.after_ok_minutes(plan), 2), "uses_gpu": gpu > 0}


def build_plan(description: str, kind: str | None = None, params: dict | None = None,
               name: str | None = None, project: A.Project | None = None) -> tuple[M.Plan, list]:
    proj = project or A.current_project()
    with A.using_project(proj):
        st = proj.style
        guesses = M.classify(description, st)
        if not guesses:
            raise ValueError("Describe what to make, e.g. 'a wolf' or 'rusty iron sword'")
        if kind:
            if kind not in M.KINDS:
                raise ValueError(f"kind must be one of {', '.join(M.KINDS)}")
            plan = M.plan_for(kind, description, style=st)
            plan.why = "kind given"
        else:
            plan = guesses[0].plan
        return _apply(plan, params, name), [g for g in guesses if g.kind != plan.kind]


def plan(description: str, kind: str | None = None, params: dict | None = None,
         name: str | None = None, project: str | None = None) -> dict:
    """What make() would do: the plan, minutes of GPU/CPU, other kinds it could be, the fields."""
    proj = _project(project)
    p, others = build_plan(description, kind, params, name, proj)
    with A.using_project(proj):
        st = proj.style
        return {"plan": _plan_dict(p, st), "fields": _fields(p.kind),
                "alternatives": [{"kind": g.kind, "name": g.plan.name, "why": g.plan.why,
                                  "gpu_minutes": round(M.gpu_minutes(g.plan, st), 2)} for g in others],
                "project": proj.name}


def _make_plan(args: dict):
    try:
        return M.progress_plan(M.Plan.from_dict(args["plan"]))
    except Exception:  # noqa: BLE001
        return None, None


@J.register("make", plan=_make_plan)
def _op_make(args: dict, tracker) -> dict:
    proj = A.current_project()
    pl = M.Plan.from_dict(args["plan"])
    made = M.run_plan(pl, proj, tracker, client=J.manager().client)
    if not made.saved:                        # tilesets: the best texture first, pick another later
        M.keep(made, 1, proj)
    job_id = args.get("_job")
    if job_id:
        J.manager().extras[job_id] = made
        if pl.kind == "tileset":
            try:
                with open(Path(proj.folder) / J.FOLDER / f"{job_id}.made.pkl", "wb") as fh:
                    pickle.dump(made, fh)
            except Exception:  # noqa: BLE001 - only the later pick needs it
                pass
    out = {"asset": asset_summary(made.asset, proj), "seconds": round(made.seconds, 1)}
    if pl.kind == "tileset":
        sw = made.extra.get("swatches") or []
        if len(sw) > 1:
            q = preview_folder(proj) / f"tileset-{Path(made.asset.folder).name}_variations.png"
            grid([s.texture for s in sw], [f"{i + 1}" + (" (kept)" if i == 0 else "") for i in range(len(sw))],
                 cols=len(sw), target=128).save(q)
            out["variations"] = {"preview": str(q), "count": len(sw),
                                 "checks": [{"colour_miss": round(float(s.colour_miss), 3),
                                             "pattern_miss": round(float(s.pattern_miss), 3),
                                             "method": s.method} for s in sw],
                                 "pick": "pick_variation(job=<this job>, index=n) rebuilds the set from texture n"}
    if made.notice:
        out["notice"] = made.notice
    if made.waiting:
        out["next"] = ("Look at the front drawing (previews.preview / character_review), then "
                       "character_approve, character_redraw_front or character_change_words")
    return out


def _start(op: str, args: dict, *, project: A.Project, gpu: bool, title: str, estimate_min: float,
           wait_s: float | None) -> dict:
    if wait_s is None:
        wait_s = 0.0 if gpu else DEFAULT_WAIT_CPU
    rec = J.manager().start(op, args, project=project, gpu=gpu, title=title,
                            estimate_min=estimate_min, wait_s=wait_s)
    return job_view(rec.id)


def job_view(job_id: str) -> dict:
    m = J.manager()
    rec = m.get(job_id)
    d = rec.public()
    pos = m.position(job_id)
    if rec.status == "queued" and pos:
        d["queue_position"] = pos
    if rec.status in J.ACTIVE:
        d["check_again"] = "job_status(job, wait_s=60)"
    return d


def make(description: str, kind: str | None = None, params: dict | None = None,
         name: str | None = None, dry_run: bool = False, wait_s: float | None = None,
         project: str | None = None) -> dict:
    """Route + plan + run (as a job) anything the Make tab makes. dry_run = only the plan."""
    proj = _project(project)
    if dry_run:
        return plan(description, kind, params, name, proj.name)
    p, _ = build_plan(description, kind, params, name, proj)
    if p.kind == "character" and p.params.get("seed") is None:
        p.params["seed"] = int(time.time() * 1000) % 2_000_000_000   # a resume makes the same picture
    with A.using_project(proj):
        gpu_min = M.gpu_minutes(p, proj.style)
        est = gpu_min + M.cpu_minutes(p)
    job = _start("make", {"plan": p.to_dict()}, project=proj, gpu=gpu_min > 0,
                 title=f"{p.label}: {p.name}", estimate_min=est, wait_s=wait_s)
    job["plan"] = _plan_dict(p, proj.style)
    return job


def make_kind(kind: str, description: str, *, name: str | None = None, wait_s: float | None = None,
              project: str | None = None, **params) -> dict:
    """make() with the kind fixed and its settings (None = the router's choice)."""
    return make(description, kind, {k: v for k, v in params.items() if v is not None}, name,
                wait_s=wait_s, project=project)


# --------------------------------------------------------------------------- characters: front first

def review(ref, project: str | None = None) -> dict:
    """A character's state for review: the front drawing + idle (drafts), or its sheet."""
    from . import library
    proj = _project(project)
    ch = character(ref, proj)
    a = resolve(ref, "character", proj)
    out = asset_summary(a, proj)
    info = library.preview_info(ch)
    if info:
        out["review"] = {"direction": info["direction"], "key": str(info["key"]),
                         "idle_frames": [str(p) for p in info["frames"]], "frame_ms": info["frame_ms"],
                         "clip": info["clip"], "picture": str(info["picture"]) if info["picture"] else None,
                         "words": info["words"], "after_ok": info["estimate"],
                         "tries": [{"n": t["n"], "seed": t.get("seed"), "words": t.get("words"),
                                    "key": str(t["key"]), "current": t["current"]} for t in info["attempts"]],
                         "choices": ["character_approve (looks good: draw the rest)",
                                     "character_redraw_front (new seed)",
                                     "character_change_words (new words, optionally a new 3D model)",
                                     "character_pick_try (an earlier try)", "delete_asset (throw it away)"]}
        tries = [t for t in info["attempts"] if Path(t["key"]).exists()]
        if len(tries) > 1:
            q = preview_folder(proj) / f"character-{Path(a.folder).name}_tries.png"
            grid([t["key"] for t in tries], [f"try {t['n']}" + (" *" if t["current"] else "") for t in tries],
                 cols=len(tries), target=160).save(q)
            out["previews"]["tries"] = str(q)
    return out


def _char_plan(op: str):
    def plan_fn(args):
        from . import progress as P
        n = int(args.get("_directions") or 16)
        if op == "approve":
            return P.character_plan(model=False, drawn=True, preview=False, directions=n), None
        return None, (M.GPU["drawn_dir"] * (len(args.get("directions") or []) or 1) + 0.3) * 60
    return plan_fn


def _cancel_fn(tracker):
    return lambda: tracker.stopped


@J.register("character_approve", plan=_char_plan("approve"))
def _op_approve(args, tracker):
    from . import library
    ch = character(args["ref"])
    ch = library.approve_preview(ch, progress=tracker, cancel=_cancel_fn(tracker))
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


@J.register("character_redraw_front", plan=lambda a: (None, (3.4 if a.get("new_model") else 0.8) * 60))
def _op_redraw_front(args, tracker):
    from . import library
    ch = character(args["ref"])
    library.redraw_preview(ch, seed=args.get("seed"), words=args.get("words"),
                           new_model=bool(args.get("new_model")), progress=tracker,
                           cancel=_cancel_fn(tracker))
    return {"asset": review(args["ref"])}


@J.register("character_finish", plan=_char_plan("approve"))
def _op_finish(args, tracker):
    from . import library
    ch = character(args["ref"])
    library.finish_drawing(ch, progress=tracker, cancel=_cancel_fn(tracker))
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


@J.register("character_redraw", plan=_char_plan("redraw"))
def _op_redraw(args, tracker):
    from . import library
    ch = character(args["ref"])
    library.redraw(ch, args.get("directions") or None, progress=tracker, cancel=_cancel_fn(tracker))
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


@J.register("character_check", plan=lambda a: (None, float(a.get("_gpu_min", 2.0)) * 60 + 90))
def _op_check(args, tracker):
    from . import library
    ch = character(args["ref"])
    q = library.check_and_fix(ch, progress=tracker, cancel=_cancel_fn(tracker),
                              redraw_bad=bool(args.get("redraw", True)))
    return {"quality": q, "asset": asset_summary(resolve(args["ref"]), A.current_project())}


def _char_job(op: str, ref, project, wait_s, title: str, gpu: bool = True, minutes: float = 1.0,
              **args) -> dict:
    proj = _project(project)
    with A.using_project(proj):
        a = resolve(ref, "character", proj)
        ch = character(a, proj)
    from . import library
    meta = library._keys_meta(ch) or {}
    dirs = int((meta.get("camera") or {}).get("directions") or proj.style.directions)
    return _start(op, {"ref": asset_ref(a), "_directions": dirs, **args}, project=proj, gpu=gpu,
                  title=f"{title}: {ch.name}", estimate_min=minutes, wait_s=wait_s)


def approve(ref, wait_s: float | None = None, project: str | None = None) -> dict:
    from . import library
    proj = _project(project)
    with A.using_project(proj):
        ch = character(ref, proj)
        if not library.in_preview(ch):
            raise ValueError(f"{ch.name} isn't waiting for an OK ({_state(resolve(ref, project=proj))})")
        dirs = int((library._keys_meta(ch).get("camera") or {}).get("directions") or proj.style.directions)
    return _char_job("character_approve", ref, project, wait_s, "Draw the rest",
                     minutes=library.rest_minutes(dirs) + M.GPU["bend"])


def redraw_front(ref, seed: int | None = None, wait_s: float | None = None, project: str | None = None) -> dict:
    return _char_job("character_redraw_front", ref, project, wait_s, "Redraw the front", minutes=0.8,
                     seed=seed)


def change_words(ref, words: str, new_model: bool = False, wait_s: float | None = None,
                 project: str | None = None) -> dict:
    if not str(words or "").strip():
        raise ValueError("Give the new words")
    return _char_job("character_redraw_front", ref, project, wait_s, "New words",
                     minutes=3.4 if new_model else 0.8, words=words.strip(), new_model=bool(new_model))


def pick_try(ref, n: int, project: str | None = None) -> dict:
    from . import library
    proj = _project(project)
    with A.using_project(proj):
        ch = character(ref, proj)
        library.pick_preview(ch, int(n))
        return review(ref, proj.name)


def finish(ref, wait_s: float | None = None, project: str | None = None) -> dict:
    from . import library
    proj = _project(project)
    with A.using_project(proj):
        ch = character(ref, proj)
        if library.in_preview(ch):
            raise ValueError(f"{ch.name} is a draft waiting for an OK: use character_approve")
    return _char_job("character_finish", ref, project, wait_s, "Finish drawing", minutes=8.0)


def redraw_directions(ref, directions: list[str] | None, wait_s: float | None = None,
                      project: str | None = None) -> dict:
    from . import library
    from .turnaround import COMPASS16
    proj = _project(project)
    dirs = [d.upper() for d in (directions or [])]
    bad = [d for d in dirs if d not in COMPASS16]
    if bad:
        raise ValueError(f"Unknown direction(s) {bad} (use {', '.join(COMPASS16)})")
    with A.using_project(proj):
        ch = character(ref, proj)
        if not library.is_drawn(ch):
            raise ValueError(f"{ch.name} isn't a finished drawn character ({_state(resolve(ref, project=proj))})")
    return _char_job("character_redraw", ref, project, wait_s,
                     "Redraw " + (", ".join(dirs) if dirs else "every direction"),
                     minutes=M.GPU["drawn_dir"] * (len(dirs) or 16) + M.GPU["bend"], directions=dirs)


def check_quality(ref, redraw: bool = True, confirm: bool = False, wait_s: float | None = None,
                  project: str | None = None) -> dict:
    """Check and fix quality on an existing character (library.check_and_fix): every direction
    measured as the game shows it; failing ones drawn again on the GPU when redraw (bounded).
    Without confirm=True it only returns the estimate (GPU minutes) - say it to the user first."""
    from . import library
    proj = _project(project)
    with A.using_project(proj):
        ch = character(ref, proj)
        est = library.fix_estimate(ch) if library.is_drawn(ch) else {"text": "3D-rendered: checked only",
                                                                      "gpu_min": 0.0, "new_camera": False}
    if not confirm and redraw and est.get("gpu_min"):
        return {"needs_confirm": True, "estimate": est,
                "say": est["text"] + " Call again with confirm=true to go ahead."}
    return _char_job("character_check", ref, project, wait_s, "Check and fix quality",
                     gpu=bool(redraw), minutes=float(est.get("gpu_min") or 0.5) + 1.5,
                     redraw=bool(redraw), _gpu_min=float(est.get("gpu_min") or 0.0))


def metrics(n: int = 10, project: str | None = None) -> dict:
    """How long making a character takes: median / mean minutes per stage over the last n
    characters (metrics.aggregate; project = only that one)."""
    from . import metrics as MT
    root = _project(project).folder if project else None
    return MT.aggregate(int(n), root)


# --------------------------------------------------------------------------- animation

def motions(kind: str = "person") -> dict:
    """Clips that can be added: built-in motions, project clips, mocap library entries."""
    from .motions import HUMAN, QUAD
    if kind == "animal":
        from .quadruped import all_actions
        return {"kind": "animal", "have_by_default": list(all_actions()), "add": sorted(QUAD)}
    from .autorig import all_actions
    from .mocap import library_index, mocap_root
    clips = library_index().get("clips", {})
    root = mocap_root()
    moc = [{"name": k, "downloaded": (root / v["file"]).exists(), "loop": v.get("loop")}
           for k, v in sorted(clips.items())]
    return {"kind": "person", "have_by_default": list(all_actions()), "add": sorted(HUMAN),
            "replace_example": "attack=shoot (with replace=true)", "mocap": moc,
            "mocap_folder": str(root)}


@J.register("add_animation", plan=lambda a: (None, 90.0))
def _op_add_animation(args, tracker):
    from . import library
    ch = character(args["ref"])
    library.add_character_animations(ch, args["names"], progress=tracker, replace=bool(args.get("replace")))
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


@J.register("add_mocap", plan=lambda a: (None, 120.0))
def _op_add_mocap(args, tracker):
    from . import library
    ch = character(args["ref"])
    library.add_mocap(ch, args["names"], progress=tracker, replace=bool(args.get("replace", True)))
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


@J.register("add_springs", plan=lambda a: (None, 150.0))
def _op_add_springs(args, tracker):
    from . import library
    ch = character(args["ref"])
    opts = {k: v for k, v in (args.get("options") or {}).items()
            if k in ("parts", "stiffness", "damping", "gravity", "wind")}
    library.add_character_springs(ch, progress=tracker, **opts)
    return {"asset": asset_summary(resolve(args["ref"]), A.current_project())}


def _anim_job(op, ref, project, wait_s, title, **args) -> dict:
    from . import library
    proj = _project(project)
    with A.using_project(proj):
        a = resolve(ref, "character", proj)
        ch = character(a, proj)
        if library.in_preview(ch):
            raise ValueError(f"{ch.name} is waiting for your OK on its front drawing: approve it first")
        if not ch.rigged.exists():
            raise ValueError(f"{ch.name} has no rig (rigged.glb)")
    # drawn characters bend their drawings again (CPU); no GPU either way
    return _start(op, {"ref": asset_ref(a), **args}, project=proj, gpu=False,
                  title=f"{title}: {ch.name}", estimate_min=2.0, wait_s=wait_s)


def add_animation(ref, names: list[str], replace: bool = False, wait_s: float | None = None,
                  project: str | None = None) -> dict:
    if not names:
        raise ValueError("Name the clips to add (motions() lists them)")
    return _anim_job("add_animation", ref, project, wait_s, "Add " + ", ".join(names),
                     names=list(names), replace=bool(replace))


def add_mocap(ref, names: list[str], replace: bool = True, wait_s: float | None = None,
              project: str | None = None) -> dict:
    if not names:
        raise ValueError("Name the mocap clips (motions() lists them) or give .bvh paths")
    return _anim_job("add_mocap", ref, project, wait_s, "Mocap " + ", ".join(map(str, names)),
                     names=list(names), replace=bool(replace))


def add_springs(ref, options: dict | None = None, wait_s: float | None = None,
                project: str | None = None) -> dict:
    return _anim_job("add_springs", ref, project, wait_s, "Capes/hair swing", options=options or {})


# --------------------------------------------------------------------------- reskin / restyle

def reskin(ref, name: str, hue: float | None = None, colour: str | None = None,
           source_colour: str | None = None, saturation: float = 1.0, lightness: float = 1.0,
           project: str | None = None) -> dict:
    """A library character in new colours, as a new character (no GPU, seconds)."""
    from . import reskin as R
    proj = _project(project)
    with A.using_project(proj):
        ch = character(ref, proj)
        new = R.reskin(ch, name, hue=hue, target_colour=colour, source_colour=source_colour,
                       saturation=saturation, lightness=lightness)
        a = resolve(Path(new.folder), project=proj)
        return {"asset": asset_summary(a, proj)}


@J.register("restyle", plan=lambda a: (None, 300.0))
def _op_restyle(args, tracker):
    lines = A.restyle(A.current_project(), tuple(args.get("kinds") or ("character", "item", "prop")),
                      progress=tracker)
    return {"done": lines}


def restyle(kinds: list[str] | None = None, dry_run: bool = True, wait_s: float | None = None,
            project: str | None = None) -> dict:
    """Redo the project's characters/items/props in its CURRENT house style. dry_run: say what."""
    proj = _project(project)
    if dry_run:
        return {"project": proj.name, "plan": A.restyle_plan(proj, A.Style(), proj.style)}
    drawn = any("GPU" in ln for ln in A.restyle_plan(proj, A.Style(), proj.style))
    return _start("restyle", {"kinds": kinds}, project=proj, gpu=drawn, title="Restyle",
                  estimate_min=10.0, wait_s=wait_s)


# --------------------------------------------------------------------------- game 3D model

@J.register("game_model", plan=lambda a: (None, (90 if a.get("_shape") else 0) + 120.0))
def _op_game_model(args, tracker):
    from . import gamemodel
    a = resolve(args["ref"])
    opts = {k: v for k, v in (args.get("options") or {}).items() if v is not None}
    gm = gamemodel.game_model_for(a, progress=tracker, client=J.manager().client, **opts)
    folder = Path(a.folder) / gamemodel.FOLDER
    files = sorted(str(p) for p in folder.glob("*") if p.is_file())
    out = {"asset": asset_ref(a), "folder": str(folder), "files": files}
    for name in ("preview.png", "wire.png"):
        if (folder / name).exists():
            out[name.split(".")[0]] = str(folder / name)
    info = folder / "info.json"
    if info.exists():
        try:
            d = json.loads(info.read_text(encoding="utf-8"))
            out["info"] = {k: d[k] for k in ("look", "lods", "triangles", "bones", "clips", "kind")
                           if k in d}
        except (OSError, ValueError):
            pass
    _ = gm
    return out


def game_model(ref, look: str | None = None, wait_s: float | None = None,
               project: str | None = None) -> dict:
    """Game-ready 3D model (LODs, UVs, baked textures, skin + clips) of a character/item/prop.
    Picture-only drawn props/items first get a 3D shape on the GPU (~1.5 min)."""
    from . import gamemodel
    proj = _project(project)
    with A.using_project(proj):
        a = resolve(ref, project=proj)
        if a.kind not in ("character", "item", "prop"):
            raise ValueError("Game 3D models are made of characters, items and props")
        shape = gamemodel.needs_shape(a)
    return _start("game_model", {"ref": asset_ref(a), "options": {"look": look}, "_shape": shape},
                  project=proj, gpu=shape, title=f"Game 3D model: {a.name}",
                  estimate_min=(1.5 if shape else 0) + 2.0, wait_s=wait_s)


# --------------------------------------------------------------------------- maps

def _map_text(doc) -> dict:
    from .diorama import TERRAINS
    letters = {}
    used = set()
    for t in TERRAINS:
        ch = t[0]
        if ch in used:
            ch = next(c for c in t[1:] + "0123456789" if c not in used)
        used.add(ch)
        letters[t] = ch
    d = doc.diorama
    rows = ["".join(letters.get(d.terrain[r, c], "?") for c in range(d.cols)) for r in range(d.rows)]
    heights = ["".join(str(int(min(9, max(0, d.height[r, c])))) for c in range(d.cols)) for r in range(d.rows)]
    return {"cols": d.cols, "rows": d.rows, "legend": {v: k for k, v in letters.items()},
            "terrain": rows, "height": heights,
            "props": [{"kind": k, "col": c, "row": r} for k, c, r in d.props][:200],
            "spawns": [{"team": t, "col": c, "row": r} for t, c, r in doc.spawns]}


def map_info(ref, project: str | None = None) -> dict:
    from .maplib import PROP_KINDS, load_map
    proj = _project(project)
    a = resolve(ref, "map", proj)
    doc = load_map(a)
    return {"id": asset_ref(a), "name": a.name, "template": doc.template, **_map_text(doc),
            "prop_kinds": list(PROP_KINDS), "previews": write_previews(a, proj)}


def map_edit(ref, edits: list[dict], project: str | None = None) -> dict:
    """Apply edits and save: each {"op": paint|rect|fill|raise|lower|height|prop|erase_prop|spawn|
    clear_spawn|resize, "col", "row", ...}: paint {terrain, brush}, rect {col, row, col2, row2,
    terrain | level}, fill {terrain}, raise/lower {amount, brush}, height {level, brush},
    prop {kind}, spawn {team 0|1}, resize {cols, rows}."""
    from .diorama import TERRAINS
    from .maplib import PROP_KINDS, MapEdit, load_map, save_map
    proj = _project(project)
    with A.using_project(proj):
        a = resolve(ref, "map", proj)
        doc = load_map(a)
        ed = MapEdit(doc)
        done = 0
        for e in edits or []:
            op = str(e.get("op", "")).lower()
            c, r = int(e.get("col", 0)), int(e.get("row", 0))
            brush = int(e.get("brush", 1))
            if op == "paint":
                ed.paint(c, r, e["terrain"], brush)
            elif op == "rect":
                c2, r2 = int(e.get("col2", c)), int(e.get("row2", r))
                for y in range(min(r, r2), max(r, r2) + 1):
                    for x in range(min(c, c2), max(c, c2) + 1):
                        if 0 <= x < doc.cols and 0 <= y < doc.rows:
                            if "terrain" in e:
                                if e["terrain"] not in TERRAINS:
                                    raise ValueError(f"Unknown terrain {e['terrain']!r} ({', '.join(TERRAINS)})")
                                ed.paint(x, y, e["terrain"])
                            if "level" in e:
                                ed.set_height(x, y, int(e["level"]))
            elif op == "fill":
                ed.fill(c, r, e["terrain"])
            elif op == "raise":
                ed.raise_cells(c, r, int(e.get("amount", 1)), brush)
            elif op == "lower":
                ed.lower_cells(c, r, int(e.get("amount", 1)), brush)
            elif op == "height":
                ed.set_height(c, r, int(e["level"]), brush)
            elif op == "prop":
                if e.get("kind") not in PROP_KINDS:
                    raise ValueError(f"Unknown prop {e.get('kind')!r} ({', '.join(PROP_KINDS)})")
                ed.put_prop(c, r, e["kind"])
            elif op == "erase_prop":
                ed.erase_prop(c, r)
            elif op == "spawn":
                ed.set_spawn(c, r, int(e.get("team", 0)))
            elif op == "clear_spawn":
                ed.clear_spawn(c, r)
            elif op == "resize":
                ed.resize(int(e["cols"]), int(e["rows"]))
            else:
                raise ValueError(f"Unknown map edit op {op!r}")
            done += 1
        save_map(doc, asset=a, project=proj)
        out = map_info(asset_ref(a), proj.name)
        out["edits"] = done
        return out


# --------------------------------------------------------------------------- export into a game

def _export_items(refs, collection, kinds, project: A.Project) -> list[A.Asset]:
    items: list[A.Asset] = []
    if refs:
        items += [resolve(r, project=project) for r in refs]
    if collection:
        items += A.list_assets(collection=collection, project=project)
    if kinds:
        for k in kinds:
            items += A.list_assets(k, project=project)
    if not (refs or collection or kinds):
        items = A.list_assets(project=project)
    seen, out = set(), []
    for a in items:
        key = str(a.folder)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


@J.register("export", plan=lambda a: (None, 20.0 + 5 * len(a.get("refs") or [])))
def _op_export(args, tracker):
    from .gameexport import export_to_game
    proj = A.current_project()
    items = _export_items(args.get("refs"), args.get("collection"), args.get("kinds"), proj)
    rep = export_to_game(items, args["engine"], args["project_path"], args.get("subdir", ""),
                         project=proj, force=bool(args.get("force")), models=bool(args.get("models")),
                         build_models=bool(args.get("build_models")),
                         pixel_filter=bool(args.get("pixel_filter", True)), preset=args.get("preset"),
                         ppu=args.get("ppu"), map_target=args.get("map_target"), progress=tracker)
    if args.get("verify") and args["engine"] == "godot":
        from .gameexport import find_godot, verify_godot
        if find_godot():
            tracker.say("Checking the import in Godot (headless)…")
            v = verify_godot(args["project_path"], rep["subdir"])
            rep["verify"] = {"ok": v["ok"], "failed": {k: x["failed"] for k, x in v["assets"].items()
                                                      if not x.get("ok")}, "errors": v["errors"][:10]}
        else:
            rep["warnings"].append("verify skipped: Godot not found (set GODOT_BIN)")
    return rep


def export(refs: list[str] | None = None, engine: str | None = None, project_path: str = "",
           subdir: str = "", collection: str | None = None, kinds: list[str] | None = None,
           force: bool = False, models: bool = False, build_models: bool = False,
           verify: bool = False, preset: str | None = None, wait_s: float | None = None,
           project: str | None = None) -> dict:
    """Write assets into an existing Godot/Unity project + pixel_pipeline_manifest.json."""
    from .gameexport import check_project
    proj = _project(project)
    engine = engine or proj.style.engine or "godot"
    check_project(engine, project_path)
    with A.using_project(proj):
        items = _export_items(refs, collection, kinds, proj)
        if not items:
            raise ValueError("Nothing to export")
        shape = False
        if models and build_models:
            from . import gamemodel
            shape = any(gamemodel.needs_shape(a) for a in items
                        if a.kind in ("item", "prop") and not gamemodel.has_game_model(a))
    args = {"refs": [asset_ref(a) for a in items], "engine": engine, "project_path": str(project_path),
            "subdir": subdir, "force": force, "models": models, "build_models": build_models,
            "verify": verify, "preset": preset}
    return _start("export", args, project=proj, gpu=shape,
                  title=f"Export {len(items)} asset(s) to {engine}", estimate_min=0.5 + 0.05 * len(items),
                  wait_s=DEFAULT_WAIT_CPU if wait_s is None and not shape else wait_s)


@J.register("verify_godot", plan=lambda a: (None, 60.0))
def _op_verify(args, tracker):
    from .gameexport import verify_godot
    tracker.say("Importing the project in Godot (headless)…")
    return verify_godot(args["project_path"], args.get("subdir", ""))


def verify_godot(project_path: str, subdir: str = "", wait_s: float | None = None,
                 project: str | None = None) -> dict:
    from .gameexport import check_project
    check_project("godot", project_path)
    proj = _project(project)
    return _start("verify_godot", {"project_path": str(project_path), "subdir": subdir}, project=proj,
                  gpu=False, title="Verify the Godot import", estimate_min=1.0,
                  wait_s=DEFAULT_WAIT_CPU if wait_s is None else wait_s)


# --------------------------------------------------------------------------- jobs

def job_status(job_id: str, wait_s: float = 0.0) -> dict:
    m = J.manager()
    if wait_s:
        m.wait(job_id, float(wait_s))
    d = job_view(job_id)
    rec = m.get(job_id)
    if rec.status == "running" and rec.op.startswith("character"):
        try:
            a = resolve(rec.args["ref"], project=rec.project)
            keys = Path(a.folder) / "keys"
            newest = max(keys.glob("drawn_*.png"), key=lambda p: p.stat().st_mtime, default=None)
            if newest is not None:
                d["latest_drawing"] = str(newest)
        except Exception:  # noqa: BLE001
            pass
    if rec.log and rec.status in ("failed", "cancelled"):
        d["log"] = rec.log[-8:]
    return d


def list_jobs(status: str | None = None, project: str | None = None, limit: int = 20) -> dict:
    recs = J.manager().list(status, project, limit)
    rows = []
    for r in recs:
        row = {"job": r.id, "op": r.op, "title": r.title, "status": r.status, "project": r.project,
               "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created))}
        if r.status in J.ACTIVE:
            row["percent"] = r.percent
            if r.eta_s is not None and r.status == "running":
                row["eta"] = J.P.eta_text(r.eta_s)
        if r.error:
            row["error"] = r.error.splitlines()[0][:200]
        rows.append(row)
    holder = J.lock_holder()
    return {"jobs": rows, "gpu_lock": holder or "free",
            "interrupted": sum(1 for r in recs if r.status == "interrupted")}


def cancel_job(job_id: str) -> dict:
    J.manager().cancel(job_id)
    return job_view(job_id)


def resume_jobs(job_ids: list[str] | None = None) -> dict:
    recs = J.manager().resume(set(job_ids) if job_ids else None)
    return {"resumed": [r.id for r in recs], "jobs": [job_view(r.id) for r in recs]}


# --------------------------------------------------------------------------- what can be made

def kinds() -> dict:
    """Every kind make() knows, its editable settings and the GPU table (minutes, 4070 laptop)."""
    return {"kinds": {k: {"label": M.LABELS[k], "fields": _fields(k)} for k in M.KINDS},
            "gpu_minutes_table": M.GPU,
            "notes": {"character": "drawn front first (review=true): ~3.3 min model + front, then "
                                   "~12 min after character_approve for 16 directions",
                      "cpu_only": ["ui", "map", "effect (paint=false)", "weather", "water",
                                   "background (no words)"]}}
