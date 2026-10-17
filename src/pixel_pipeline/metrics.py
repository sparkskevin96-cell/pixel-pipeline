"""How long making a character takes (2026-10-17, Kevin: "how long to create a character:
currently X, after Y").

Every character creation records its stage times in the character (asset.json extra["timing"],
mirrored into keys/keys.json "timing"):

    {"stages": {"model": s, "rig": s, "front": s, "drawings": s, "quality_check": s,
                "quality_redraw": s, "bending": s, ...},
     "gpu_s": seconds on the GPU (picture + 3D shape + every drawing),
     "wall_s": the sum of the stages = the time the app worked on it, EXCLUDING the time it
               waited for the user's front-first OK (that wait is not a stage),
     "runs": how many sittings it took (preview, approve, a later fix), "updated": time}

`stage(ch_folder, name, gpu=False)` is the context manager the pipeline wraps its steps in; a
stage that runs again (a later quality fix) adds to the time already there. `summary(folder)`
reads one; `aggregate(n)` = median/mean per stage over the last n characters of every project
(CLI `pixel-pipeline metrics --json`, agent.metrics / MCP character_metrics).
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

GPU_STAGES = {"model", "front", "drawings", "quality_redraw"}
ORDER = ("model", "rig", "front", "drawings", "quality_check", "quality_redraw", "bending", "sheets")
LABELS = {"model": "picture + 3D shape", "rig": "skeleton + clips", "front": "front drawing + preview",
          "drawings": "other directions drawn", "quality_check": "per-direction quality check",
          "quality_redraw": "redraws of failing directions", "bending": "bending every animation (+ clean-up)",
          "sheets": "sheets / normals / glow written"}


def _asset_json(folder: Path) -> Path:
    return Path(folder) / "asset.json"


def read(folder) -> dict:
    p = _asset_json(folder)
    if not p.exists():
        return {}
    try:
        return dict((json.loads(p.read_text(encoding="utf-8")).get("extra") or {}).get("timing") or {})
    except (OSError, ValueError):
        return {}


def write(folder, timing: dict) -> None:
    folder = Path(folder)
    p = _asset_json(folder)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data.setdefault("kind", "character")
    data.setdefault("extra", {})["timing"] = timing
    folder.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    keys = folder / "keys" / "keys.json"
    if keys.exists():
        try:
            meta = json.loads(keys.read_text(encoding="utf-8"))
            meta["timing"] = timing
            keys.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass


def _finish(t: dict) -> dict:
    st = t.setdefault("stages", {})
    t["wall_s"] = round(sum(st.values()), 1)
    t["gpu_s"] = round(sum(v for k, v in st.items() if k in GPU_STAGES), 1)
    t["updated"] = time.time()
    return t


def add(folder, name: str, seconds: float, *, new_run: bool = False) -> dict:
    """Add seconds to a stage of the character in folder."""
    t = read(folder)
    st = t.setdefault("stages", {})
    st[name] = round(float(st.get(name, 0.0)) + float(seconds), 2)
    if new_run or "runs" not in t:
        t["runs"] = int(t.get("runs", 0)) + 1
    _finish(t)
    write(folder, t)
    return t


def merge(folder, stages: dict) -> dict:
    t = read(folder)
    st = t.setdefault("stages", {})
    for k, v in (stages or {}).items():
        st[k] = round(float(st.get(k, 0.0)) + float(v), 2)
    t.setdefault("runs", 1)
    _finish(t)
    write(folder, t)
    return t


@contextmanager
def stage(folder, name: str):
    """Time a stage of making the character in folder (added to what it had)."""
    t0 = time.time()
    try:
        yield
    finally:
        try:
            add(folder, name, time.time() - t0)
        except OSError:
            pass


def summary(folder) -> dict:
    t = read(folder)
    if not t:
        return {}
    return {"wall_min": round(t.get("wall_s", 0) / 60, 1), "gpu_min": round(t.get("gpu_s", 0) / 60, 1),
            "stages_min": {k: round(v / 60, 2) for k, v in sorted(
                (t.get("stages") or {}).items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99)},
            "runs": t.get("runs", 1)}


def made_in(folder) -> str:
    """'made in 14 min (11 on the GPU)' or ''."""
    s = summary(folder)
    if not s or not s["wall_min"]:
        return ""
    return f"made in {s['wall_min']:.0f} min ({s['gpu_min']:.0f} on the GPU)" if s["wall_min"] >= 1 else \
        f"made in {s['wall_min'] * 60:.0f} s"


def _characters(root: Path | None = None):
    from . import assets
    roots = [Path(root)] if root else [Path(p.folder) for p in assets.list_projects()]
    for r in roots:
        for c in (r / "characters").glob("*") if (r / "characters").exists() else ():
            t = read(c)
            if t.get("stages"):
                yield c, t


def aggregate(n: int = 10, root=None) -> dict:
    """median / mean minutes per stage over the last n timed characters (newest by update)."""
    import statistics
    rows = sorted(_characters(root), key=lambda ct: -float(ct[1].get("updated", 0)))[:n]
    stages = {}
    for _, t in rows:
        for k, v in (t.get("stages") or {}).items():
            stages.setdefault(k, []).append(v / 60)
    walls = [t.get("wall_s", 0) / 60 for _, t in rows]
    gpus = [t.get("gpu_s", 0) / 60 for _, t in rows]

    def stat(v):
        return {"median": round(statistics.median(v), 2), "mean": round(statistics.mean(v), 2), "n": len(v)}
    return {"characters": [{"folder": str(c), "name": c.name, **summary(c)} for c, _ in rows],
            "count": len(rows),
            "wall_min": stat(walls) if walls else None, "gpu_min": stat(gpus) if gpus else None,
            "stages_min": {k: stat(v) for k, v in sorted(stages.items(), key=lambda kv: ORDER.index(kv[0])
                                                          if kv[0] in ORDER else 99)},
            "labels": LABELS}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pixel-pipeline metrics", description="time to make a character")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = aggregate(a.n)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print(format_report(rep))
    return 0


def format_report(rep: dict) -> str:
    if not rep["count"]:
        return "No timed characters yet."
    lines = [f"Last {rep['count']} characters: median {rep['wall_min']['median']} min "
             f"({rep['gpu_min']['median']} on the GPU)"]
    for k, s in rep["stages_min"].items():
        lines.append(f"  {LABELS.get(k, k):42s} median {s['median']:6.2f} min  mean {s['mean']:6.2f}  (n={s['n']})")
    return "\n".join(lines)
