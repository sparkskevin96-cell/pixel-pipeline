"""Prove the Unity export in a real Unity editor (batch mode, no GPU work).

    .venv/Scripts/python tools/verify_unity/verify_unity.py [asset folders...]
        [--unity <Unity.exe>] [--project <folder>] [--reference]

Makes (once) a throwaway Unity project, exports each asset with the "unity" preset into
Assets/PP/<name>/, copies Verify.cs into Assets/Editor/, writes expected.json (+ the expected
pixels of every frame, cropped top-down from the sheet PNG) and runs
`Unity -batchmode -nographics -executeMethod Verify.Run -quit`. Verify.cs writes
results.json; this prints it and exits 0 only if every check passed.
--reference instead runs Verify.Reference: Unity re-serialises the exported .meta files and
saves a clip made through its API (Assets/PPRef/ref.anim), and the diffs are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from pixel_pipeline import exports  # noqa: E402

UNITY = r"C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe"
LIB = Path.home() / "Pixel Pipeline" / "Projects" / "Default"
DEFAULT_ASSETS = [LIB / "characters" / "knight", LIB / "items" / "health-potion"]


def run_unity(unity: str, args: list[str], log: Path, timeout: int = 900) -> int:
    cmd = [unity, "-batchmode", "-nographics", *args, "-logFile", str(log)]
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd, timeout=timeout).returncode


def ensure_project(unity: str, project: Path) -> None:
    if (project / "ProjectSettings").exists():
        return
    project.parent.mkdir(parents=True, exist_ok=True)
    code = run_unity(unity, ["-quit", "-createProject", str(project)], project.parent / "create.log")
    if code != 0 or not (project / "ProjectSettings").exists():
        raise SystemExit(f"Unity could not create the project (exit {code}); see "
                         f"{project.parent / 'create.log'} (licence?)")


def prepare(project: Path, assets: list[Path], **opts) -> list[dict]:
    pp = project / "Assets" / "PP"
    if pp.exists():
        shutil.rmtree(pp)
    for stale in (project / "Assets" / "PP.meta", project / "results.json"):
        stale.unlink(missing_ok=True)
    crops = project / "expected_crops"
    if crops.exists():
        shutil.rmtree(crops)
    (project / "Assets" / "Editor").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "Verify.cs", project / "Assets" / "Editor" / "Verify.cs")
    expected = []
    for folder in assets:
        src = exports.source(folder)
        out = pp / src.file_name
        files = exports.export_asset(folder, "unity", out, **opts)
        rel = lambda p: p.relative_to(project).as_posix()  # noqa: E731
        png = out / f"{src.file_name}.png"
        w, h = Image.open(png).size
        # expected pixels come from the ORIGINAL sheet; rects from the (maybe repacked) layout
        img = np.asarray(Image.open(src.image).convert("RGBA"))
        layout = exports.unity_layout(src.frames, Image.open(src.image).size) if src.sheet else None
        rects = layout[0] if layout else src.frames
        names = exports.unity_sprite_names(src.frames)
        cdir = crops / src.file_name
        cdir.mkdir(parents=True)
        frames = []
        multiple = src.sheet is not None
        for i, (f, r, n) in enumerate(zip(src.frames, rects, names)):
            crop = cdir / f"{i}.png"
            Image.fromarray(np.ascontiguousarray(img[f.y:f.y + f.h, f.x:f.x + f.w]), "RGBA").save(crop)
            frames.append({"name": n if multiple else png.stem, "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                           "crop": rel(crop)})
        normal = out / f"{src.file_name}_n.png"
        meta = (out / f"{png.name}.meta").read_text(encoding="utf-8")
        ppu = float(meta.split("spritePixelsToUnits: ", 1)[1].split()[0])
        clips = []
        for c in exports.unity_clip_plan(src):
            path = out / "clips" / f"{c['name']}.anim"
            if path in files:
                clips.append({"path": rel(path), "sprites": c["sprites"], "times": c["times"],
                              "length": c["length"], "fps": c["fps"], "loop": c["loop"]})
        expected.append({"name": src.name, "png": rel(png), "normal": rel(normal) if normal.exists() else "",
                         "width": w, "height": h, "ppu": ppu, "multiple": multiple,
                         "frames": frames, "clips": clips,
                         "singles": [rel(p) for p in files if p.suffix == ".png" and p not in (png, normal)]})
    (project / "expected.json").write_text(json.dumps({"assets": expected}, indent=1), encoding="utf-8")
    return expected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", nargs="*", type=Path)
    ap.add_argument("--unity", default=os.environ.get("UNITY_BIN", UNITY))
    ap.add_argument("--project", type=Path,
                    default=Path(tempfile.gettempdir()) / "pixel-pipeline-unity-verify" / "VerifyProj")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--no-clips", action="store_true")
    a = ap.parse_args(argv)
    assets = a.assets or DEFAULT_ASSETS
    ensure_project(a.unity, a.project)
    prepare(a.project, assets, clips=not a.no_clips)
    log = a.project / "verify.log"
    if a.reference:
        before = {p: p.read_text(encoding="utf-8") for p in (a.project / "Assets" / "PP").rglob("*.png.meta")}
        code = run_unity(a.unity, ["-projectPath", str(a.project), "-executeMethod", "Verify.Reference", "-quit"], log)
        import difflib
        for p, old in before.items():
            new = p.read_text(encoding="utf-8")
            diff = list(difflib.unified_diff(old.splitlines()[:400], new.splitlines()[:400], str(p.name), "unity", lineterm="", n=1))
            print(f"--- {p.name}: {'same' if old == new else f'{len(diff)} diff lines'}")
            print("\n".join(diff[:150]))
        ref = a.project / "Assets" / "PPRef" / "ref.anim"
        print(ref.read_text(encoding="utf-8") if ref.exists() else "no ref.anim")
        return code
    code = run_unity(a.unity, ["-projectPath", str(a.project), "-executeMethod", "Verify.Run", "-quit"], log, )
    res_path = a.project / "results.json"
    if not res_path.exists():
        print(f"Unity wrote no results (exit {code}); log: {log}")
        return 2
    res = json.loads(res_path.read_text(encoding="utf-8"))
    print(f"Unity {res['unity']}")
    for c in res["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['asset']}: {c['name']}  {c['detail']}")
    print("ALL OK" if res["all_ok"] else "SOME CHECKS FAILED")
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
