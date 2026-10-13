"""Prove the UI kit's Unity export in a real Unity editor (batch mode, no GPU work).

    .venv/Scripts/python tools/verify_unity/verify_uikit.py [--theme wood] [--scale 2]
        [--unity <Unity.exe>] [--project <folder>]

Makes (once) a throwaway Unity project (same one verify_unity.py uses), exports a kit with
uikit.export_kit(..., "unity") into Assets/UIKit/, copies VerifyUIKit.cs into Assets/Editor/,
writes expected_uikit.json and runs `Unity -batchmode -nographics -executeMethod
VerifyUIKit.Run -quit`: sprites + 9-slice borders, the Font built from the .fnt, the demo
Canvas. Prints results_uikit.json; exit 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from pixel_pipeline import uikit  # noqa: E402
from pixel_pipeline.assets import Style  # noqa: E402
from verify_unity import UNITY, ensure_project, run_unity  # noqa: E402


def prepare(project: Path, theme: str, scale: int) -> dict:
    out = project / "Assets" / "UIKit"
    for stale in (out, project / "Assets" / "UIKit.meta", project / "results_uikit.json"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    manifest = project / "Packages" / "manifest.json"
    deps = json.loads(manifest.read_text(encoding="utf-8"))
    if "com.unity.ugui" not in deps["dependencies"]:      # -createProject leaves it out
        deps["dependencies"]["com.unity.ugui"] = "2.0.0"
        manifest.write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (project / "Assets" / "Editor").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "VerifyUIKit.cs", project / "Assets" / "Editor" / "VerifyUIKit.cs")
    kit = uikit.make_kit(theme, Style(), scale)
    uikit.export_kit(kit, out, "unity")
    data = json.loads((out / "kit.json").read_text(encoding="utf-8"))
    rel = "Assets/UIKit/"
    exp = {"kit": rel + "kit.json", "font": rel + data["font"]["file"], "chars": 95,
           "pieces": [{"file": rel + p["file"], "w": p["w"], "h": p["h"], "nine": p["nine"],
                       **p["margins"]} for p in data["pieces"]]}
    (project / "expected_uikit.json").write_text(json.dumps(exp, indent=1), encoding="utf-8")
    return exp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="wood", choices=uikit.THEME_NAMES)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--unity", default=os.environ.get("UNITY_BIN", UNITY))
    ap.add_argument("--project", type=Path,
                    default=Path(tempfile.gettempdir()) / "pixel-pipeline-unity-verify" / "VerifyProj")
    a = ap.parse_args(argv)
    ensure_project(a.unity, a.project)
    prepare(a.project, a.theme, a.scale)
    log = a.project / "verify_uikit.log"
    code = run_unity(a.unity, ["-projectPath", str(a.project), "-executeMethod", "VerifyUIKit.Run",
                               "-quit"], log)
    res_path = a.project / "results_uikit.json"
    if not res_path.exists():
        print(f"Unity wrote no results (exit {code}); log: {log}")
        return 2
    res = json.loads(res_path.read_text(encoding="utf-8"))
    print(f"Unity {res['unity']}")
    for c in res["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}  {c['detail']}")
    print("ALL OK" if res["all_ok"] else "SOME CHECKS FAILED")
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
