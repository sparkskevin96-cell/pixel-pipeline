"""Prove the 3D game model's Unity export in a real Unity editor (batch mode, no GPU work).

    .venv/Scripts/python tools/verify_unity/verify_model.py <asset folder> [<asset folder>...]
        [--unity <Unity.exe>] [--project <folder>]

Exports each asset with exports.export_asset(..., "unity-3d-model") into Assets/Models/m<i>/
(builds model_game/ first if missing), copies VerifyModel.cs into Assets/Editor/, writes
expected_model.json and runs `Unity -batchmode -nographics -executeMethod VerifyModel.Run
-quit`: prefab, LODGroup, triangle counts, textures + filter, bones, clips, a clip deforms the
skinned mesh, loop flags, controller. Exit 0 only if every check passed.
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
from pixel_pipeline import exports  # noqa: E402
from verify_unity import UNITY, ensure_project, run_unity  # noqa: E402


def prepare(project: Path, folders: list[Path]) -> dict:
    base = project / "Assets" / "Models"
    for stale in (base, project / "Assets" / "Models.meta", project / "results_model.json"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    (project / "Assets" / "Editor").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "VerifyModel.cs", project / "Assets" / "Editor" / "VerifyModel.cs")
    models = []
    for i, folder in enumerate(folders):
        out = base / f"m{i}"
        files = exports.export_asset(folder, "unity-3d-model", out)
        pp = next(p for p in files if p.suffix == ".ppmodel")
        info = json.loads((Path(folder) / "model_game" / "info.json").read_text(encoding="utf-8"))
        models.append({"path": pp.relative_to(project).as_posix(), "skinned": info["skinned"],
                       "bones": info["bones"], "clips": info["clips"],
                       "tris": [lod["tris"] for lod in info["lods"]],
                       "filter": "point" if info["look"] == "pixel" else "bilinear"})
    for extra in sorted(base.glob("*/Editor"))[1:]:      # one importer per project
        shutil.rmtree(extra)
    exp = {"models": models}
    (project / "expected_model.json").write_text(json.dumps(exp, indent=1), encoding="utf-8")
    return exp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--unity", default=os.environ.get("UNITY_BIN", UNITY))
    ap.add_argument("--project", type=Path,
                    default=Path(tempfile.gettempdir()) / "pixel-pipeline-unity-verify" / "VerifyProj")
    a = ap.parse_args(argv)
    ensure_project(a.unity, a.project)
    prepare(a.project, a.folders)
    log = a.project / "verify_model.log"
    code = run_unity(a.unity, ["-projectPath", str(a.project), "-executeMethod", "VerifyModel.Run",
                               "-quit"], log)
    res_path = a.project / "results_model.json"
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
    sys.exit(main())
