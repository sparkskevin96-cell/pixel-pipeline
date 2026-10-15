"""Screenshot of the drawn-character review panel (ui/character_review.py), offscreen.

    python tools/review_shot.py                     # from Neo16 test / knight-v3 (read-only copy)
    python tools/review_shot.py <draft folder>      # a library character waiting for its OK

The knight's own S drawings are reused (no GPU): a copy of its rig + drawn_S_* files is made
in a temp library and the preview (S key + bent idle, CPU) is built from them.
Writes C:/Users/spark/Pictures/PixelPipeline/character_review_ui.png (or the 2nd argument).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

KNIGHT = Path(r"C:\Users\spark\Pixel Pipeline\Projects\Neo16 test\characters\knight-v3")
OUT = Path(r"C:\Users\spark\Pictures\PixelPipeline\character_review_ui.png")


def knight_draft(tmp: Path):
    """A draft copy of knight-v3 (front only) in a temp library, its preview built on the CPU."""
    from pixel_pipeline import library
    from pixel_pipeline.assets import Project
    proj = Project("Neo16 test", KNIGHT.parent.parent)
    library.style = lambda: proj.style
    dst = tmp / "lib" / "knight-v3"
    (dst / "keys").mkdir(parents=True)
    for name in ("character.json", "model.glb", "picture.png", "rigged.glb", "thumb.png"):
        if (KNIGHT / name).exists():
            shutil.copyfile(KNIGHT / name, dst / name)
    for f in (KNIGHT / "keys").glob("drawn_S_*.png"):
        shutil.copyfile(f, dst / "keys" / f.name)
    old = json.loads((KNIGHT / "keys" / "keys.json").read_text(encoding="utf-8"))
    (dst / "keys" / "keys.json").write_text(json.dumps(
        {"camera": old["camera"], "partial": True, "prompt": old.get("prompt", ""),
         "seeds": {"S": old.get("seeds", {}).get("S", 1234)}}), encoding="utf-8")
    ch = library.from_folder(dst)
    library._draw_preview(ch, print)
    return library.from_folder(dst)


def shoot(ch, out: Path) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    from PySide6.QtWidgets import QApplication
    from pixel_pipeline.ui.character_review import CharacterReview
    app = QApplication.instance() or QApplication([])
    w = CharacterReview(lambda job, done: done(job(print)), print)
    w.setStyleSheet("CharacterReview{background:#252a3a;} QLabel{color:#e4e8f0;} "
                    "QGroupBox{color:#c9d1e0;}")
    w.setAutoFillBackground(True)
    w.resize(1180, 780)
    w.set_character(ch)
    w.show()
    for _ in range(6):                   # let the idle play a few frames
        app.processEvents()
        w._next_frame()
    app.processEvents()
    out.parent.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(out))
    print("wrote", out)
    return out


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="pp-review-"))
    os.environ["PIXEL_LIBRARY"] = str(tmp / "lib")
    os.environ.setdefault("PIXEL_CACHE", str(tmp / "cache"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        from pixel_pipeline import library
        ch = library.from_folder(Path(sys.argv[1]))
    else:
        ch = knight_draft(tmp)
    shoot(ch, out)
