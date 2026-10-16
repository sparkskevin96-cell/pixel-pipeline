"""Screenshot of the progress bars on the Characters page mid-job, offscreen, no GPU.

    python tools/progress_shot.py [out.png]

A fake "Add to library" (drawn, 16 directions) reports what the real one says: the skeleton,
then drawing 9 of 25 with the GPU picture at sampler step 14/25, so the page's panel (overall +
stage bars, about how long is left, Stop) and the main window's compact bar + % title show.
Copies knight-v3's model + picture into a temp library (read-only for the real one).
Writes C:/Users/spark/Pictures/PixelPipeline/progress_ui.png.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

KNIGHT = Path(r"C:\Users\spark\Pixel Pipeline\Projects\Neo16 test\characters\knight-v3")
OUT = Path(r"C:\Users\spark\Pictures\PixelPipeline\progress_ui.png")


def main(out: Path = OUT) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    os.environ["PIXEL_WARM"] = "0"
    os.environ["PIXEL_DRAWN"] = "1"
    tmp = Path(tempfile.mkdtemp(prefix="pp-progress-shot-"))
    lib = tmp / "lib"
    if KNIGHT.exists():                          # a character on the left, the knight on show
        shutil.copytree(KNIGHT, lib / "knight-v3", ignore=shutil.ignore_patterns("keys", "*.aseprite"))
    os.environ["PIXEL_LIBRARY"] = str(lib)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pixel_pipeline import library
    from pixel_pipeline import progress as P
    from pixel_pipeline.ui.app import MainWindow
    win = MainWindow()
    win.resize(1500, 920)
    win.tabs.setCurrentWidget(win.studio)
    win.show()
    studio = win.studio
    studio.sync = True
    chars = library.list_characters()
    if chars:
        studio.open_character(chars[0])
        studio._anim_timer.stop()
        studio._spin_timer.stop()
    studio.character = None
    studio.name.setText("Knight (copy)")
    studio.prompt.setText("a knight with a red cape and a big sword")
    studio.drawn.setChecked(True)
    studio.front_first.setChecked(False)
    if studio.result is None:
        from pixel_pipeline.model3d import ModelResult
        m = tmp / "m.glb"
        m.write_bytes(b"x")
        studio.result = ModelResult(m, None, None, None)

    def fake_add(name, model, progress=None, **kw):
        progress("Adding a skeleton and animations…")
        progress("Adding the project's clips: idle, walk, run, attack, hurt, death…")
        for n in range(1, 10):
            eta = f", about {round((26 - n) * 26 / 60)} min left" if n > 1 else ""
            progress(f"Drawing it facing {['S', 'E', 'W', 'N', 'SE', 'SW', 'NE', 'NW', 'SSE'][n - 1]} "
                     f"(beside S, try o0.6; {n} of 25{eta})…")
        with P.using(P.current()):
            P.gpu_pictures(0, 1)
            P.gpu_sampling(14, 25)
        for _ in range(5):
            app.processEvents()
        studio.progress_panel.refresh()
        win.job_progress.show_update(studio.progress_panel.last)
        app.processEvents()
        out.parent.mkdir(parents=True, exist_ok=True)
        win.grab().save(str(out))
        raise P.Stopped("Stopped (screenshot taken)")
    library.add_character = fake_add
    studio.add_to_library()
    win.close()
    print(out)
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
