import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
try:
    from PySide6.QtWidgets import QApplication
except ImportError as e:  # missing system GL libs on headless machines
    pytest.skip(f"Qt unavailable: {e}", allow_module_level=True)

from pixel_pipeline.ui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_preset_and_render(app, blob_png):
    win = MainWindow()
    assert win.subject.currentText() == "object"
    win.apply_preset("person")
    assert win.subject.currentText() == "person"
    assert (win.width_.value(), win.height_.value()) == (32, 48)
    win.open_path(blob_png)
    assert win.sprite is not None and win.sprite.size == (32, 48)
    win.palette.setCurrentText("gameboy")
    win.render()
    assert len(win.sprite.palette) == 4


def test_video_import_animates_and_exports(app, tmp_path):
    from test_animation import make_video
    clip = tmp_path / "clip.mp4"
    make_video(clip)
    win = MainWindow()
    win.apply_preset("object")
    win.remove_bg.setChecked(False)
    win.load_video(clip, frames=4)
    assert win.anim is not None and len(win.anim["frames"]) == 4
    win._next_anim_frame()
    assert win._anim_i == 1
    from pixel_pipeline.export import write_animation
    written = write_animation(win.anim["frames"], tmp_path / "walk.png", win.anim["ms"], "walk",
                              aseprite=True, godot=True, palette=win.anim["palette"])
    assert all(p.exists() for p in written)


def test_model_studio_makes_saves_and_updates(app, tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from PIL import Image

    import pixel_pipeline.model3d as model3d
    from pixel_pipeline.ui.model_studio import ModelStudio

    def fake_prompt_to_model(prompt, out, *, progress=None, seed=None, **kw):
        body = trimesh.creation.box(extents=[0.6, 1.2, 0.4])
        body.visual.vertex_colors = np.tile(np.array([200, 40, 40, 255], np.uint8), (8, 1))
        body.export(out)
        pic = out.with_suffix(".front.png")
        Image.new("RGBA", (64, 64), (200, 40, 40, 255)).save(pic)
        progress("picture", "Making a front picture")
        return model3d.ModelResult(out, pic, None, 42)

    monkeypatch.setattr(model3d, "prompt_to_model", fake_prompt_to_model)
    win = ModelStudio()
    win.sync = True
    win.prompt.setText("a red box knight")
    win.directions.setCurrentText("8")
    win.make()
    assert len(win.sprites) == 8 and len(win.spin) == 8 and win.result.seed == 42
    assert not win.model_view.pixmap().isNull() and not win.sprite_view.pixmap().isNull()
    assert win.save_btn.isEnabled()
    written = win.save_all(str(tmp_path / "knight.glb"))
    names = {p.name for p in written}
    assert {"knight.glb", "knight.front.png", "knight_dirs.png", "knight_dirs.tres",
            "knight_dirs_facing.gd", "knight_dirs_spin.gif"} <= names
    win.directions.setCurrentText("4")
    win.update_sprites()
    assert len(win.sprites) == 4
    assert win.animate.isEnabled() and "skeleton" in win.animate.text()   # can be auto-rigged


def test_model_studio_plays_a_rigged_model(app, tmp_path):
    from test_anim3d import write_robot
    from pixel_pipeline.ui.model_studio import ModelStudio
    robot = write_robot(tmp_path / "robot.glb")
    win = ModelStudio()
    win.sync = True
    win.directions.setCurrentText("4")
    win.width_.setValue(24)
    win.height_.setValue(24)
    win.load_model(robot)
    assert win.animate.isEnabled() and win.anim is None and win.anim_names == ["wave", "slide", "fall"]
    win.animate.setChecked(True)
    win.frames.setValue(3)
    win.update_sprites()
    assert win.anim is not None and win.anim.frames == 3 and len(win.anim.actions) == 3
    for _ in range(4):
        win._next_anim_frame()
    assert win._anim_i == 4 and "wave" in win.sprite_view.toolTip()
    names = {p.name for p in win.save_all(str(tmp_path / "out" / "robot.glb"))}
    assert {"robot_dirs.tres", "robot_dirs_facing.gd", "robot_dirs_anim.gif"} <= names


def test_main_window_has_every_tool(app):
    win = MainWindow()
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Sprite", "Prompt", "Directions", "Teams", "Portrait", "Map tiles",
                      "Godot demo", "Prompt to Model"]
    win.open_model_studio()
    assert win.tabs.currentWidget() is win.studio


def test_directions_tab_animated_and_still(app, tmp_path):
    from test_anim3d import write_robot
    from pixel_pipeline.ui.tools import DirectionsTab
    tab = DirectionsTab()
    tab.sync = True
    tab.add_sources([write_robot(tmp_path / "robot.glb")])
    assert tab.animate.isChecked() and "wave" in tab.animate.text()
    tab.directions.setCurrentText("4")
    tab.frames.setValue(2)
    tab.animations.setText("wave,fall")
    tab.options.width_.setValue(24)
    tab.options.height_.setValue(24)
    tab.make()
    assert tab.anim is not None and [a.name for a in tab.anim.actions] == ["wave", "fall"]
    assert tab.view.pixmap() is not None and not tab.view.pixmap().isNull()
    names = {p.name for p in tab.save(tmp_path / "out" / "robot_dirs.png")}
    assert {"robot_dirs.png", "robot_dirs.tres", "robot_dirs_facing.gd", "robot_dirs_anim.gif"} <= names
    tab.animate.setChecked(False)
    tab.make()
    assert tab.anim is None and len(tab.sprites) == 4
    assert "robot_dirs_spin.gif" in {p.name for p in tab.save(tmp_path / "still" / "robot_dirs.png")}


def test_teams_portrait_map_and_demo_tabs(app, tmp_path):
    from PIL import Image
    from test_teams import unit
    from pixel_pipeline.ui.tools import DemoTab, MapTab, PortraitTab, TeamsTab
    Image.fromarray(unit()).save(tmp_path / "unit.png")
    teams = TeamsTab()
    teams.open_path(tmp_path / "unit.png")
    assert teams.save_btn.isEnabled()
    assert (tmp_path / "t" / "unit_enemy.png") in teams.save(tmp_path / "t")

    fig = np.zeros((240, 100, 4), np.uint8)
    fig[10:60, 30:70] = (240, 190, 150, 255)
    fig[60:240, 15:85] = (200, 40, 40, 255)
    Image.fromarray(fig).save(tmp_path / "hero.png")
    portrait = PortraitTab()
    portrait.sync = True
    portrait.width_.setValue(48)
    portrait.height_.setValue(48)
    portrait.open_path(tmp_path / "hero.png")
    assert portrait.sprite.size == (48, 48)
    assert len(portrait.save(tmp_path / "p.png")) == 2

    tiles = MapTab()
    tiles.sync = True
    tiles.tile.setValue(16)
    tiles.make()
    assert tiles.ts.names == ["grass", "water", "sand", "road"]
    assert (tmp_path / "terrain.tres") in tiles.save(tmp_path / "terrain.png")

    demo = DemoTab()
    demo.sync = True
    demo.directions.setCurrentText("4")
    demo.frames.setValue(2)
    demo.options.width_.setValue(24)
    demo.options.height_.setValue(24)
    demo.make(tmp_path / "Demo")
    assert (tmp_path / "Demo" / "project.godot").exists() and demo.open_btn.isEnabled()
    assert demo.view.rgba.shape == (360, 640, 4)


def test_crash_reports_survive_to_the_next_start(app, tmp_path, monkeypatch):
    import faulthandler
    import sys
    import threading
    from pixel_pipeline.ui import crash
    monkeypatch.setattr(crash, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    (tmp_path / "running").write_text("x")                     # the last run never closed
    (tmp_path / "faults.log").write_text("\n--- started 1 old\nok\n--- started 2 last\nFatal Python error: Aborted\n")
    try:
        crash.install()
        assert crash._previous.startswith("2 last") and "Aborted" in crash._previous
        assert sys.excepthook is crash._hook and (tmp_path / "running").exists()
    finally:
        faulthandler.disable()
    log = crash.report("Traceback (most recent call last):\nValueError: boom")
    assert "boom" in log.read_text() and "Pixel Pipeline" in log.read_text()


def test_updates_menu_prompts_and_tokens(app, tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QMessageBox
    from pixel_pipeline import online
    from pixel_pipeline.ui import update_ui, updater
    from pixel_pipeline.ui.tools import KeysDialog
    from pixel_pipeline.ui.updater import Build, Release
    monkeypatch.setattr(online, "KEYS_FILE", tmp_path / "keys.json")
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or 0)
    win = MainWindow()
    assert win.version_label.text() == "From source"            # bottom-right corner
    monkeypatch.setattr(updater, "current_build", lambda: Build(7, "abc1234", "2026-10-10"))
    built = MainWindow()
    assert built.version_label.text() == "Version 7" and "version 7" in built.windowTitle()
    help_menu = [a.menu() for a in win.menuBar().actions() if a.text() == "&Help"][0]
    assert "Check for &updates…" in [a.text() for a in help_menu.actions()]
    assert not update_ui.offer_install()                            # running from source
    win.updates.check(loud=True)
    assert "from source" in shown[-1]
    # A quiet check that needs a token only says so in the status bar.
    win.updates.loud = False
    win.updates._on_checked(("the repo is private", True))
    assert "private" in win.statusBar().currentMessage()
    # A newer build asks first ("Later" here: the stub clicks nothing).
    monkeypatch.setattr(updater, "current_build", lambda: Build(3, "aaa", "2026-10-01"))
    fetched = []
    monkeypatch.setattr(win.updates, "fetch", fetched.append)
    win.updates._on_checked(Release(Build(4, "bbb", "2026-10-10"), "http://x/zip", 1))
    assert shown[-1] == "A new Pixel Pipeline is ready." and fetched == []
    win.updates.loud = True
    win.updates._on_checked(Release(Build(3), "http://x/zip", 1))
    assert "newest" in shown[-1]
    # The token and the AI keys share a file; saving one keeps the other.
    online.save_keys({"pixellab": "pl-key"})
    dlg = update_ui.TokenDialog(win)
    dlg.edit.setText(" github_pat_x ")
    dlg.save()
    keys = KeysDialog(win)
    keys.edits["retrodiffusion"].setText("rd-key")
    keys.save()
    assert json.loads((tmp_path / "keys.json").read_text()) == {
        "pixellab": "pl-key", "retrodiffusion": "rd-key", "github": "github_pat_x"}
    assert updater.github_token() == "github_pat_x"


def test_drops_from_a_browser_or_a_folder(app, tmp_path, monkeypatch):
    import time
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from PIL import Image
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QMessageBox
    from pixel_pipeline.ui import app as app_module
    from test_drops import Site, png_bytes
    monkeypatch.setattr(app_module, "drop_folder", lambda: tmp_path / "dropped")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    Site.pages = {"/knight.png": png_bytes((20, 30))}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    Thread(target=server.serve_forever, daemon=True).start()
    win = MainWindow()

    def drop(mime):
        win.source = None
        win.handle_drop(mime)
        for _ in range(200):
            app.processEvents()
            if win.source is not None or warnings:
                return
            time.sleep(0.02)

    try:
        win.tabs.setCurrentWidget(win.map_tab)            # any tab: pictures land on Sprite
        web = QMimeData()
        web.setUrls([QUrl(f"http://127.0.0.1:{server.server_port}/knight.png")])
        web.setHtml(f'<img src="http://127.0.0.1:{server.server_port}/knight.png">')
        drop(web)
        assert win.source.shape == (30, 20, 4) and win.tabs.currentIndex() == 0
        assert win.source_path == tmp_path / "dropped" / "knight.png"
        raw = QMimeData()                                 # a browser that hands over the pixels
        raw.setImageData(QImage(8, 6, QImage.Format_RGBA8888))
        drop(raw)
        assert win.source.shape == (6, 8, 4)
        local = tmp_path / "photo.png"
        Image.new("RGB", (9, 7), (10, 200, 10)).save(local)
        files = QMimeData()
        files.setUrls([QUrl.fromLocalFile(str(local))])
        win.tabs.setCurrentWidget(win.directions_tab)     # a tool that takes files keeps them
        win.handle_drop(files)
        assert win.directions_tab.sources == [local]
        gone = QMimeData()
        gone.setUrls([QUrl(f"http://127.0.0.1:{server.server_port}/missing.png")])
        drop(gone)
        assert warnings and "wouldn't give the picture" in warnings[-1]
    finally:
        server.shutdown()
