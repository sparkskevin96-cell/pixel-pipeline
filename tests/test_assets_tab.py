import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
except ImportError as e:  # missing system GL libs on headless machines
    pytest.skip(f"Qt unavailable: {e}", allow_module_level=True)

from pixel_pipeline import assets as A  # noqa: E402
from pixel_pipeline import jobqueue  # noqa: E402
from pixel_pipeline.ui.assets_tab import AssetsTab, ReskinDialog, StyleDialog  # noqa: E402
from test_exports import make_character  # noqa: E402


class Counter:
    def __init__(self):
        self.n = 0

    def hit(self, *_):
        self.n += 1


@pytest.fixture(scope="module")
def app():
    from conftest import qt_app
    return qt_app()


def test_browse_filter_tag_export_reskin_delete(app, tmp_path):
    knight = make_character(tmp_path)
    sword = A.new_asset("item", "Iron Sword", prompt="a plain iron sword")
    Image.fromarray(np.full((16, 16, 4), 200, np.uint8), "RGBA").save(sword.folder / "icon.png")
    tab = AssetsTab()
    tab.sync = True
    changed = Counter()
    tab.library_changed.connect(changed.hit)             # a method, not a lambda (teardown)
    assert {a.name for a in tab.assets} == {"Knight", "Iron Sword"} and tab.grid.count() == 2
    tab.set_kind("item")
    assert [a.name for a in tab.assets] == ["Iron Sword"]
    assert not tab.reskin_btn.isEnabled() or tab.selected is None
    tab.set_kind(None)
    tab.search.setText("plain")
    assert [a.name for a in tab.assets] == ["Iron Sword"]
    tab.search.setText("")
    tab.select("Knight")
    assert tab.reskin_btn.isEnabled() and tab.name_label.text() == "Knight"
    tab.tags.setText("hero, metal")
    tab.collections.setText("starter")
    tab.save_tags()
    assert tab.tag_box.findData("hero") >= 0 and tab.collection_box.findData("starter") >= 0
    tab.collection_box.setCurrentIndex(tab.collection_box.findData("starter"))
    assert [a.name for a in tab.assets] == ["Knight"] and tab.selected.name == "Knight"
    tab.collection_box.setCurrentIndex(0)
    files = tab.export_selected("unity", tmp_path / "out")
    assert any(p.name == "knight.png.meta" for p in files)
    tab.export_collection_to("starter", "generic", tmp_path / "coll")
    assert (tmp_path / "coll" / "index.json").exists()
    tab.select("Knight")
    dlg = ReskinDialog(knight)
    assert dlg.view.rgba is not None and dlg.options()["target_colour"] == "blue"
    dlg.setParent(None)
    blue = tab.reskin_selected("Blue Knight", target_colour="blue")
    assert tab.selected.name == "Blue Knight" and "re-skin of Knight" in tab.kind_label.text()
    assert blue.folder.exists() and changed.n
    tab.delete_selected()
    assert not blue.folder.exists() and {a.name for a in tab.assets} == {"Knight", "Iron Sword"}


def test_projects_and_queue(app, tmp_path):
    tab = AssetsTab()
    tab.sync = True
    dlg = StyleDialog(None, A.Style(person_height=32, outline="none"), "Retro")
    st = dlg.style()
    assert st.person_height == 32 and st.outline == "none" and st.directions == 16
    projects = []
    tab.project_changed.connect(projects.append)
    tab.new_project("Retro", st)
    assert tab.project.name == "Retro" and A.current_project().style.person_height == 32
    assert projects == ["Retro"] and tab.project_box.currentText() == "Retro"
    made = []

    def fake(name, prompt, options, project):
        made.append(name)
        return A.new_asset("prop", name, project=project, prompt=prompt)
    tab.worker_fns = {"prop": fake}
    tab.pause.setValue(0)
    tab.add_job("prop", "Barrel", "an oak barrel", variations=2)
    assert tab.job_list.count() == 1 and tab.start_btn.isEnabled()
    tab.start_queue()
    assert made == ["Barrel v1", "Barrel v2"]
    assert tab.queue.jobs[0].status == "done" and "done" in tab.job_list.item(0).text()
    assert {a.name for a in tab.assets} >= {"Barrel v1", "Barrel v2"}
    assert (tab.project.folder / "queue.json").exists()
    assert jobqueue.Queue(tab.project).jobs[0].done_count == 2
    tab.switch_project("Default")
    assert tab.job_list.count() == 0
