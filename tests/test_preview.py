"""Drawn characters front first (library.add_character(preview=True)): only the reference
direction + a short idle are made, the character waits as a draft for the user's OK; approving
draws the rest paired to that drawing (never drawn again), redrawing changes only it, older tries
can be picked, and it all survives a restart. Batches collect the drafts; the review panel works
in the Characters page and the Make page. Fake ComfyUI, the test_anim3d robot."""
import json
import os

import numpy as np
import pytest

from pixel_pipeline import drawnrig as dr
from pixel_pipeline.assets import Style
from test_anim3d import write_robot
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)


@pytest.fixture
def robot_lib(comfy, tmp_path, monkeypatch):  # noqa: F811
    from pixel_pipeline import autorig, generate, library
    fake, client = comfy
    monkeypatch.setattr(generate, "ComfyClient", lambda *a, **k: client)
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    fake.history_delay = 0
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    monkeypatch.setattr(library, "add_project_clips", lambda *a, **k: [])
    monkeypatch.setattr(autorig, "auto_rig", lambda model, out: write_robot(out))
    monkeypatch.setenv("PIXEL_DRAWN", "1")
    return fake, write_robot(tmp_path / "robot_model.glb")


def _inpaint(wf) -> bool:
    return any(n["class_type"] == "LoadImageMask" for n in wf.values())


def _seed(wf) -> int:
    return next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]["seed"]


def _texts(wfs):
    return [n["inputs"]["text"] for wf in wfs for n in wf.values()
            if n["class_type"] == "CLIPTextEncode" and n.get("_meta", {}).get("title") == "Prompt"]


def _front_files(ch):
    return {f.name: f.read_bytes() for f in (ch.folder / "keys").glob("drawn_S_*.png")}


FULL = 3 * len(dr.CANDIDATES) + len(dr.BACK_CANDIDATES)


def test_preview_stops_after_the_front_and_approving_keeps_it(robot_lib):
    from pixel_pipeline import library
    fake, model = robot_lib
    said = []
    ch = library.add_character("Bot", model, prompt="a robot", preview=True, progress=said.append)
    front = len(fake.prompts)
    assert 1 <= front <= len(dr.CANDIDATES)                    # S only (its candidates)
    assert not any(_inpaint(wf) for wf in fake.prompts)        # no paired views yet
    assert library.in_preview(ch) and ch.waiting and not library.is_drawn(ch)
    assert not ch.sheet.exists() and ch.thumb.exists()         # a draft: no sprites yet
    assert any("waiting for your OK" in s for s in said)
    info = library.preview_info(ch)
    assert info["direction"] == "S" and info["key"].exists() and len(info["frames"]) == 8
    assert info["clip"] == "wave"                              # the robot has no idle: its 1st clip
    assert "other 3 directions" in info["estimate"]
    meta = json.loads((ch.folder / "keys" / "keys.json").read_text())
    assert meta["partial"] and meta["preview"]["current"] == 1 and meta["camera"]["directions"] == 4
    before = _front_files(ch)
    assert before and all((ch.folder / "keys" / "attempts" / "1" / n).read_bytes() == b
                          for n, b in before.items())
    # an app restart: the draft is found again as it was
    again = library.get("Bot")
    assert again.waiting and library.preview_info(again)["attempts"][0]["seed"] == 1234
    assert library.is_draft_folder(again.folder)
    assert again not in library.ready_characters()
    with pytest.raises(ValueError, match="waiting for your OK"):
        library.add_character_animations(again, ["jump"])      # no 3D-rendered half-sheets
    # Looks good: the rest is drawn beside this S, which is NOT drawn again
    ch = library.approve_preview(again)
    new = fake.prompts[front:]
    assert len(new) == FULL - len(dr.CANDIDATES)               # every other direction once
    assert all(_inpaint(wf) for wf in new)                     # all paired to the S drawing
    assert _front_files(ch) == before                          # byte-identical
    assert library.is_drawn(ch) and not ch.waiting and ch.sheet.exists()
    assert ch in library.ready_characters()
    with pytest.raises(ValueError, match="isn't waiting"):
        library.approve_preview(ch)


def test_redraw_changes_only_the_front_and_older_tries_can_be_picked(robot_lib, monkeypatch):
    from pixel_pipeline import library
    fake, model = robot_lib
    ch = library.add_character("Bot", model, prompt="a robot", preview=True)
    first = _front_files(ch)
    n0 = len(fake.prompts)
    library.redraw_preview(ch)                                 # a new seed, S only
    redrawn = fake.prompts[n0:]
    assert redrawn and not any(_inpaint(wf) for wf in redrawn)
    assert {_seed(wf) for wf in redrawn}.isdisjoint({_seed(wf) for wf in fake.prompts[:n0]})
    second = _front_files(ch)
    assert second != first
    info = library.preview_info(ch)
    assert [a["n"] for a in info["attempts"]] == [1, 2] and info["current"] == 2
    assert (ch.folder / "keys" / "attempts" / "1" / "key.png").exists()
    # the first try again: its files back, no GPU
    n1 = len(fake.prompts)
    library.pick_preview(ch, 1)
    assert len(fake.prompts) == n1 and _front_files(ch) == first
    meta = json.loads((ch.folder / "keys" / "keys.json").read_text())
    assert meta["seeds"]["S"] == 1234 and meta["preview"]["current"] == 1
    # new words: S drawn again from them (same model), the words saved on the character
    library.redraw_preview(ch, words="a shiny chrome robot")
    assert any("shiny chrome robot" in t for t in _texts(fake.prompts[n1:]))
    ch = library.get("Bot")
    assert ch.prompt == "a shiny chrome robot" and len(library.preview_info(ch)["attempts"]) == 3
    # back to try 1 (its words too), then approve: try 1's drawing is the one used
    library.pick_preview(ch, 1)
    assert library.get("Bot").prompt == "a robot"
    n2 = len(fake.prompts)
    ch = library.approve_preview(library.get("Bot"))
    assert _front_files(ch) == first and len(fake.prompts) - n2 == FULL - len(dr.CANDIDATES)
    with pytest.raises(ValueError):
        library.pick_preview(ch, 1)                            # not a draft any more


def test_new_model_too_starts_the_tries_again(robot_lib, monkeypatch):
    from pixel_pipeline import library, model3d
    fake, model = robot_lib
    ch = library.add_character("Bot", model, prompt="a robot", preview=True)
    library.redraw_preview(ch)
    made = []

    def fake_model(prompt, out, subject="person", **kw):
        write_robot(out)
        made.append((prompt, subject))
        return model3d.ModelResult(out, None, None, 1)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    n = len(fake.prompts)
    library.redraw_preview(ch, words="a tall robot", new_model=True)
    assert made == [("a tall robot", "person")]
    assert len(fake.prompts) > n                               # drawn again on the new model
    info = library.preview_info(library.get("Bot"))
    assert [a["n"] for a in info["attempts"]] == [1]           # the old model's tries are gone
    assert info["words"] == "a tall robot"


def test_drafts_are_skipped_by_exports_and_the_game(robot_lib, tmp_path):
    from pixel_pipeline import exports, library
    fake, model = robot_lib
    ch = library.add_character("Bot", model, prompt="a robot", preview=True)
    with pytest.raises(ValueError, match="draft"):
        exports.export_asset(ch, "generic", tmp_path / "out")
    with pytest.raises(ValueError, match="draft"):
        exports.export_collection([ch], "generic", tmp_path / "out2")
    plain = library.add_character("Plain", model, prompt="a robot", drawn=False)
    exports.export_collection([ch, plain], "generic", tmp_path / "out3")
    index = json.loads((tmp_path / "out3" / "index.json").read_text())
    assert [a["name"] for a in index["assets"]] == ["Plain"]
    assert index["skipped"][0]["name"] == "Bot"
    from pixel_pipeline.assets import _restyle_how
    from types import SimpleNamespace
    assert _restyle_how(SimpleNamespace(kind="character", folder=ch.folder, extra={})) == ""


def test_jobqueue_characters_stop_for_the_ok(robot_lib, tmp_path, monkeypatch):
    from pixel_pipeline import jobqueue as J
    from pixel_pipeline import library, makeany as M, model3d
    fake, model = robot_lib

    def fake_model(prompt, out, subject="person", **kw):
        write_robot(out)
        return model3d.ModelResult(out, None, None, 1)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    item_made = []

    def item(plan, project, say, client=None):
        item_made.append(plan.name)
        return M.Made(plan, None)
    plans = [M.route("a robot knight"), M.route("rusty iron sword")]
    assert plans[0].params["review"] and "once you OK the front" in plans[0].summary()
    assert plans[0].gpu_minutes() < M.Plan("character", "x", "x",
                                           {**plans[0].params, "review": False}).gpu_minutes()
    proj = tmp_path / "proj"
    proj.mkdir()
    q = J.Queue(proj)
    jobs = M.queue_plans(proj, plans, queue=q)

    def worker(name, prompt, options, project):
        return M.queue_worker(name, prompt, options, J._as_project(project),
                              makers={"item": item})
    J.run_queue(proj, {"make": worker}, pause_s=0, queue=q)
    assert item_made == ["Rusty iron sword"]                   # items as before
    assert jobs[0].results[0]["waiting"] and "waiting" not in jobs[1].results[0]
    text = M.batch_summary(q.jobs)
    assert text.startswith("2 made (1 waiting for your OK)") and "waiting for your OK" in \
        text.splitlines()[1]
    ch = library.from_folder(jobs[0].results[0]["folder"])
    assert ch.waiting and len(fake.prompts) <= len(dr.CANDIDATES)
    # the plain character job too (jobqueue kind "character"): review on by default, off = all
    q.add("character", "Bot2", "a robot", {"review": False})
    J.run_queue(proj, {"character": J._character_worker}, pause_s=0, queue=q)
    done = library.from_folder(q.jobs[-1].results[0]["folder"])
    assert library.is_drawn(done) and not done.waiting


# --------------------------------------------------------------------------- the review panel

@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from conftest import qt_app
    return qt_app()


def test_characters_page_review_flow(robot_lib, app, tmp_path):
    from pixel_pipeline import library
    from pixel_pipeline.model3d import ModelResult
    from pixel_pipeline.ui.model_studio import ModelStudio
    fake, model = robot_lib
    win = ModelStudio()
    win.sync = True
    assert win.drawn.isChecked() and win.front_first.isChecked()
    win.result = ModelResult(model, None, None, None)
    win.name.setText("Bot")
    win.prompt.setText("a robot")
    win.add_to_library()
    ch = library.get("Bot")
    assert ch.waiting and win.stack.currentWidget() is win.review
    assert win.review.info["direction"] == "S" and win.review.key_view.rgba is not None
    assert len(win.review.frames) == 8 and win.review._timer.isActive()
    assert "3 directions" in win.review.estimate.text()
    assert "4 directions" in win.review.approve_btn.text()
    item = win.library.item(0)
    assert "waiting for OK" in item.text() and "waiting for your OK" in win.lib_title.text()
    first = _front_files(ch)
    win.review.redraw()
    assert win.review.tries.count() == 2 and win.review.tries_box.isVisibleTo(win.review)
    assert _front_files(ch) != first
    win.review.pick(1)
    assert _front_files(ch) == first and win.review.info["current"] == 1
    # Not now: a draft; clicking it in the library shows the review again
    win.review.keep_draft()
    assert win.stack.currentWidget() is win.panels_page
    win.open_character(library.get("Bot"))
    assert win.stack.currentWidget() is win.review
    n = len(fake.prompts)
    win.review.approve()
    assert library.is_drawn(library.get("Bot")) and not win.review.last_error
    assert len(fake.prompts) - n == FULL - len(dr.CANDIDATES)
    assert _front_files(ch) == first
    assert win.stack.currentWidget() is win.panels_page and win.anim is not None
    assert "waiting" not in win.library.item(0).text()
    win.close()


def test_characters_page_throw_away_and_stop(robot_lib, app, tmp_path, monkeypatch):
    from pixel_pipeline import library
    from pixel_pipeline.model3d import ModelResult
    from pixel_pipeline.ui.model_studio import ModelStudio
    fake, model = robot_lib
    win = ModelStudio()
    win.sync = True
    win.result = ModelResult(model, None, None, None)
    win.name.setText("Bot")
    win.add_to_library()
    ch = library.get("Bot")
    # Stop while the rest is drawn: still a draft, the drawings kept, the panel works again
    n = len(fake.prompts)
    real = dr.draw_keys_paired

    def stop_soon(*a, **k):
        cancel = k["cancel"]
        k["cancel"] = lambda: cancel() or len(fake.prompts) >= n + 2
        return real(*a, **k)
    monkeypatch.setattr(dr, "draw_keys_paired", stop_soon)
    win.review.approve()
    assert "Stopped" in win.review.last_error and library.get("Bot").waiting
    assert win.review.approve_btn.isEnabled()
    monkeypatch.setattr(dr, "draw_keys_paired", real)
    win.review.discard(ask=False)
    assert library.get("Bot") is None and win.stack.currentWidget() is win.panels_page
    assert not (ch.folder / "character.json").exists()
    win.close()


def test_make_page_review_single_and_batch(robot_lib, app, tmp_path, monkeypatch):
    from pixel_pipeline import library, makeany as M, model3d
    from pixel_pipeline.assets import create_project
    from pixel_pipeline.ui.make_tab import MakeTab
    fake, model = robot_lib

    def fake_model(prompt, out, subject="person", **kw):
        write_robot(out)
        return model3d.ModelResult(out, None, None, 1)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    proj = create_project("Preview test")
    opened = []
    tab = MakeTab(open_result=opened.append)
    tab.sync = True
    monkeypatch.setattr(tab, "project", lambda: proj)
    tab.plan_text("a robot knight")
    assert tab.current_plan().params["review"] and "OK the front" in tab.summary.text()
    tab.make()
    assert tab.view_stack.currentWidget() is tab.review and not opened
    ch = tab.review.character
    assert ch.waiting and tab.made.waiting
    tab.review.approve()
    assert opened and opened[-1].asset.folder == ch.folder and library.is_drawn(opened[-1].asset)
    assert tab.view_stack.currentWidget() is tab.view
    # many at once: the characters wait in a list, the item is made as usual
    items = []

    def item(plan, project, say, client=None):
        items.append(plan.name)
        return M.Made(plan, None)
    tab.makers = {"item": item}
    tab.rest.setValue(0)
    assert tab.batch_review.isChecked()
    tab.batch.setPlainText("a robot knight\nrusty iron sword\na robot guard")
    tab.queue_batch()
    assert items == ["Rusty iron sword"]
    assert "2 waiting for your OK" in tab.report.toPlainText().splitlines()[0]
    assert tab.waiting.count() == 2 and tab.waiting_box.isVisibleTo(tab)
    first = tab.waiting.item(0).data(0x0100)
    tab.review_character(first)
    assert tab.view_stack.currentWidget() is tab.review
    tab.review.approve()
    assert tab.waiting.count() == 1 and library.is_drawn(library.from_folder(first.folder))
    # off: as before, every character all the way
    tab.batch_review.setChecked(False)
    tab.batch.setPlainText("a robot guard")
    tab.queue_batch()
    assert tab.waiting.count() == 1                            # only the earlier one waits
    last = tab.batch_jobs[-1].results[0]["folder"]
    assert library.is_drawn(library.from_folder(last))
