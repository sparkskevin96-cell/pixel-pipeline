"""Progress bars: progress.Tracker (stages, n of N, GPU steps, ETAs, Stop), ComfyUI's websocket
steps (comfyws, fake server), and the UI panels (ToolTab, Characters page, global bar)."""
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from pixel_pipeline import comfyws, progress as P
from pixel_pipeline.generate import ComfyClient, generate_images


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def add(self, s):
        self.t += s


def fractions(tr):
    return [u.fraction for u in tr.history if u.fraction is not None]


def monotonic(xs):
    return all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))


# --------------------------------------------------------------------------- text reading

def test_read_steps_and_time_left():
    assert P.read_steps("Drawing it facing SE (beside S, try o0.6; 3 of 16, about 4 min left)…") == (2, 16)
    assert P.read_steps("Bending the drawings: 3 of 16 directions done (about 20 s left)…") == (3, 16)
    assert P.read_steps("Card art 2/3: the scene…") == (1, 3)
    assert P.read_steps("Making texture 1 of 3 seamless…") == (0, 3)
    for text in ("ComfyUI failed (failure 1 of 3 allowed)…", "Reusing 12 of 64 clip directions",
                 "2 of the pictures were not one subject", "Painting the model"):
        assert P.read_steps(text) is None, text
    assert P.read_left("…, about 4 min left)…") == 240
    assert P.read_left("(about 20 s left)") == 20
    assert P.read_left("nothing") is None
    assert P.eta_text(None) == "" and P.eta_text(30) == "about 30 s left"
    assert P.eta_text(240) == "about 4 min left" and P.eta_text(0) == "almost done"


def test_text_only_and_old_callback_forms():
    clock = Clock()
    tr = P.Tracker(clock=clock, min_interval=0)
    tr("Working on it…")                               # text only: busy bar, no fraction yet
    assert tr.snapshot().fraction is None and tr.snapshot().text == "Working on it…"
    tr("picture", "Making a picture…")                 # (step, text), model3d's form
    tr(1, 4)                                           # (done, total), updater's form
    assert tr.snapshot().fraction == pytest.approx(0.25, abs=0.01)
    tr("Making 3 of 4…")
    assert tr.snapshot().fraction == pytest.approx(0.5, abs=0.25)
    tr(0.9)                                            # a fraction of the stage
    assert tr.snapshot().fraction >= 0.9
    said = []
    old = P.wrap(said.append)                          # an old text callback takes every form
    old("shape", "Building the 3D shape…")
    old("plain")
    old(3, 4)
    assert said == ["Building the 3D shape…", "plain"]
    assert P.wrap(tr) is tr
    P.wrap(None)("anything")


# --------------------------------------------------------------------------- whole jobs

def _drawn_character(tr, clock, sampling=True):
    """The messages of a real drawn character (picture -> shape -> rig -> 25 drawings ->
    bending), with ComfyUI steps inside each picture. Returns [(true seconds left, eta)]."""
    script = [("picture", "Making a front picture of “a knight” on your GPU…", 15),
              ("cutout", "Cutting out the subject…", 3),
              ("shape", "Building the 3D shape on your GPU (Hunyuan3D, about a minute)…", 100),
              ("views", "Drawing the back of it on your GPU…", 30),
              ("colour", "Painting the model from the pictures…", 6)]
    total = sum(s[2] for s in script) + 15 + 25 * 26 + 16 * 7
    left, seen = total, []
    for key, text, dt in script:
        tr(key, text)
        for i in range(5):
            if sampling:
                tr.sampling(i * 5 + 5, 25)
            clock.add(dt / 5)
            left -= dt / 5
        seen.append((left, tr.snapshot().eta))
    tr("Adding a skeleton and animations…")
    clock.add(15)
    left -= 15
    for n in range(1, 26):
        eta = f", about {round((26 - n) * 26 / 60)} min left" if n > 1 else ""
        tr(f"Drawing it facing S (beside S, try o0.6; {n} of 25{eta})…")
        for i in range(5):
            if sampling:
                tr.sampling(i * 5 + 5, 25)
            clock.add(26 / 5)
            left -= 26 / 5
        seen.append((left, tr.snapshot().eta))
    for n in range(1, 17):
        clock.add(7)
        left -= 7
        tr(f"Bending the drawings: {n} of 16 directions done (about {7 * (16 - n)} s left)…")
        seen.append((left, tr.snapshot().eta))
    tr.finish("Done")
    return seen


@pytest.mark.parametrize("sampling", [True, False])
def test_drawn_character_bar_is_monotonic_and_etas_sane(sampling):
    clock = Clock()
    tr = P.Tracker(P.character_plan(directions=16), clock=clock, min_interval=0)
    seen = _drawn_character(tr, clock, sampling)
    fr = fractions(tr)
    assert monotonic(fr) and fr[-1] == 1.0
    assert len(set(round(f, 3) for f in fr)) > 40              # it moves, not jumps
    stage_fr = [u.stage_fraction for u in tr.history if u.stage_fraction is not None]
    assert all(0 <= f <= 1 for f in stage_fr)
    u = tr.snapshot()
    assert u.done and u.eta == 0.0
    # ETAs: within 2x of the truth once a couple of minutes are done (the plan's minutes are
    # the measured GPU table; the live rate takes over), and they count down
    for true_left, eta in seen[8:-3]:
        assert eta is not None and true_left / 2.2 <= eta <= true_left * 2.2 + 30, (true_left, eta)
    etas = [e for _, e in seen[6:]]
    assert etas[-1] < etas[0] / 5
    stages = [h.stage for h in tr.history]
    for name in ("Front picture", "3D shape", "Skeleton + animations", "Drawing 16 directions",
                 "Bending the drawings"):
        assert name in stages


def test_gpu_steps_move_the_bar_inside_one_picture():
    clock = Clock()
    tr = P.Tracker([P.Stage("picture", "Picture", 0.25, pictures=1),
                    P.Stage("cut", "Cut out", 0.05, r"cutting")], clock=clock, min_interval=0)
    with P.using(tr):
        P.gpu_pictures(0, 1)
        for v in range(1, 26):
            P.gpu_sampling(v, 25)
            clock.add(0.4)
    u = tr.snapshot()
    assert u.sampling == (25, 25) and u.stage_fraction == pytest.approx(1.0)
    assert u.fraction == pytest.approx(0.25 / 0.30, abs=0.01)
    f = fractions(tr)
    assert monotonic(f) and f[0] < 0.1


def test_variations_in_one_call_count_pictures():
    tr = P.Tracker([P.Stage("make", "3 icons", 0.6, pictures=1)], min_interval=0)
    with P.using(tr):
        for i in range(3):
            P.gpu_pictures(i, 3)
            P.gpu_sampling(12, 25)
            assert tr.snapshot().stage_fraction == pytest.approx((i + 0.48) / 3, abs=0.02)


def test_skipped_stages_drop_out_and_never_go_back():
    clock = Clock()
    tr = P.Tracker(P.character_plan(drawn=False), clock=clock, min_interval=0)
    clock.add(5)
    tr("Rendering every animation from 16 directions…")       # straight to the last stage
    u = tr.snapshot()
    assert u.stage.startswith("Rendering") and u.stage_count == 1
    tr("Making a front picture…")                             # an earlier stage's words: ignored
    assert tr.snapshot().stage.startswith("Rendering")
    assert monotonic(fractions(tr))


def test_batch_children_job_k_of_n():
    clock = Clock()
    batch = P.Tracker([P.Stage("batch", "3 things", 3.0)], clock=clock, min_interval=0,
                      stop_raises=False)
    for k in range(3):
        child = batch.child(k, 3, [P.Stage("a", "A", 0.5), P.Stage("b", "B", 0.5, r"second")],
                            title=f"job {k}")
        with P.using(child):
            child("first…")
            clock.add(20)
            child("second half…")
            clock.add(20)
            P.gpu_sampling(10, 20)
        child.finish()
        f = batch.snapshot().fraction
        assert f == pytest.approx((k + 1) / 3, abs=0.02)
        assert P.child_update(batch).title == f"job {k}"
    assert monotonic(fractions(batch))


def test_stop_raises_at_the_next_message():
    tr = P.Tracker()
    tr("one")
    threading.Thread(target=tr.stop).start()
    time.sleep(0.05)
    assert tr.stopped
    with pytest.raises(P.Stopped):
        tr("two")
    soft = P.Tracker(stop_raises=False)               # a batch: "after this one"
    soft.stop()
    soft("still fine")
    with P.using(soft):
        assert not P.stop_requested()


# --------------------------------------------------------------------------- ComfyUI websocket

def _ws_handler():
    from test_generate import Handler

    class WSHandler(Handler):
        def do_GET(self):
            if not self.path.startswith("/ws"):
                return super().do_GET()
            fake = self.server.fake
            fake.ws_paths.append(self.path)
            if fake.ws_refuse:
                return self.send(404, b"404: Not Found", "text/plain")
            key = self.headers.get("Sec-WebSocket-Key", "")
            self.wfile.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                             b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + key.encode()
                             + b"\r\n\r\n")
            self.wfile.flush()

            def frame(msg):
                data = json.dumps(msg).encode()
                head = bytes([0x81, len(data)]) if len(data) < 126 else \
                    bytes([0x81, 126]) + len(data).to_bytes(2, "big")
                self.wfile.write(head + data)
                self.wfile.flush()
            frame({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}})
            sent, idle = set(), 0
            while idle < 300:
                todo = [f"p-{i}" for i in range(len(fake.prompts)) if f"p-{i}" not in sent]
                if not todo:
                    idle += 1
                    time.sleep(0.01)
                    continue
                idle = 0
                for pid in todo:
                    sent.add(pid)
                    for v in range(1, fake.steps + 1):
                        frame({"type": "progress", "data": {"value": v, "max": fake.steps,
                                                            "prompt_id": pid, "node": "3"}})
                        time.sleep(fake.step_s)
                    frame({"type": "executing", "data": {"node": None, "prompt_id": pid}})
                    fake.sampled.add(pid)
    return WSHandler


@pytest.fixture
def wscomfy():
    from test_generate import FakeComfy
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ws_handler())
    fake = FakeComfy()
    fake.ws_paths, fake.ws_refuse, fake.steps, fake.step_s = [], False, 20, 0.02
    fake.sampled = set()
    fake.history_delay = 10**9                     # "done" only once the steps were sent
    server.fake = fake
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                              daemon=True)
    thread.start()
    client = ComfyClient(f"http://127.0.0.1:{server.server_address[1]}", timeout=5,
                         poll_interval=0.01)

    def gate():                                     # history: done after the ws steps
        while True:
            for i in range(len(fake.prompts)):
                pid = f"p-{i}"
                if pid in fake.sampled and pid not in fake.history:
                    fake.history[pid] = fake.finish(pid)
            time.sleep(0.005)
    threading.Thread(target=gate, daemon=True).start()
    yield fake, client
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _white_key(monkeypatch):
    import numpy as np
    from pixel_pipeline import pipeline as pl
    monkeypatch.setattr(pl, "remove_background", lambda rgb, model="isnet-general-use":
                        np.where((rgb.min(-1) > 240), 0, 255).astype(np.uint8), raising=False)


def test_comfyui_websocket_steps_reach_the_bar(wscomfy):
    fake, client = wscomfy
    tr = P.Tracker([P.Stage("make", "2 pictures", 0.5, pictures=1)], min_interval=0)
    with P.using(tr):
        out = generate_images("a red apple", client=client, seed=1, count=2)
    assert len(out) == 2
    assert fake.ws_paths and fake.ws_paths[0] == f"/ws?clientId={client.client_id}"
    steps = [u.sampling for u in tr.history if u.sampling]
    assert steps and steps[-1] == (20, 20)
    assert len({s[0] for s in steps}) >= 5          # it watched the steps go by
    f = [u.stage_fraction for u in tr.history if u.stage_fraction is not None]
    assert monotonic(f) and max(f) > 0.9
    mid = [u.stage_fraction for u in tr.history if u.sampling and 6 <= u.sampling[0] <= 14]
    assert mid and 0.1 < min(mid) < 0.9             # half-way through a picture = half-way


def test_no_websocket_falls_back_quietly(wscomfy):
    fake, client = wscomfy
    fake.ws_refuse = True
    fake.history_delay = 3
    fake.sampled = {f"p-{i}" for i in range(10)}     # history answers without the steps
    tr = P.Tracker(estimate=10, min_interval=0)
    t0 = time.monotonic()
    with P.using(tr):
        generate_images("a red apple", client=client, seed=2, count=1)
    assert time.monotonic() - t0 < 5
    assert not any(u.sampling for u in tr.history)
    assert comfyws.current(client.base_url, client.client_id) is None


def test_untracked_calls_open_no_websocket(wscomfy):
    fake, client = wscomfy
    fake.sampled = {f"p-{i}" for i in range(10)}
    generate_images("a red apple", client=client, seed=3, count=1)
    assert fake.ws_paths == []


def test_stop_interrupts_the_gpu_picture(wscomfy, monkeypatch):
    fake, client = wscomfy
    fake.steps, fake.step_s = 200, 0.02             # a long picture
    calls = []
    real = client._request

    def spy(method, path, *a, **k):
        if path in ("/interrupt", "/queue"):
            calls.append(path)
        return real(method, path, *a, **k)
    monkeypatch.setattr(client, "_request", spy)
    tr = P.Tracker(min_interval=0)
    threading.Timer(0.4, tr.stop).start()
    t0 = time.monotonic()
    with P.using(tr), pytest.raises(P.Stopped):
        generate_images("a red apple", client=client, seed=4, count=1)
    assert time.monotonic() - t0 < 3
    assert "/interrupt" in calls


def test_websocket_frames_parse():
    ws = comfyws.ProgressSocket("http://127.0.0.1:1", "x")
    ws._message({"type": "progress", "data": {"value": 3, "max": 25, "prompt_id": "a"}})
    assert ws.progress("a") == (3, 25)
    ws._message({"type": "progress_state", "data": {"prompt_id": "b", "nodes": {
        "1": {"state": "finished", "value": 1, "max": 1},
        "3": {"state": "running", "value": 7, "max": 30, "node_id": "3"}}}})
    assert ws.progress("b") == (7, 30)
    ws._message({"type": "executing", "data": {"node": None, "prompt_id": "a"}})
    assert ws.finished_prompt("a") and not ws.finished_prompt("b")
    ws._message("junk")
    ws._message({"type": "progress", "data": {"value": "x"}})


# --------------------------------------------------------------------------- UI

@pytest.fixture
def qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from conftest import qt_app
    return qt_app()


def test_tooltab_panel_moves_and_hides(qt):
    from PySide6.QtWidgets import QVBoxLayout
    from pixel_pipeline.ui.common import ToolTab, hub
    said = []
    tab = ToolTab(said.append)
    QVBoxLayout(tab)
    tab.sync = True
    seen = []

    def job(say):
        for n in range(1, 5):
            say(f"Drawing {n} of 4…")
            P.gpu_sampling(10, 20)
            seen.append((tab.progress_panel.isVisibleTo(tab), tab.progress_panel.overall.value(),
                         hub().current() is not None))
        return "ok"
    got = []
    tab.run(job, got.append, "Testing…", estimate=5)
    assert got == ["ok"]
    vals = [v for _, v, _ in seen]
    assert all(vis and live for vis, _, live in seen) and monotonic(vals) and vals[-1] > vals[0]
    assert not tab.progress_panel.isVisibleTo(tab) and hub().current() is None
    assert "Drawing 4 of 4…" in said                 # the status line still gets every text


def test_tooltab_text_only_job_still_works(qt):
    from pixel_pipeline.ui.common import ToolTab
    said = []
    tab = ToolTab(said.append)
    tab.sync = True
    got = []
    tab.run(lambda say: (say("one"), say("two"), 5)[-1], got.append)
    assert got == [5] and said[-2:] == ["one", "two"]


def test_tooltab_stop(qt):
    from PySide6.QtWidgets import QVBoxLayout
    from pixel_pipeline.ui.common import ToolTab
    said, hooks = [], []
    tab = ToolTab(said.append)
    QVBoxLayout(tab)
    tab.sync = True
    tab.on_stop = lambda: hooks.append(1)
    got = []

    def job(say):
        say("first")
        tab.progress_panel.stop_btn.click()          # Stop while it works
        say("never gets past this")
        got.append("past")
    tab.run(job, got.append)
    assert got == [] and hooks == [1] and said[-1].startswith("Stopped")
    assert not tab.progress_panel.isVisibleTo(tab)


def test_every_page_gets_a_panel(qt):
    """The automatic panel lands inside each page's layout (not floating)."""
    from PySide6.QtWidgets import QScrollArea
    from pixel_pipeline.ui.cards_tab import CardsTab
    from pixel_pipeline.ui.effects_tab import EffectsTab
    from pixel_pipeline.ui.make_tab import MakeTab
    from pixel_pipeline.ui.portraits_tab import PortraitsTab
    from pixel_pipeline.ui.tilegen_tab import TilegenTab
    from pixel_pipeline.ui.uikit_tab import UIKitTab
    from pixel_pipeline.ui.weather_tab import WeatherTab
    from pixel_pipeline.ui.objects_tab import ObjectsTab
    from pixel_pipeline.ui.assets_tab import AssetsTab
    def in_layout(lay, w):
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.widget() is w or (item.layout() is not None and in_layout(item.layout(), w)):
                return True
            sub = item.widget()
            if sub is not None and sub.layout() is not None and in_layout(sub.layout(), w):
                return True
            if isinstance(sub, QScrollArea) and sub.widget() is not None and                     sub.widget().layout() is not None and in_layout(sub.widget().layout(), w):
                return True
        return False
    for cls in (MakeTab, CardsTab, EffectsTab, PortraitsTab, TilegenTab, UIKitTab, WeatherTab,
                lambda: ObjectsTab(kind="item"), AssetsTab):
        page = cls()
        assert in_layout(page.layout(), page.progress_panel), page
        page.close()


def test_characters_page_add_shows_stages(qt, monkeypatch, tmp_path):
    from pixel_pipeline import library
    from pixel_pipeline.model3d import ModelResult
    from pixel_pipeline.ui.common import hub
    from pixel_pipeline.ui.model_studio import ModelStudio
    win = ModelStudio()
    win.sync = True
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    win.result = ModelResult(model, None, None, None)
    win.drawn.setChecked(True)
    win.front_first.setChecked(False)
    win.name.setText("Bot")
    snaps = []

    def fake_add(name, model, progress=None, **kw):
        progress("Adding a skeleton and animations…")
        for n in range(1, 17):
            progress(f"Drawing it facing S (reference, try o0.6; {n} of 16)…")
            P.gpu_sampling(20, 25)
            u = win.progress_panel.last
            snaps.append((u.stage, u.fraction, win.progress_panel.isVisibleTo(win)))
        for n in range(1, 17):
            progress(f"Bending the drawings: {n} of 16 directions done (about 10 s left)…")
        snaps.append((win.progress_panel.last.stage, win.progress_panel.last.fraction, True))
        ch = library.Character(name, folder=tmp_path / "bot")
        ch.folder.mkdir(parents=True, exist_ok=True)
        return ch
    monkeypatch.setattr(library, "add_character", fake_add)
    monkeypatch.setattr(win, "open_character", lambda ch: None)
    monkeypatch.setattr(win, "refresh_library", lambda select=None: None)
    win.add_to_library()
    assert snaps and all(vis for *_, vis in snaps)
    assert snaps[0][0] == "Drawing 16 directions" and snaps[-1][0] == "Bending the drawings"
    assert monotonic([f for _, f, _ in snaps]) and snaps[-1][1] > 0.9
    assert not win.progress_panel.isVisibleTo(win) and hub().current() is None
    win.close()


def test_global_bar_and_title(qt):
    from PySide6.QtWidgets import QMainWindow
    from pixel_pipeline.ui.common import GlobalProgress, ToolTab, hub
    w = QMainWindow()
    w.setWindowTitle("Pixel Pipeline — test")
    g = GlobalProgress(w)
    w.statusBar().addPermanentWidget(g)
    tab = ToolTab()
    tab.sync = True
    titles = []

    def job(say):
        say(1, 4)
        titles.append((w.windowTitle(), g.isVisibleTo(w), g.bar.value()))
        say(3, 4)
        titles.append((w.windowTitle(), g.isVisibleTo(w), g.bar.value()))
    tab.run(job, lambda r: None, title="Two steps")
    assert titles[0][0].startswith("25% · Pixel Pipeline") and titles[0][1]
    assert titles[1][2] > titles[0][2]
    assert w.windowTitle() == "Pixel Pipeline — test" and not g.isVisibleTo(w)
    assert hub().current() is None
    w.close()
