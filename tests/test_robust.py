"""Surviving a GPU driver reset / ComfyUI crash (generate.GpuRetry, drawn characters resuming,
batches stopping cleanly), ComfyUI log rotation, tile pattern checks, and drawn props becoming
game 3D models. The GPU is the fake ComfyUI of test_generate, made to die mid-run."""
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import generate as g
from pixel_pipeline.assets import Style
from test_generate import FakeComfy, Handler, white_key  # noqa: F401  (pytest fixture)


class DyingHandler(Handler):
    """The fake ComfyUI, except that once `die_at` prompts are queued it stops answering (the
    connection drops, like a crashed ComfyUI) for the next `down` requests."""

    def _dead(self) -> bool:
        f = self.server.fake
        if f.die_at is not None and len(f.prompts) >= f.die_at and f.down > 0:
            f.down -= 1
            f.refused += 1
            if f.down == 0:
                f.die_at = None                  # back up (one crash)
            self.close_connection = True
            return True
        return False

    def do_GET(self):
        if not self._dead():
            super().do_GET()

    def do_POST(self):
        if not self._dead():
            super().do_POST()


@pytest.fixture
def dying(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), DyingHandler)
    fake = server.fake = FakeComfy()
    fake.die_at, fake.down, fake.refused = None, 0, 0
    fake.history_delay = 0
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    client = g.ComfyClient(f"http://127.0.0.1:{server.server_address[1]}", timeout=5, poll_interval=0.01)
    slept = []
    monkeypatch.setattr(g, "_sleep", slept.append)          # backoff without waiting
    monkeypatch.setenv("PIXEL_GPU_RETRIES", "3")
    fake.slept = slept
    yield fake, client
    server.shutdown()
    server.server_close()


# --------------------------------------------------------------------------- generate level

def test_pictures_survive_a_crash_and_only_the_missing_ones_are_made_again(dying):
    fake, client = dying
    fake.die_at, fake.down = 3, 3            # all 3 queued, then ComfyUI drops 3 requests
    said = []
    with g.gpu_retry(say=said.append) as r:
        out = g.generate_images("a fox", client=client, seed=5, count=3)
    assert [s for _, s in out] == [5, 6, 7] and all(img.size for img, _ in out)
    assert r.failures == 1 and fake.refused == 3 and fake.slept        # waited, backing off
    assert len(fake.prompts) == 6           # the 3 queued before the crash were lost: queued again
    assert any("ComfyUI failed while making a picture" in s for s in said)
    assert any("Waiting for ComfyUI" in s for s in said)


def test_gives_up_after_n_failures_with_the_log(dying, tmp_path, monkeypatch):
    fake, client = dying
    monkeypatch.setattr(g, "COMFYUI_LOG", tmp_path / "comfyui.log")
    (tmp_path / "comfyui.log").write_text("loading\ntorch.AcceleratorError: CUDA error: unspecified launch failure\n")
    monkeypatch.setenv("PIXEL_GPU_RECOVER_S", "1")
    fake.die_at, fake.down = 1, 10**9          # up, then dies for good
    with pytest.raises(g.ComfyUIGaveUp) as e:
        g.generate_images("a fox", client=client, seed=5)
    text = str(e.value)
    assert "kept failing" in text and "comfyui.log" in text and "unspecified launch failure" in text
    assert isinstance(e.value, g.ComfyUIUnavailable)      # callers that fall back still do


def test_cuda_errors_are_retried_and_restart_our_comfyui(monkeypatch):
    err = g._execution_error({"messages": [["execution_error", {
        "node_type": "KSampler", "node_id": "5", "exception_type": "torch.AcceleratorError",
        "exception_message": "CUDA error: unspecified launch failure"}]]})
    assert g.gpu_fault(err) and g.gpu_fault(g.ComfyUIUnavailable("down"))
    assert not g.gpu_fault(g.ComfyUIWorkflowError("ComfyUI rejected the workflow: bad node"))
    assert not g.gpu_fault(g.ComfyUIModelMissing("no model", "checkpoints", "x"))
    assert not g.gpu_fault(g.ComfyUIGaveUp("gave up"))
    # our ComfyUI answers but its GPU is gone: it is stopped and started again

    class Proc:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, t):
            return 0
    proc = Proc()
    monkeypatch.setattr(g, "_started", [proc])
    monkeypatch.setattr(g, "find_comfyui", lambda: g.Path("C:/AI/ComfyUI_windows_portable"))
    client = g.ComfyClient(g.DEFAULT_URL)
    started = []
    monkeypatch.setattr(client, "ensure_running", lambda wait=0, say=None: started.append(wait) or {"system": {}})
    tries = []

    def flaky():
        tries.append(1)
        if len(tries) == 1:
            raise err
        return "ok"
    said = []
    assert g.GpuRetry(attempts=2, say=said.append).run(flaky, "the knight's NE drawing", client) == "ok"
    assert proc.stopped and started and len(tries) == 2
    assert any("Restarting ComfyUI" in s for s in said)


def test_comfyui_log_is_rotated_on_every_start(tmp_path, monkeypatch):
    import subprocess

    class FakeProc:
        def poll(self):
            return 0
    monkeypatch.setattr(g, "COMFYUI_LOG", tmp_path / "comfyui.log")
    monkeypatch.setattr(g, "comfyui_command", lambda folder: ["python", "main.py"])
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: FakeProc())
    monkeypatch.setattr(g, "_started", [])
    for i in range(8):
        g.COMFYUI_LOG.write_text(f"run {i}: CUDA error\n")
        g.start_comfyui(tmp_path)
    assert g.COMFYUI_LOG.exists() and g.COMFYUI_LOG.read_text() == ""       # the new run's
    assert g.old_log(1).read_text() == "run 7: CUDA error\n"                # the crash before it
    assert [g.old_log(i).exists() for i in range(1, 7)] == [True] * 5 + [False]
    assert g.old_log(5).read_text() == "run 3: CUDA error\n"
    assert "run 7" in g._log_tail(3, g.old_log())


# --------------------------------------------------------------------------- drawn characters

def _robot_library(monkeypatch, tmp_path, client):
    from pixel_pipeline import autorig, drawnrig as dr, library
    from test_anim3d import write_robot
    monkeypatch.setattr(g, "ComfyClient", lambda *a, **k: client)
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    monkeypatch.setattr(library, "add_project_clips", lambda *a, **k: [])
    monkeypatch.setattr(autorig, "auto_rig", lambda model, out: write_robot(out))
    monkeypatch.setenv("PIXEL_DRAWN", "1")
    return write_robot(tmp_path / "robot_model.glb")


def _full():
    from pixel_pipeline import drawnrig as dr
    return 3 * len(dr.CANDIDATES) + len(dr.BACK_CANDIDATES)


def test_drawn_character_waits_out_a_crash_and_carries_on(dying, tmp_path, monkeypatch):
    from pixel_pipeline import library
    fake, client = dying
    model = _robot_library(monkeypatch, tmp_path, client)
    fake.die_at, fake.down = 3, 4              # dies after the 3rd drawing is queued
    said = []
    ch = library.add_character("Bot", model, prompt="a robot", progress=said.append)
    assert library.is_drawn(ch) and not ch.notice                    # drawn after all
    assert len(fake.prompts) == _full() + 1        # only the lost drawing was made twice
    assert any("ComfyUI failed while making" in s for s in said)


def test_drawn_character_falls_back_keeps_drawings_and_finishes_later(dying, tmp_path, monkeypatch):
    from pixel_pipeline import library
    fake, client = dying
    model = _robot_library(monkeypatch, tmp_path, client)
    monkeypatch.setenv("PIXEL_GPU_RETRIES", "1")
    monkeypatch.setenv("PIXEL_GPU_RECOVER_S", "1")
    fake.die_at, fake.down = 3, 10**9          # dies after the 3rd drawing and stays down
    said = []
    ch = library.add_character("Bot", model, prompt="a robot", progress=said.append)
    keys = ch.folder / "keys"
    drawn = sorted(p.name for p in keys.glob("drawn_*.png"))
    assert len(drawn) == 2                                           # the finished ones, kept
    assert not library.is_drawn(ch) and library.drawing_unfinished(ch)
    assert ch.sheet.exists()                                         # the 3D render for now
    assert "3D render" in ch.notice and "2 finished drawing" in ch.notice
    assert "Finish the drawn sprites" in ch.notice and "kept failing" in ch.notice
    assert any("couldn't draw the sprites" in s for s in said)       # never silent
    assert json.loads((keys / "keys.json").read_text())["partial"]
    before = {n: (keys / n).read_bytes() for n in drawn}
    # ComfyUI is back: finishing draws only what is missing
    fake.die_at, fake.down = None, 0
    n = len(fake.prompts)
    ch = library.finish_drawing(library.get("Bot"))
    assert library.is_drawn(ch) and not library.drawing_unfinished(ch)
    assert len(fake.prompts) - n == _full() - 2
    assert all((keys / k).read_bytes() == v for k, v in before.items())


def test_make_jobs_share_one_retry_and_batches_stop_on_a_dead_gpu(tmp_path):
    from pixel_pipeline import jobqueue, makeany as M
    from pixel_pipeline.assets import create_project
    proj = create_project("Robust")
    seen = []

    def maker(plan, project, say, client=None):
        r = g.current_retry()
        seen.append(r)
        assert r is g.current_retry()                 # one policy for the whole job
        raise g.ComfyUIGaveUp("ComfyUI kept failing (GPU driver reset or crash?)")
    with pytest.raises(g.ComfyUIGaveUp):
        M.run_plan(M.route("rusty iron sword"), proj, makers={"item": maker})
    q = jobqueue.Queue(proj)
    jobs = M.queue_plans(proj, [M.route("rusty iron sword"), M.route("a wolf")], queue=q)

    def worker(name, prompt, options, project):
        return M.queue_worker(name, prompt, options, project, makers={"item": maker})
    done = jobqueue.run_queue(proj, {"make": worker}, pause_s=0, queue=q, sleep=lambda s: None,
                              job_ids={j.id for j in jobs})
    first, second = q.get(jobs[0].id), q.get(jobs[1].id)
    assert [j.id for j in done] == [first.id]                        # stopped: the GPU is gone
    assert first.status == "pending" and "kept failing" in first.error   # carries on next run
    assert second.status == "pending" and not second.error
    assert "not done" in M.batch_summary([first, second])


# --------------------------------------------------------------------------- tiles

N = 512
_yy, _xx = np.mgrid[0:N, 0:N]


def bricks() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((N, N, 3), 150, np.uint8)
    img[(_yy % 32) < 4] = 60
    img[((_xx + (_yy // 32 % 2) * 32) % 64) < 4] = 60
    return (img + rng.normal(0, 12, img.shape)).clip(0, 255).astype(np.uint8)


def cobbles() -> np.ndarray:
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, N, (220, 2))
    tree = cKDTree(np.concatenate([pts + [dx, dy] for dx in (-N, 0, N) for dy in (-N, 0, N)]))
    d, _ = tree.query(np.stack([_xx.ravel(), _yy.ravel()], 1), k=2)
    img = np.clip(60 + (d[:, 1] - d[:, 0]).reshape(N, N) * 6, 60, 170)[..., None].repeat(3, 2)
    return (img + rng.normal(0, 12, img.shape)).clip(0, 255).astype(np.uint8)


def water(grid: bool = False) -> np.ndarray:
    import cv2
    rng = np.random.default_rng(2)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (N, N)).astype(np.float32), (0, 0), 6)
    img = np.stack([40 + noise * 300, 90 + noise * 300, 200 + noise * 200 + 20 * np.sin(_yy / 9.0)],
                   -1).clip(0, 255).astype(np.uint8)
    if grid:                                  # QA lagoon: straight light dashes on a grid
        on = ((_yy % 40) < 2) & ((_xx // 20) % 3 != 0) | ((_xx % 40) < 2) & ((_yy // 20) % 3 != 0)
        img[on] = np.clip(img[on].astype(int) + 70, 0, 255)
    return img


def test_pattern_checks_tell_bricks_from_cobbles_and_spot_grids():
    from pixel_pipeline import tilegen as tg
    assert tg.pattern_miss("road", bricks()) > 0.3
    assert tg.pattern_miss("road", cobbles()) == 0
    assert tg.pattern_miss("road", tg.seamless(cobbles())) == 0      # after tiling too
    assert tg.pattern_miss("water", water(grid=True)) > 0.2
    assert tg.pattern_miss("water", water()) == 0
    assert tg.pattern_miss("water", tg.even_out(tg.seamless(water()))) == 0
    assert tg.pattern_miss("wall", bricks()) == 0                    # walls are bricks
    assert tg.pattern_miss("cliff", water(grid=True)) == 0           # not checked there
    pos, neg = tg.swatch_prompt("road", "", style=Style())
    assert "rounded" in pos and "bricks" in neg and "cobble" in pos
    pos2, neg2 = tg.swatch_prompt("road", "", style=Style(), pattern=True)
    assert "(round cobbles" in pos2 and "(bricks:1.5)" in neg2


def test_a_brick_road_or_a_grid_lagoon_is_drawn_again(monkeypatch):
    from pixel_pipeline import assets, tilegen as tg
    calls = []

    def fake_generate(prompt, negative="", *, seed=None, count=1, **kw):
        calls.append((prompt, negative, seed))
        good = "(round cobbles" in prompt if "cobble" in prompt else "(organic flowing" in prompt
        img = (cobbles() if good else bricks()) if "cobble" in prompt else water(grid=not good)
        return [(Image.fromarray(img), (seed or 0) + i) for i in range(count)]
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", fake_generate)
    said = []
    sw, = tg.make_swatches("road", count=1, seed=7, style=assets.Style(tile=16), say=said.append)
    assert len(calls) == 2 and sw.pattern_miss == 0 and sw.seed == 7 + 1000
    assert "(bricks:1.5)" in calls[1][1] and any("wrong pattern" in s for s in said)
    calls.clear()
    sw, = tg.make_swatches("water", count=1, seed=3, style=assets.Style(tile=16))
    assert len(calls) == 2 and sw.pattern_miss == 0 and sw.colour_miss == 0
    calls.clear()                              # never good: the least bad is kept, said so
    monkeypatch.setattr(tg, "PATTERN_WORDS", {"road": ("(still bricks", "x")})
    sw, = tg.make_swatches("road", count=1, seed=7, style=assets.Style(tile=16), retries=2)
    assert len(calls) == 3 and sw.pattern_miss > 0


# --------------------------------------------------------------------------- drawn props -> 3D

def test_drawn_prop_gets_a_3d_shape_on_export(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    import trimesh
    from pixel_pipeline import gamemodel, model3d
    from pixel_pipeline.assets import create_project, new_asset
    from pixel_pipeline.ui.game_model_dialog import can_make, info_text
    proj = create_project("Props3D")
    barrel = new_asset("prop", "Wooden barrel", proj, source="drawn", look="drawn", directions=1,
                       chosen=2)
    for i in (1, 2, 3):
        Image.new("RGB", (64, 64), (120 + i, 80, 40)).save(barrel.folder / f"drawn_v{i}.png")
    assert gamemodel.drawn_picture(barrel) == barrel.folder / "drawn_v2.png"
    assert gamemodel.needs_shape(barrel) and can_make(barrel)
    assert "GPU" in info_text(barrel) and "1.5 min" in info_text(barrel)
    with pytest.raises(ValueError):
        gamemodel.source_model(barrel)          # no model yet
    shaped, built = [], []

    def fake_model(prompt, out, *, picture=None, subject="person", views=None, **kw):
        shaped.append((prompt, picture, subject, views))
        trimesh.creation.box((0.6, 1.0, 0.6)).export(out)
        return model3d.ModelResult(out, None, None, 1)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    monkeypatch.setattr(gamemodel, "make_game_model",
                        lambda model, out, **kw: built.append((model, kw["kind"])) or "built")
    said = []
    assert gamemodel.game_model_for(barrel, progress=said.append, look="pixel") == "built"
    assert shaped == [(None, barrel.folder / "drawn_v2.png", "object", "front")]
    assert built == [(barrel.folder / "model.glb", "prop")]
    assert any("GPU" in s for s in said)
    assert not gamemodel.needs_shape(barrel)    # built once: the next export reuses it
    sword = new_asset("item", "Sword", proj, icon_choice=3)
    Image.new("RGB", (64, 64), "grey").save(sword.folder / "picture_v3.png")
    assert gamemodel.drawn_picture(sword) == sword.folder / "picture_v3.png"


def test_not_running_at_the_start_fails_at_once(dying):
    fake, client = dying
    dead = g.ComfyClient("http://127.0.0.1:9", timeout=2)
    with pytest.raises(g.ComfyUIUnavailable) as e:
        g.generate_images("a fox", client=dead, seed=5)
    assert not isinstance(e.value, g.ComfyUIGaveUp) and not fake.slept     # no waiting


def _cobbles(n: int, cells: int, seed: int = 1) -> np.ndarray:
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    pts = rng.uniform(0, n, (cells, 2))
    tree = cKDTree(np.concatenate([pts + [dx, dy] for dx in (-n, 0, n) for dy in (-n, 0, n)]))
    d, _ = tree.query(np.stack([xx.ravel(), yy.ravel()], 1), k=2)
    img = np.clip(50 + (d[:, 1] - d[:, 0]).reshape(n, n) * 5, 50, 175)[..., None].repeat(3, 2)
    return img.astype(np.uint8)


def test_the_swatch_is_cropped_so_cobbles_survive_at_tile_size():
    """GPU 2026-10-15: ~15 cobbles across a 1024 picture squeezed into 32 px = grey noise."""
    from pixel_pipeline import tilegen as tg
    st = Style(tile=32)
    raw = _cobbles(1024, 15 * 15)                     # cells ~68 px, like the GPU road
    assert 50 <= tg.feature_size(raw) <= 90
    whole = tg.pixel_texture(tg.even_out(tg.seamless(raw)), 32, st)
    crop = tg.tile_crop(raw, "road", 32)
    assert 250 <= crop.shape[0] <= 450                  # ~5 cobbles across the tile
    cropped = tg.pixel_texture(tg.even_out(tg.seamless(crop)), 32, st)
    assert tg.tile_miss("road", whole) > 0             # the old way: lost to noise
    assert tg.tile_miss("road", cropped) == 0
    assert tg.tile_stats(cropped)["blob"] > 1.5 * tg.tile_stats(whole)["blob"]
    noise = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    assert tg.tile_miss("road", noise) > 0
    assert tg.tile_crop(_cobbles(512, 9), "road", 32).shape[0] >= 400   # coarse: (nearly) all of it


def test_flat_water_is_lifted_or_drawn_as_calm_ripples(monkeypatch):
    from pixel_pipeline import assets, tilegen as tg
    flat = np.zeros((N, N, 3), np.uint8)
    flat[:] = (40, 130, 205)
    flat = (flat + np.random.default_rng(3).normal(0, 0.4, flat.shape)).clip(0, 255).astype(np.uint8)
    assert tg.tile_miss("water", tg.pixel_texture(flat, 32, Style(tile=32))) > 0
    calls = []

    def fake_generate(prompt, negative="", *, seed=None, count=1, **kw):
        calls.append(prompt)
        return [(Image.fromarray(flat), (seed or 0) + i) for i in range(count)]
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", fake_generate)
    said = []
    sw, = tg.make_swatches("water", count=1, seed=3, style=assets.Style(tile=32), say=said.append)
    assert len(calls) == 3 and "(clearly visible ripples" in calls[1]       # redrawn, aimed at it
    st = tg.tile_stats(sw.texture)
    assert st["lstd"] >= tg.FLAT_MIN["water"] and sw.pattern_miss == 0      # readable in the end
    assert any("calm drawn ripples" in s for s in said)
    blue = sw.texture[..., :3].reshape(-1, 3).mean(0)
    assert blue[2] > blue[0] + 60                                            # still its blue


def test_rows_and_streaks_and_water_with_land_in_it_are_drawn_again():
    """House-style GPU run 2026-10-15: dirt / snow drawn in brick courses, sand as wood-grain
    streaks, the lagoon with islands and a palm in it."""
    from pixel_pipeline import tilegen as tg
    rng = np.random.default_rng(4)
    import cv2
    streaks = cv2.GaussianBlur(rng.normal(0, 1, (N, 8)).astype(np.float32), (0, 0), 1.5)
    streaks = cv2.resize(streaks, (N, N), interpolation=cv2.INTER_LINEAR)       # long horizontal
    sand = np.stack([200 + streaks * 60, 160 + streaks * 50, 100 + streaks * 30], -1).clip(0, 255).astype(np.uint8)
    assert tg.pattern_stats(sand)["rows"] > 0.6 and tg.pattern_miss("sand", sand) > 0
    assert tg.pattern_miss("road", bricks()) > tg.pattern_miss("road", cobbles()) == 0
    assert tg.pattern_stats(cobbles())["rows"] < 0.1
    lagoon = water()
    lagoon[150:350, 120:330] = (120, 80, 40)                    # an island
    assert tg.colour_miss("water", lagoon) > 0 and tg.colour_miss("water", water()) == 0
    crop = tg.tile_crop(lagoon, "water", 32)                     # the crop avoids the island
    assert tg.colour_stats(crop)["warm"] < tg.colour_stats(lagoon)["warm"]


def test_water_that_keeps_coming_out_as_islands_becomes_drawn_ripples(monkeypatch):
    from pixel_pipeline import assets, tilegen as tg
    lagoon = water()
    lagoon[100:400, 80:420] = (120, 80, 40)                      # mostly island, every try
    monkeypatch.setattr("pixel_pipeline.generate.generate_images",
                        lambda p, n="", *, seed=None, count=1, **kw:
                        [(Image.fromarray(lagoon), (seed or 0) + i) for i in range(count)])
    said = []
    sw, = tg.make_swatches("water", count=1, seed=3, style=assets.Style(tile=32), say=said.append)
    assert sw.colour_miss == 0 and "[neo water" in sw.prompt and any("land" in s for s in said)
    assert tg.tile_stats(sw.texture)["lstd"] >= tg.FLAT_MIN["water"]
