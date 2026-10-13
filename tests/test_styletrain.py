import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import assets as A
from pixel_pipeline import generate as G
from pixel_pipeline import styletrain as S


@pytest.fixture
def proj(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    return A.create_project("Moon Keep")


def sprite(w=16, h=24, colour=(200, 40, 40)):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[2:h - 2, 3:w - 3] = (*colour, 255)
    rgba[4, 5] = (10, 10, 10, 255)
    return rgba


def add_item(proj, name, tags=(), colour=(200, 40, 40)):
    a = A.new_asset("item", name, project=proj, prompt=f"a {name.lower()}")
    Image.fromarray(sprite(colour=colour), "RGBA").save(a.folder / "icon.png")
    if tags:
        A.set_tags(a, list(tags))
    return a


def add_character(proj, name):
    a = A.new_asset("character", name, project=proj, body="person", role="knight")
    sheet = np.zeros((24, 32, 4), np.uint8)
    sheet[:, :16] = sprite(colour=(0, 0, 255))        # idle_E 0
    sheet[:, 16:] = sprite(colour=(0, 255, 0))        # idle_S 0 = the front
    Image.fromarray(sheet, "RGBA").save(a.folder / "sheet.png")
    frames = [{"filename": n, "frame": {"x": x, "y": 0, "w": 16, "h": 24}}
              for n, x in (("idle_E 0", 0), ("idle_S 0", 16))]
    (a.folder / "sheet.json").write_text(json.dumps({"frames": frames, "meta": {}}))
    return a


def test_trigger_steps_estimate():
    assert S.trigger_word("Default") == "dfltpx"
    assert S.trigger_word("Moon Keep") == "mnkppx"
    assert S.trigger_word("Io") == "ppspx"
    assert S.auto_steps(30) == 1200 and S.auto_steps(5) == 600 and S.auto_steps(200) == 1650
    m30 = S.estimate_seconds(30) / 60
    assert 60 < m30 < 90                                  # 30 pictures fit the 90 min budget
    assert S.estimate_seconds(30, S.TrainConfig(resolution=768)) < S.estimate_seconds(30)
    assert S.estimate_seconds(30, done_steps=600) < S.estimate_seconds(30)


def test_sources_prepare_and_captions(proj):
    ch = add_character(proj, "Sir Brick")
    it = add_item(proj, "Iron Sword")
    rgba, src = S.source_sprite(ch)
    assert src.endswith("sheet.png") and tuple(rgba[10, 8, :3]) == (0, 255, 0)   # front frame
    assert S.source_sprite(it)[1].endswith("icon.png")
    img = S.prepare_image(rgba, 1024)
    arr = np.asarray(img)
    assert img.size == (1024, 1024) and img.mode == "RGB"
    assert tuple(arr[0, 0]) == (255, 255, 255)
    # crop 10x20 -> x38 (fill .75 of 1024 = 768 // 20), crisp: only the sprite's colours + white
    colours = {tuple(c) for c in arr.reshape(-1, 3)}
    assert colours == {(255, 255, 255), (0, 255, 0), (10, 10, 10)}
    ys, xs = np.nonzero((arr != 255).any(-1))
    assert (ys.max() - ys.min() + 1) == 20 * 38 and (xs.max() - xs.min() + 1) == 10 * 38
    cap = S.caption(ch, "mnkppx")
    assert cap.startswith("mnkppx style, pixel art sprite of Sir Brick, knight") and "front view" in cap
    assert "a iron sword" in S.caption(it, "x") and "item icon" in S.caption(it, "x")


def test_collect_approved_then_all(proj):
    add_item(proj, "Sword", ["approved"])
    add_item(proj, "Shield")
    add_character(proj, "Hero")
    assert [s.asset.name for s in S.collect(proj)] == ["Sword"]
    A.set_tags(next(a for a in A.list_assets("item", project=proj) if a.name == "Sword"), [])
    allp = S.collect(proj)
    assert {s.asset.name for s in allp} == {"Sword", "Shield", "Hero"}
    assert all(s.caption.startswith("mnkppx style") for s in allp)
    only = S.collect(proj, tag=None, only={"item/shield"})
    assert [s.asset.name for s in only] == ["Shield"]


def test_write_dataset(proj, tmp_path):
    samples = [S.Sample(add_item(proj, f"Gem {i}", colour=(i * 40, 90, 90)), sprite(colour=(i * 40, 90, 90)),
                        f"cap {i}", "") for i in range(3)]
    cfg = S.TrainConfig(resolution=256)
    ds = S.write_dataset(samples, tmp_path / "ds", cfg, "trg")
    pngs = sorted((tmp_path / "ds").glob("*.png"))
    assert ds.count == 3 and len(pngs) == 3
    assert pngs[0].with_suffix(".txt").read_text() == "cap 0"
    assert Image.open(pngs[0]).size == (256, 256)
    again = S.write_dataset(samples, tmp_path / "ds", cfg, "trg")
    assert again.signature == ds.signature
    mirrored = S.write_dataset(samples, tmp_path / "ds", S.TrainConfig(resolution=256, mirror=True), "trg")
    assert mirrored.count == 6 and mirrored.signature != ds.signature
    other = S.write_dataset(samples, tmp_path / "ds", S.TrainConfig(resolution=256, rank=16), "trg")
    assert other.signature != ds.signature
    with pytest.raises(S.StyleTrainError):
        S.write_dataset([], tmp_path / "ds", cfg, "trg")


# Every input of TrainLoraNode in ComfyUI 0.39 comfy_extras/nodes_train.py.
TRAIN_INPUTS = {"model", "latents", "positive", "batch_size", "grad_accumulation_steps", "steps",
                "learning_rate", "rank", "optimizer", "loss_function", "seed", "training_dtype",
                "lora_dtype", "quantized_backward", "algorithm", "gradient_checkpointing",
                "checkpoint_depth", "offloading", "existing_lora", "bucket_mode", "bypass_mode"}


def test_workflows():
    cfg = S.TrainConfig()
    wf = S.train_workflow("pp_style/moon-keep", cfg, steps=150, seed=7,
                          existing_lora="pp_style\\moon-keep_ckpt_150_steps_00001_.safetensors",
                          prefix="loras/pp_style/moon-keep_c150")
    t = wf["4"]["inputs"]
    assert wf["4"]["class_type"] == "TrainLoraNode" and set(t) == TRAIN_INPUTS
    assert t["model"] == ["2", 0] and wf["2"]["inputs"]["lora_name"] == "pixel-art-xl.safetensors"
    assert t["latents"] == ["3", 0] and t["positive"] == ["3", 1] and t["gradient_checkpointing"]
    assert wf["3"] == {"class_type": "LoadTrainingDataset", "inputs": {"folder_name": "pp_style/moon-keep"},
                       "_meta": {"title": "Encoded dataset"}}
    assert wf["5"]["inputs"] == {"lora": ["4", 0], "prefix": "loras/pp_style/moon-keep_c150", "steps": ["4", 2]}
    alone = S.train_workflow("d", S.TrainConfig(base_lora="none"), steps=1, seed=0, prefix="p")
    assert "2" not in alone and alone["4"]["inputs"]["model"] == ["1", 0]
    enc = S.encode_workflow("pp_style/m", "pp_style/m", cfg, vae="sdxl_vae.safetensors")
    assert enc["5"]["inputs"] == {"images": ["4", 0], "texts": ["4", 1], "vae": ["3", 0], "clip": ["2", 1]}
    assert enc["6"]["class_type"] == "SaveTrainingDataset" and enc["6"]["inputs"]["folder_name"] == "pp_style/m"


def test_apply_style_lora_hook():
    wf = G.build_workflow("p", "n", seed=1)
    S.apply_style_lora(wf, "moon-keep_style.safetensors", 0.7)
    assert wf["15"]["inputs"]["model"] == ["2", 0] and wf["15"]["inputs"]["clip"] == ["2", 1]
    assert wf["10"]["inputs"]["model"] == ["15", 0]
    assert wf["6"]["inputs"]["clip"] == wf["7"]["inputs"]["clip"] == ["15", 1]
    fast = G.build_workflow("p", "n", seed=1, lcm_lora="lcm.safetensors")
    S.apply_style_lora(fast, "s.safetensors")
    assert fast["4"]["inputs"]["model"] == ["15", 0] and fast["15"]["inputs"]["model"] == ["3", 0]
    assert fast["10"]["inputs"]["model"] == ["4", 0]
    assert S.styled_prompt("pixel art of a cat", "mnkppx") == "mnkppx style, pixel art of a cat"
    assert S.styled_prompt("mnkppx style, cat", "mnkppx") == "mnkppx style, cat"


def test_parse_progress():
    text = ("\rTraining LoRA:  10%|#         | 15/150 [00:30<04:30,  2.01s/it, loss=0.1234]"
            "\rTraining LoRA:  11%|#1        | 16/150 [00:32<04:28,  2.00s/it, loss=0.0987]")
    assert S.parse_progress(text) == (16, 150, 0.0987, 2.0)
    fast = S.parse_progress("Training LoRA:  50%|#####| 2/4 [00:01<00:01,  2.50it/s, loss=1e-2]")
    assert fast[:3] == (2, 4, 0.01) and abs(fast[3] - 0.4) < 1e-9
    assert S.parse_progress("loading model") is None


def test_read_gpu(monkeypatch):
    class Out:
        returncode, stdout = 0, "71, 98, 6123, 8188\n"
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: Out())
    assert S.read_gpu() == S.GpuStatus(71.0, 98.0, 6123.0, 8188.0)

    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(S.subprocess, "run", missing)
    assert S.read_gpu() is None


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def gems(proj, n=4):
    return S.collect(proj, tag=None) if n == 0 else [
        S.Sample(add_item(proj, f"Gem {i}"), sprite(), f"mnkppx style, gem {i}", "") for i in range(n)]


def cfg_small(**kw):
    base = dict(resolution=64, steps=6, chunk_steps=2, rest_seconds=5, poll_seconds=0)
    base.update(kw)
    return S.TrainConfig(**base)


def test_full_dry_run(proj, tmp_path):
    clock, seen = Clock(), []
    fake = S.FakeBackend(tmp_path / "comfy")
    tr = S.StyleTrainer(proj, cfg_small(), backend=fake, gpu=lambda: S.GpuStatus(60.0),
                        sleep=clock.sleep, clock=clock, progress=seen.append)
    st = tr.run(gems(proj))
    assert st.status == "done" and st.done_steps == 6 and st.result == "moon-keep_style.safetensors"
    final = tmp_path / "comfy" / "loras" / "moon-keep_style.safetensors"
    assert final.exists() and final.read_bytes()[8:9] == b"{"
    # chunks continue the previous checkpoint; only the newest checkpoint is kept
    assert [c["existing"] for c in fake.chunks] == [
        None, "moon-keep_ckpt_2_steps_00001_.safetensors", "moon-keep_ckpt_4_steps_00001_.safetensors"]
    assert [p.name for p in (tmp_path / "comfy" / "loras" / "pp_style").iterdir()] == [
        "moon-keep_ckpt_6_steps_00001_.safetensors"]
    assert fake.prepared == ["pp_style/moon-keep"]
    assert [c["seed"] for c in fake.chunks] == [0, 2, 4]
    phases = [p.phase for p in seen]
    assert phases[0] == "dataset" and "encode" in phases and "rest" in phases and phases[-1] == "done"
    assert max(p.step for p in seen) == 6 and any(p.loss for p in seen)
    rec = S.project_style_lora(proj)
    assert rec == {"lora": "moon-keep_style.safetensors", "trigger": "mnkppx", "strength": 0.8}
    saved = json.loads((proj.folder / "style_lora.json").read_text())
    assert saved["steps"] == 6 and saved["base_lora"] == "pixel-art-xl.safetensors"
    assert S.load_state(proj).status == "done"


def test_stop_and_resume(proj, tmp_path):
    clock = Clock()
    fake = S.FakeBackend(tmp_path / "comfy")
    samples = gems(proj)
    tr = None

    def stop_after_first_checkpoint(p):
        if p.message.startswith("Checkpoint at step 2"):
            tr.stop()
    tr = S.StyleTrainer(proj, cfg_small(), backend=fake, gpu=lambda: None, sleep=clock.sleep,
                        clock=clock, progress=stop_after_first_checkpoint)
    st = tr.run(samples)
    assert st.status == "stopped" and st.done_steps == 2 and S.load_state(proj).done_steps == 2
    assert not (tmp_path / "comfy" / "loras" / "moon-keep_style.safetensors").exists()
    plan = S.StyleTrainer(proj, cfg_small(), backend=fake).plan(samples)
    assert plan["done"] == 2 and plan["steps"] == 6

    # Stop pressed in the middle of a chunk: that chunk is dropped, the checkpoint stays
    tr2 = None

    def stop_mid_chunk(p):
        if p.phase == "train" and p.step == 3:
            tr2.stop()
    tr2 = S.StyleTrainer(proj, cfg_small(), backend=fake, gpu=lambda: None, sleep=clock.sleep,
                         clock=clock, progress=stop_mid_chunk)
    st = tr2.run(samples)
    assert st.status == "stopped" and st.done_steps == 2 and len(fake.prepared) == 1  # encoded once

    st = S.StyleTrainer(proj, cfg_small(), backend=fake, gpu=lambda: None, sleep=clock.sleep,
                        clock=clock).run(samples)
    assert st.status == "done" and st.done_steps == 6
    assert any("resumed at step 2" in n for n in st.notes)
    # new pictures = new signature = a fresh run
    more = samples + gems(proj, 1)
    st = S.StyleTrainer(proj, cfg_small(), backend=fake, gpu=lambda: None, sleep=clock.sleep,
                        clock=clock).run(more)
    assert st.done_steps == 6 and len(fake.prepared) == 2
    # same pictures again after done = train more on top
    st = S.StyleTrainer(proj, cfg_small(steps=2), backend=fake, gpu=lambda: None,
                        sleep=clock.sleep, clock=clock).run(more)
    assert st.status == "done" and st.done_steps == 8


def test_pacing_cool_down_and_hard_limit(proj, tmp_path):
    clock = Clock()
    cur = [85.0]                                         # hot start, cooling 5 °C per reading

    def gpu():
        t = cur[0]
        cur[0] = 60.0 if t >= 90 else max(60.0, t - 5)   # a spike is read once
        return S.GpuStatus(t)

    class Spiky(S.FakeBackend):
        def train_chunk(self, *a, **k):
            if len(self.chunks) == 1:
                cur[0] = 90.0                            # the second chunk gets hot
            return super().train_chunk(*a, **k)
    fake = Spiky(tmp_path / "comfy")
    seen = []
    st = S.StyleTrainer(proj, cfg_small(steps=4), backend=fake, gpu=gpu, sleep=clock.sleep,
                        clock=clock, progress=seen.append).run(gems(proj))
    assert st.status == "done" and st.done_steps == 4
    assert any(p.phase == "cool" and "waiting for 70" in p.message for p in seen)
    assert any("interrupted (GPU reached 90" in n for n in st.notes)
    assert len(fake.chunks) == 3                         # one chunk redone after the spike
    assert clock.t >= 20                                 # it waited / rested on the fake clock


def test_never_cools_stops_resumable(proj, tmp_path):
    clock = Clock()
    st = S.StyleTrainer(proj, cfg_small(max_cool_minutes=1), backend=S.FakeBackend(tmp_path),
                        gpu=lambda: S.GpuStatus(86.0), sleep=clock.sleep, clock=clock).run(gems(proj))
    assert st.status == "stopped" and "didn't cool" in st.notes[-1]


def test_always_too_hot_in_chunks_stops(proj, tmp_path):
    clock = Clock()
    temps = iter([60, 95] * 100)
    st = S.StyleTrainer(proj, cfg_small(), backend=S.FakeBackend(tmp_path),
                        gpu=lambda: S.GpuStatus(float(next(temps))), sleep=clock.sleep,
                        clock=clock).run(gems(proj))
    assert st.status == "stopped" and "3 chunks in a row" in st.notes[-1]


def test_time_budget_trims(proj, tmp_path):
    clock = Clock()

    class Slow(S.FakeBackend):
        def train_chunk(self, *a, steps, **k):
            clock.t += steps * 600                       # 10 min per step
            return super().train_chunk(*a, steps=steps, **k)
    st = S.StyleTrainer(proj, cfg_small(max_minutes=50), backend=Slow(tmp_path), gpu=lambda: None,
                        sleep=clock.sleep, clock=clock).run(gems(proj))
    assert st.status == "done" and st.done_steps == 4 and st.total_steps == 4
    assert any("time budget" in n for n in st.notes)
    assert st.sec_per_step > 500


class FakeClient:
    """Enough of generate.ComfyClient for ComfyBackend."""

    def __init__(self, root: Path):
        self.root = root
        self.queued, self.polls, self.interrupted = [], 0, 0
        self.loras = []
        self.fail = None

    def ensure_running(self, **_):
        return {}

    def list_models(self, kind):
        return {"vae": ["sdxl_vae.safetensors"], "loras": self.loras}[kind]

    def queue(self, wf):
        self.queued.append(wf)
        node = next(n for n in wf.values() if n["class_type"] in ("SaveLoRA", "SaveTrainingDataset"))
        if node["class_type"] == "SaveLoRA":
            train = next(n for n in wf.values() if n["class_type"] == "TrainLoraNode")["inputs"]
            ex = train["existing_lora"]
            before = 0 if ex == "[None]" else int(ex.split("_steps_")[0].split("_")[-1])
            pre = Path(node["inputs"]["prefix"])
            out = self.root / "output" / pre.parent / f"{pre.name}_{before + train['steps']}_steps_00001_.safetensors"
            S._fake_safetensors(out, {})
        return f"p{len(self.queued)}"

    def _get_json(self, path):
        if path == "/object_info/TrainLoraNode":
            return {"TrainLoraNode": {}}
        if path == "/internal/logs/raw":
            return {"entries": [{"m": "\rTraining LoRA:  50%|#| 1/2 [00:02<00:02,  2.00s/it, loss=0.2]"}]}
        self.polls += 1
        pid = path.rsplit("/", 1)[1]
        if self.polls < 2:
            return {}
        if self.fail:
            return {pid: {"status": {"status_str": "error", "messages": [[self.fail, {}]]}}}
        return {pid: {"status": {"status_str": "success"}, "outputs": {}}}

    def _request(self, method, path, body=None, headers=None):
        if path == "/interrupt":
            self.interrupted += 1
        return b""


def test_comfy_backend_with_fake_client(proj, tmp_path):
    root = tmp_path / "ComfyUI"
    client = FakeClient(root)
    be = S.ComfyBackend(client, root)
    ds = S.write_dataset(gems(proj, 2), tmp_path / "ds", cfg_small(), "trg")
    be.prepare(ds, "pp_style/moon-keep", cfg_small())
    assert sorted(p.name for p in (root / "input" / "pp_style" / "moon-keep").iterdir())[:2] == [
        "000_gem-0.png", "000_gem-0.txt"]
    assert client.queued[0]["3"]["inputs"]["vae_name"] == "sdxl_vae.safetensors"
    steps = []
    client.polls = 0
    out = be.train_chunk("pp_style/moon-keep", cfg_small(), steps=2, seed=0, existing=None,
                         prefix="moon-keep_c0", on_step=lambda *a: steps.append(a),
                         should_abort=lambda: None)
    assert out.name == "moon-keep_c0_2_steps_00001_.safetensors" and out.exists()
    assert steps and steps[0] == (1, 2, 0.2, 2.0)
    # continuing a checkpoint: the name ComfyUI lists (Windows backslash) is passed on
    ck = root / "models" / "loras" / "pp_style" / "moon-keep_ckpt_2_steps_00001_.safetensors"
    client.loras = ["pixel-art-xl.safetensors", "pp_style\\" + ck.name]
    client.polls = 0
    out = be.train_chunk("pp_style/moon-keep", cfg_small(), steps=2, seed=2, existing=ck,
                         prefix="moon-keep_c2", on_step=lambda *a: None, should_abort=lambda: None)
    assert client.queued[-1]["4"]["inputs"]["existing_lora"] == "pp_style\\" + ck.name
    assert out.name == "moon-keep_c2_4_steps_00001_.safetensors"
    # abort -> /interrupt + Interrupted
    client.polls = -5
    with pytest.raises(S.Interrupted):
        be.train_chunk("pp_style/moon-keep", cfg_small(), steps=2, seed=4, existing=None,
                       prefix="x", on_step=lambda *a: None, should_abort=lambda: "hot")
    assert client.interrupted == 1
    client.polls, client.fail = 0, "execution_interrupted"
    with pytest.raises(S.Interrupted):
        be.train_chunk("pp_style/moon-keep", cfg_small(), steps=2, seed=4, existing=None,
                       prefix="y", on_step=lambda *a: None, should_abort=lambda: None)


def test_dialog_dry_run(proj, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from PySide6.QtCore import Qt
    from pixel_pipeline.ui.style_train import StyleTrainDialog
    for i in range(3):
        add_item(proj, f"Gem {i}", ["approved"])
    add_item(proj, "Rock")
    dlg = StyleTrainDialog(proj, backend=S.FakeBackend(tmp_path / "comfy"), gpu=lambda: S.GpuStatus(55.0),
                           cfg=cfg_small(rest_seconds=0))
    dlg.sync = True
    assert dlg.list.count() == 3 and "3 pictures" in dlg.info.text() and "fewer than 20" in dlg.info.text()
    assert not dlg.test_btn.isEnabled()
    dlg.approved.setChecked(False)
    assert dlg.list.count() == 4
    dlg.list.item(0).setCheckState(Qt.Unchecked)
    assert "3 pictures" in dlg.info.text()
    dlg.trigger.setText("moonpx")
    dlg.start()
    assert dlg.result_state.status == "done" and "Done: moon-keep_style.safetensors" in dlg.status.text()
    assert dlg.bar.value() == 6 and "55" in dlg.temp.text()
    caps = sorted((proj.folder / "style_train" / "dataset").glob("*.txt"))
    assert len(caps) == 3 and all(c.read_text().startswith("moonpx style") for c in caps)
    dlg.tick_none()
    assert not dlg.start_btn.isEnabled()
    dlg._show_record()
    assert dlg.test_btn.isEnabled()
    dlg.setParent(None)


# ---- kohya backend (fake kohya: a run_kohya.py that prints kohya's progress and writes the file)

FAKE_KOHYA = r'''
import sys, time, pathlib
a = sys.argv
out = pathlib.Path(a[a.index("--output_dir") + 1]) / (a[a.index("--output_name") + 1] + ".safetensors")
n = int(a[a.index("--max_train_steps") + 1])
for i in range(1, n + 1):
    sys.stdout.write(f"\rsteps: {100 * i // n}%|##| {i}/{n} [00:0{i}<00:00, 1.76s/it, avr_loss=0.0{i}]")
    sys.stdout.flush()
    time.sleep(0.05)
if "--slow" not in " ".join(a):
    out.write_bytes(b"lora")
print("\nmodel saved.")
'''


def _fake_kohya(tmp_path):
    import sys
    k = tmp_path / "kohya"
    (k / "src" / "sd-scripts-main").mkdir(parents=True)
    (k / "run_kohya.py").write_text(FAKE_KOHYA)
    root = tmp_path / "ComfyUI"
    for d in ("checkpoints", "loras", "vae"):
        (root / "models" / d).mkdir(parents=True)
    (root / "models" / "vae" / "sdxl_vae.safetensors").write_bytes(b"v")
    return S.KohyaBackend(k, root, python=sys.executable)


def test_parse_kohya_progress():
    text = "steps:  45%|####5     | 9/20 [00:16<00:19,  1.76s/it, avr_loss=0.0319]"
    assert S.parse_kohya(text) == (9, 20, 0.0319, 1.76)
    assert S.parse_kohya("loading U-Net") is None


def test_kohya_command_continues_and_stacks_pixel_lora(tmp_path):
    b = _fake_kohya(tmp_path)
    b.data = tmp_path / "data"
    cmd = b.command(S.TrainConfig(resolution=768), steps=150, seed=3,
                    existing=tmp_path / "ck.safetensors", out_dir=tmp_path / "out", prefix="p")
    j = " ".join(cmd)
    assert "--fp8_base" in cmd and "AdamW8bit" in cmd and "--resolution 768" in j
    assert "--network_weights" in cmd and "pixel-art-xl.safetensors" in j and "sdxl_vae" in j
    cmd = b.command(S.TrainConfig(base_lora="none"), steps=1, seed=0, existing=None,
                    out_dir=tmp_path, prefix="p")
    assert "--base_weights" not in cmd and "--network_weights" not in cmd


def test_kohya_chunk_reports_steps_and_returns_the_lora(tmp_path, monkeypatch):
    import pixel_pipeline.generate as gen
    monkeypatch.setattr(gen, "stop_started_comfyui", lambda: None)
    monkeypatch.setattr(S.KohyaBackend, "_free_gpu", lambda self: None)
    b = _fake_kohya(tmp_path)
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    (ds_dir / "000_a.png").write_bytes(b"x")
    (ds_dir / "000_a.txt").write_text("cap")
    b.prepare(S.Dataset(ds_dir, 1, 1, "trig", "sig"), "pp_style/proj", S.TrainConfig())
    assert (b.data / "1_trig" / "000_a.txt").exists()
    seen = []
    out = b.train_chunk("pp_style/proj", S.TrainConfig(), steps=5, seed=1, existing=None,
                        prefix="proj_c0", on_step=lambda *a: seen.append(a),
                        should_abort=lambda: None)
    assert out.read_bytes() == b"lora" and seen[-1][:2] == (5, 5) and seen[-1][3] == 1.76


def test_kohya_chunk_stops_when_too_hot(tmp_path, monkeypatch):
    monkeypatch.setattr(S.KohyaBackend, "_free_gpu", lambda self: None)
    b = _fake_kohya(tmp_path)
    b.data = tmp_path / "data"
    with pytest.raises(S.Interrupted, match="°C"):
        b.train_chunk("pp_style/proj", S.TrainConfig(), steps=200, seed=1, existing=None,
                      prefix="p", on_step=lambda *a: None, should_abort=lambda: "GPU reached 88 °C")


def test_default_backend_prefers_kohya(tmp_path, monkeypatch):
    monkeypatch.setenv(S.KOHYA_ENV, str(tmp_path / "nowhere"))
    monkeypatch.setattr(S, "KOHYA_DEFAULT", tmp_path / "nowhere2")
    assert S.find_kohya() is None
    k = tmp_path / "k"
    (k / "src" / "sd-scripts-main").mkdir(parents=True)
    (k / "run_kohya.py").write_text("")
    monkeypatch.setenv(S.KOHYA_ENV, str(k))
    assert S.find_kohya() == k
