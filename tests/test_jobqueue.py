import json
import threading

import pytest

from pixel_pipeline import jobqueue as J


class FakeMaker:
    """Records calls; fails on names in `fail`; sets `stop` after `stop_after` calls."""

    def __init__(self, tmp_path, fail=(), stop=None, stop_after=None):
        self.calls, self.tmp, self.fail, self.stop, self.stop_after = [], tmp_path, set(fail), stop, stop_after

    def __call__(self, name, prompt, options, project):
        self.calls.append((name, prompt, dict(options)))
        if self.stop is not None and len(self.calls) == self.stop_after:
            self.stop.set()
        if name in self.fail:
            raise RuntimeError(f"GPU said no to {name}")
        folder = self.tmp / "made" / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder


def _fast(project, workers, **kw):
    sleeps = []
    done = J.run_queue(project, workers, pause_s=kw.pop("pause_s", 20), sleep=sleeps.append, **kw)
    return done, sleeps


def test_variations_pauses_and_statuses(tmp_path, monkeypatch):
    # pauses: time.monotonic jumps with each fake sleep
    clock = [0.0]
    monkeypatch.setattr(J.time, "monotonic", lambda: clock[0])
    maker = FakeMaker(tmp_path, fail={"Bad"})
    q = J.Queue(tmp_path)
    q.add("character", "Knight", "a knight in a red cape", {"seed": 10}, variations=3)
    q.add("character", "Bad", "breaks")
    q.add("prop", "Barrel", "an oak barrel")
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock[0] += s
    done = J.run_queue(tmp_path, {"character": maker, "prop": maker}, pause_s=2, sleep=sleep)
    assert [j.name for j in done] == ["Knight", "Bad", "Barrel"]
    assert [c[0] for c in maker.calls] == ["Knight v1", "Knight v2", "Knight v3", "Bad", "Barrel"]
    assert [c[2]["seed"] for c in maker.calls[:3]] == [10, 11, 12]
    assert sum(sleeps) == pytest.approx(2 * 4)              # a rest between each generation
    jobs = J.Queue(tmp_path).jobs
    assert [j.status for j in jobs] == ["done", "failed", "done"]
    assert "GPU said no" in jobs[1].error and jobs[0].finished > 0
    assert [r["name"] for r in jobs[0].results] == ["Knight v1", "Knight v2", "Knight v3"]
    data = json.loads((tmp_path / "queue.json").read_text())
    assert data["jobs"][0]["kind"] == "character" and data["jobs"][0]["variations"] == 3


def test_stop_and_resume_after_restart(tmp_path):
    stop = threading.Event()
    maker = FakeMaker(tmp_path, stop=stop, stop_after=2)
    q = J.Queue(tmp_path)
    q.add("item", "Potion", "a red potion", variations=4)
    _fast(tmp_path, {"item": maker}, pause_s=0, stop_flag=stop)
    job = J.Queue(tmp_path).jobs[0]
    assert job.status == "pending" and job.done_count == 2
    # a crash mid-job: "running" on disk comes back as pending
    data = json.loads((tmp_path / "queue.json").read_text())
    data["jobs"][0]["status"] = "running"
    (tmp_path / "queue.json").write_text(json.dumps(data))
    assert J.Queue(tmp_path).jobs[0].status == "pending"
    maker2 = FakeMaker(tmp_path)
    _fast(tmp_path, {"item": maker2}, pause_s=0)
    assert [c[0] for c in maker2.calls] == ["Potion v3", "Potion v4"]    # v1, v2 not made again
    job = J.Queue(tmp_path).jobs[0]
    assert job.status == "done" and [r["variation"] for r in job.results] == [0, 1, 2, 3]


def test_stop_flag_callable_and_missing_worker(tmp_path):
    q = J.Queue(tmp_path)
    q.add("prop", "Crate")
    assert J.run_queue(tmp_path, {}, pause_s=0, stop_flag=lambda: True) == []
    J.run_queue(tmp_path, {}, pause_s=0)
    job = J.Queue(tmp_path).jobs[0]
    assert job.status == "failed" and "No maker" in job.error
    q = J.Queue(tmp_path)
    q.retry(job.id)
    assert q.jobs[0].status == "pending"
    q.remove(job.id)
    assert J.Queue(tmp_path).jobs == []
    with pytest.raises(ValueError):
        q.add("spaceship", "X")
    with pytest.raises(ValueError):
        q.add("item", "  ")


def test_default_item_worker_without_objects(tmp_path, monkeypatch):
    import sys
    import pixel_pipeline
    monkeypatch.setitem(sys.modules, "pixel_pipeline.objects", None)    # import fails
    monkeypatch.delattr(pixel_pipeline, "objects", raising=False)
    q = J.Queue(tmp_path)
    q.add("item", "Sword", "a sword")
    J.run_queue(tmp_path, {"item": J.default_workers()["item"]}, pause_s=0)
    job = J.Queue(tmp_path).jobs[0]
    assert job.status == "failed" and "objects.py" in job.error


def test_default_item_worker_passes_what_make_item_takes(tmp_path, monkeypatch):
    import sys
    import types
    seen = {}
    fake = types.ModuleType("pixel_pipeline.objects")

    def make_item(name, prompt, project=None, seed=None):
        seen.update(name=name, prompt=prompt, seed=seed, project=project)
        return tmp_path / "items" / name
    fake.make_item = make_item
    monkeypatch.setitem(sys.modules, "pixel_pipeline.objects", fake)
    import pixel_pipeline
    monkeypatch.setattr(pixel_pipeline, "objects", fake, raising=False)
    q = J.Queue(tmp_path)
    q.add("item", "Sword", "a sword", {"seed": 3, "unknown_option": 1})
    J.run_queue(tmp_path, {"item": J.default_workers()["item"]}, pause_s=0)
    assert seen["name"] == "Sword" and seen["seed"] == 3
    assert seen["project"].folder == tmp_path                           # a Project, not a path
    assert J.Queue(tmp_path).jobs[0].status == "done"


def test_default_character_worker_with_fakes(tmp_path, monkeypatch):
    """prompt_to_model + add_character faked: no GPU."""
    from pixel_pipeline import library, model3d
    calls = {}

    def fake_model(prompt, out, subject="person", seed=None, **kw):
        out.write_bytes(b"glb")
        calls["model"] = (prompt, subject, seed)
        return model3d.ModelResult(out, out.with_suffix(".front.png"), None, seed)

    def fake_add(name, model, kind="person", size=None, role="", picture=None, folder=None,
                 prompt=""):
        calls["add"] = (name, kind, size)
        calls["prompt"] = prompt
        return library.Character(name, kind, size or "normal", role, 1.0, tmp_path / "c" / name)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    monkeypatch.setattr(library, "add_character", fake_add)
    q = J.Queue(tmp_path)
    q.add("character", "Wolf", "a grey wolf", {"body": "animal", "size": "large"}, variations=2)
    J.run_queue(tmp_path, pause_s=0)
    assert calls["model"][:2] == ("a grey wolf", "animal") and calls["add"] == ("Wolf v2", "animal", "large")
    assert calls["prompt"] == "a grey wolf"             # the words reach the drawn sprites
    job = J.Queue(tmp_path).jobs[0]
    assert job.status == "done" and job.results[1]["name"] == "Wolf v2"
