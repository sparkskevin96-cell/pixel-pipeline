"""Creation queue: line up characters, items and props to make, then let the laptop work
through them one at a time with a rest between each (no hours of full-tilt GPU).

    q = Queue(project)                      # <project folder>/queue.json
    q.add("character", "Knight", "a knight in a red cape", variations=3)
    run_queue(project, stop_flag=event)     # Knight v1, Knight v2, Knight v3 ...

Jobs survive a restart: one that was "running" when the app closed is "pending" again on load,
and variations already made are not made again. Worker functions are injected
(`worker_fns={kind: fn(name, prompt, options, project) -> asset}`); default_workers() has the
real ones (GPU), loaded lazily.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("character", "item", "prop", "make")   # make = makeany.Plan in options["plan"]
STATUSES = ("pending", "running", "done", "failed", "cancelled")


@dataclass
class Job:
    kind: str
    name: str
    prompt: str = ""
    options: dict = field(default_factory=dict)
    variations: int = 1
    status: str = "pending"
    error: str = ""
    created: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    results: list = field(default_factory=list)      # [{"variation": i, "name", "folder"}]
    id: str = ""

    def variation_names(self) -> list[str]:
        n = max(1, int(self.variations))
        return [self.name] if n == 1 else [f"{self.name} v{i + 1}" for i in range(n)]

    @property
    def done_count(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        n = max(1, int(self.variations))
        what = f"{self.kind}: {self.name}" + (f" x{n}" if n > 1 else "")
        counted = n > 1 and (self.status != "pending" or self.results)
        state = self.status + (f" ({self.done_count}/{n})" if counted else "")
        return f"{what} — {state}" + (f": {self.error.splitlines()[0]}" if self.error else "")


def _folder(project) -> Path:
    return Path(getattr(project, "folder", project))


def _as_project(project):
    """An assets.Project (a plain folder becomes one)."""
    if hasattr(project, "style"):
        return project
    from .assets import Project
    folder = Path(project)
    return Project(folder.name, folder)


class Queue:
    """The project's queue.json. Every change is saved straight away."""

    def __init__(self, project):
        self.project = project
        self.path = _folder(project) / "queue.json"
        self.lock = threading.RLock()
        self.jobs: list[Job] = []
        self.load()

    def load(self) -> list[Job]:
        with self.lock:
            self.jobs = []
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            for d in data.get("jobs", []):
                known = {k: v for k, v in d.items() if k in Job.__dataclass_fields__}
                try:
                    job = Job(**known)
                except TypeError:
                    continue
                if job.status == "running":              # the app closed mid-job
                    job.status = "pending"
                self.jobs.append(job)
            return self.jobs

    def save(self) -> Path:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"jobs": [asdict(j) for j in self.jobs]}, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path)
            return self.path

    def add(self, kind: str, name: str, prompt: str = "", options: dict | None = None,
            variations: int = 1) -> Job:
        if kind not in KINDS:
            raise ValueError(f"Unknown job kind '{kind}' (use {', '.join(KINDS)})")
        if not str(name).strip():
            raise ValueError("Give the job a name.")
        job = Job(kind, str(name).strip(), prompt, dict(options or {}), max(1, int(variations)),
                  created=time.time(), id=uuid.uuid4().hex[:12])
        with self.lock:
            self.jobs.append(job)
            self.save()
        return job

    def get(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def remove(self, job_id: str) -> None:
        with self.lock:
            self.jobs = [j for j in self.jobs if j.id != job_id or j.status == "running"]
            self.save()

    def retry(self, job_id: str) -> None:
        with self.lock:
            j = self.get(job_id)
            if j and j.status in ("failed", "cancelled"):
                j.status, j.error = "pending", ""
                self.save()

    def clear_finished(self) -> None:
        with self.lock:
            self.jobs = [j for j in self.jobs if j.status not in ("done", "cancelled")]
            self.save()

    def next_pending(self) -> Job | None:
        return next((j for j in self.jobs if j.status == "pending"), None)

    def pending(self) -> list[Job]:
        return [j for j in self.jobs if j.status == "pending"]


# --------------------------------------------------------------------------- workers

def _call(fn, *args, **kwargs):
    """fn(*args, **only the kwargs it takes)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _character_worker(name: str, prompt: str, options: dict, project):
    """Prompt -> 3D model (local GPU via ComfyUI) -> rigged, animated library character.
    options["review"] (default True): drawn characters stop after the front drawing and wait in
    the library for the user's OK (library.add_character(preview=True)); False = all at once."""
    import tempfile
    from . import library
    from .model3d import prompt_to_model
    opts = dict(options)
    kind = opts.pop("body", opts.pop("kind", "person"))
    size = opts.pop("size", None)
    role = opts.pop("role", "")
    seed = opts.pop("seed", None)
    review = bool(opts.pop("review", True))
    folder = None if os.environ.get("PIXEL_LIBRARY") else _folder(project) / "characters"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.glb"
        subject = "animal" if kind == "animal" else "person"
        res = prompt_to_model(prompt or name, out, subject=subject, seed=seed, **opts)
        model = Path(getattr(res, "model", None) or out)
        picture = getattr(res, "picture", None)
        return _call(library.add_character, name, model, kind=kind, size=size, role=role,
                     picture=picture if picture and Path(picture).exists() else None,
                     folder=folder, prompt=prompt or name, preview=review)


def _object_worker(kind: str):
    def work(name: str, prompt: str, options: dict, project):
        try:
            from . import objects
        except ImportError as e:
            raise RuntimeError(f"Making {kind}s needs objects.py, which isn't in this build yet "
                               f"({e}).") from None
        fn = getattr(objects, f"make_{kind}", None)
        if fn is None:
            raise RuntimeError(f"objects.py has no make_{kind}().")
        return _call(fn, name, prompt, project=_as_project(project), **options)
    work.__name__ = f"make_{kind}"
    return work


def _make_worker(name: str, prompt: str, options: dict, project):
    """A "Make anything" plan (makeany.py): any kind of asset from words."""
    from .makeany import queue_worker
    return queue_worker(name, prompt, options, _as_project(project))


def default_workers() -> dict:
    """The real makers: character = model3d.prompt_to_model + library.add_character (GPU);
    item / prop = objects.make_item / make_prop (a clear error if objects.py isn't there);
    make = makeany.queue_worker (whatever the plan says)."""
    return {"character": _character_worker, "item": _object_worker("item"),
            "prop": _object_worker("prop"), "make": _make_worker}


def _result_info(result, name: str) -> dict:
    folder = getattr(result, "folder", None)
    if folder is None and isinstance(result, (str, Path)):
        folder = result
    info = {"name": getattr(result, "name", name), "folder": str(folder) if folder else ""}
    if getattr(result, "waiting", False):     # a drawn character's front, for the user's OK
        info["waiting"] = True
    notice = getattr(result, "notice", "")
    if notice:                       # e.g. drawn sprites failed: the 3D render, drawings kept
        info["notice"] = str(notice)
    return info


# --------------------------------------------------------------------------- running

def _stopped(stop_flag) -> bool:
    if stop_flag is None:
        return False
    if hasattr(stop_flag, "is_set"):
        return stop_flag.is_set()
    return bool(stop_flag())


def _rest(seconds: float, stop_flag, sleep) -> bool:
    """Wait `seconds` (in small steps so Stop is quick). False if stopped meanwhile."""
    end = time.monotonic() + seconds
    while True:
        if _stopped(stop_flag):
            return False
        left = end - time.monotonic()
        if left <= 0:
            return True
        sleep(min(0.25, left))


def run_queue(project, worker_fns: dict | None = None, pause_s: float = 20.0, stop_flag=None,
              *, on_update=None, queue: Queue | None = None, sleep=time.sleep,
              job_ids=None) -> list[Job]:
    """Work through the pending jobs one generation at a time, resting pause_s seconds between
    generations (each variation is one generation). Stops early when stop_flag (a
    threading.Event or a callable) is set: the current generation finishes, the job goes back to
    pending with what it made so far. on_update(job) after every change. Returns the jobs worked
    on. job_ids: only these jobs (the rest of the queue waits)."""
    q = queue or Queue(project)
    workers = default_workers() if worker_fns is None else worker_fns
    note = on_update or (lambda job: None)
    worked: list[Job] = []
    first = True
    while not _stopped(stop_flag):
        with q.lock:
            job = next((j for j in q.jobs if j.status == "pending"
                        and (job_ids is None or j.id in job_ids)), None)
            if job is None:
                break
            job.status, job.error, job.started = "running", "", time.time()
            q.save()
        worked.append(job)
        note(job)
        fn = workers.get(job.kind)
        if fn is None:
            with q.lock:
                job.status, job.error, job.finished = "failed", f"No maker for {job.kind}s.", time.time()
                q.save()
            note(job)
            continue
        made = {r.get("variation") for r in job.results}
        stopped = False
        for i, vname in enumerate(job.variation_names()):
            if i in made:
                continue
            if not first and not _rest(pause_s, stop_flag, sleep):
                stopped = True
                break
            first = False
            opts = dict(job.options)
            if "seed" in opts and opts["seed"] is not None:
                opts["seed"] = int(opts["seed"]) + i
            elif job.variations > 1:
                opts["seed"] = int(job.created * 1000) % 1_000_000 + i
            try:
                result = fn(vname, job.prompt, opts, project)
            except Exception as e:  # noqa: BLE001 - recorded on the job
                from .generate import ComfyUIGaveUp
                if isinstance(e, ComfyUIGaveUp):
                    # the GPU is gone (driver reset / crash it didn't recover from): the rest of
                    # the batch would fail the same way. This job waits as pending (what it made
                    # is kept: running the queue again carries on), the run stops here.
                    with q.lock:
                        job.status, job.error = "pending", str(e)
                        q.save()
                    note(job)
                    return worked
                with q.lock:
                    job.status = "failed"
                    job.error = str(e) or type(e).__name__
                    job.finished = time.time()
                    q.save()
                note(job)
                break
            with q.lock:
                job.results.append({"variation": i, **_result_info(result, vname)})
                q.save()
            note(job)
        else:
            with q.lock:
                job.status, job.finished = "done", time.time()
                q.save()
            note(job)
            continue
        if stopped:
            with q.lock:
                job.status = "pending"
                q.save()
            note(job)
            break
    return worked
