"""Jobs for Claude (MCP / CLI): long GPU work runs in the background, one GPU job at a time.

    m = manager()
    job = m.start("make", {"plan": plan.to_dict()}, project=proj, gpu=True, title="Knight")
    m.status(job.id)      # percent, stage, eta, text, result when done
    m.cancel(job.id)      # Stop: what was finished is kept (drawn keys, variations)
    m.wait(job.id, 60)

Records live with their project: <project>/agent_jobs/<id>.json (written at every stage change
and every few seconds while running), so a crash or a restart loses nothing but the run itself:
a job whose process is gone shows as "interrupted" and resume() queues it again (drawn
characters carry on from their kept drawings; other makers start that job over). Nothing restarts
on its own: Kevin's rule is no unattended heavy GPU runs.

GPU jobs (gpu=True) go through ONE worker thread in order, each holding the cross-process GPU lock
(PIXEL_GPU_LOCK, default <PIXEL_HOME>/gpu.lock; the same O_EXCL file protocol as
tools/gpu_lock.py, so set PIXEL_GPU_LOCK=C:\\AI\\gpu.lock to share it with other scripts), with a
rest of PIXEL_AGENT_PAUSE seconds (default 20) between GPU jobs (heat pacing, jobqueue._rest).
CPU jobs (UI kits, effects, maps, exports...) start at once on their own thread.

An op is a function op(args: dict, tracker: progress.Tracker) -> dict (JSON-able result);
agent.py registers them with register(). The tracker is the job's progress callback and Stop.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import progress as P

STATUSES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
ACTIVE = ("queued", "running")
FOLDER = "agent_jobs"

OPS: dict[str, Callable] = {}          # name -> op(args, tracker) -> dict
PLANS: dict[str, Callable] = {}        # name -> (args) -> (stages | None, estimate s | None)


def register(name: str, plan: Callable | None = None):
    """Decorator: an op the job manager can run (plan(args) -> (stages, estimate seconds))."""
    def deco(fn):
        OPS[name] = fn
        if plan is not None:
            PLANS[name] = plan
        return fn
    return deco


@dataclass
class JobRecord:
    id: str
    op: str
    args: dict = field(default_factory=dict)
    title: str = ""
    project: str = ""
    project_folder: str = ""
    gpu: bool = False
    status: str = "queued"
    created: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    percent: int | None = None
    stage: str = ""
    text: str = ""
    eta_s: float | None = None
    estimate_min: float = 0.0          # the plan's GPU+CPU minutes (makeany.GPU table)
    result: dict = field(default_factory=dict)
    error: str = ""
    log: list = field(default_factory=list)       # the last messages said
    attempts: int = 0
    pid: int = 0
    gpu_wait_s: float = 0.0            # waiting for the GPU (another job or process had it)

    @property
    def path(self) -> Path:
        return Path(self.project_folder) / FOLDER / f"{self.id}.json"

    def public(self) -> dict:
        """What Claude sees (compact)."""
        d = {"job": self.id, "op": self.op, "title": self.title, "project": self.project,
             "status": self.status, "gpu": self.gpu}
        if self.status in ACTIVE or self.percent is not None:
            d["percent"] = self.percent
        if self.stage:
            d["stage"] = self.stage
        if self.text:
            d["doing"] = self.text
        if self.status == "running" and self.eta_s is not None:
            d["eta_s"] = round(self.eta_s)
            d["eta"] = P.eta_text(self.eta_s)
        if self.estimate_min:
            d["estimate_min"] = round(self.estimate_min, 1)
        if self.started:
            end = self.finished or time.time()
            d["minutes"] = round((end - self.started - self.gpu_wait_s) / 60, 2)
        if self.gpu_wait_s > 1:
            d["waited_for_gpu_s"] = round(self.gpu_wait_s)
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        if self.status == "interrupted":
            d["hint"] = "the process running it stopped; resume_jobs queues it again"
        return d


# --------------------------------------------------------------------------- process checks

def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- GPU lock (cross-process)

def lock_path() -> Path:
    env = os.environ.get("PIXEL_GPU_LOCK")
    if env:
        return Path(env)
    from .assets import home
    return home() / "gpu.lock"


def lock_holder() -> dict | None:
    """Who holds the GPU lock ({"pid", "what", "t"}) or None."""
    p = lock_path()
    try:
        info = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not _pid_alive(int(info.get("pid", 0))) or time.time() - info.get("t", 0) > 7200:
        return None
    return info


@contextmanager
def gpu_lock(what: str = "", poll: float = 2.0, stop: Callable[[], bool] | None = None,
             waiting: Callable[[dict], None] | None = None):
    """One GPU job at a time across processes (stale locks: owner gone or > 2 h old are taken
    over). stop() -> True raises progress.Stopped while waiting; waiting(info) is told who has it."""
    lock = lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    told = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "what": what, "t": time.time()}).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                info = json.loads(lock.read_text())
                if not _pid_alive(int(info.get("pid", 0))) or time.time() - info.get("t", 0) > 7200:
                    lock.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                info = {}
            if waiting is not None and not told:
                waiting(info)
                told = True
            end = time.monotonic() + poll
            while time.monotonic() < end:
                if stop is not None and stop():
                    raise P.Stopped("Cancelled while waiting for the GPU")
                time.sleep(min(0.1, poll))
    try:
        yield
    finally:
        try:
            info = json.loads(lock.read_text())
            if int(info.get("pid", 0)) == os.getpid():
                lock.unlink(missing_ok=True)
        except (OSError, ValueError):
            lock.unlink(missing_ok=True)


# --------------------------------------------------------------------------- the manager

class Manager:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict[str, JobRecord] = {}       # every record seen (this process + loaded)
        self.trackers: dict[str, P.Tracker] = {}
        self.queue: list[str] = []                  # GPU jobs waiting, in order
        self.done_event: dict[str, threading.Event] = {}
        self.worker: threading.Thread | None = None
        self.wake = threading.Condition(self.lock)
        self.client = None                          # a ComfyClient for every job (tests: the fake)
        self.pause_s = float(os.environ.get("PIXEL_AGENT_PAUSE", "20"))
        self.last_gpu_end = 0.0
        self.extras: dict[str, object] = {}         # job id -> in-memory things (e.g. makeany.Made)

    # ---- records
    def _save(self, rec: JobRecord) -> None:
        if not rec.project_folder:
            return
        p = rec.path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        data = {k: getattr(rec, k) for k in JobRecord.__dataclass_fields__}   # args stay JSON-ish
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, p)

    def _load_folder(self, folder: Path) -> None:
        for f in sorted((folder / FOLDER).glob("*.json")) if (folder / FOLDER).exists() else []:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                rec = JobRecord(**{k: v for k, v in d.items() if k in JobRecord.__dataclass_fields__})
            except (OSError, ValueError, TypeError):
                continue
            with self.lock:
                mine = rec.id in self.trackers or rec.id in self.queue
                if rec.id in self.jobs and (mine or self.jobs[rec.id].pid == os.getpid()):
                    continue                              # this process's own: memory is newer
                if rec.status in ACTIVE and not _pid_alive(rec.pid):
                    rec.status = "interrupted"
                    rec.text = "interrupted: the process running it stopped"
                    self._save(rec)
                self.jobs[rec.id] = rec

    def refresh(self) -> None:
        """Read every project's job records (other processes' jobs, earlier sessions)."""
        from .assets import projects_folder
        base = projects_folder()
        if base.exists():
            for proj in base.iterdir():
                if proj.is_dir():
                    self._load_folder(proj)

    def get(self, job_id: str) -> JobRecord:
        with self.lock:
            rec = self.jobs.get(job_id)
        if rec is None:
            self.refresh()
            rec = self.jobs.get(job_id)
        if rec is None:
            raise ValueError(f"No job {job_id!r} (list_jobs shows them)")
        if rec.pid != os.getpid() and rec.status in ACTIVE:      # another process's: re-read
            try:
                d = json.loads(rec.path.read_text(encoding="utf-8"))
                rec = JobRecord(**{k: v for k, v in d.items() if k in JobRecord.__dataclass_fields__})
                if rec.status in ACTIVE and not _pid_alive(rec.pid):
                    rec.status = "interrupted"
                    self._save(rec)
                self.jobs[job_id] = rec
            except (OSError, ValueError, TypeError):
                pass
        return rec

    def list(self, status: str | None = None, project: str | None = None, limit: int = 20) -> list[JobRecord]:
        self.refresh()
        with self.lock:
            recs = list(self.jobs.values())
        if status:
            recs = [r for r in recs if r.status == status]
        if project:
            recs = [r for r in recs if r.project == project]
        recs.sort(key=lambda r: (r.status not in ACTIVE, -r.created))
        return recs[:limit] if limit else recs

    # ---- starting
    def start(self, op: str, args: dict | None = None, *, project=None, gpu: bool = False,
              title: str = "", estimate_min: float = 0.0, wait_s: float = 0.0) -> JobRecord:
        if op not in OPS:
            raise ValueError(f"Unknown job op {op!r}")
        from .assets import current_project
        proj = project or current_project()
        rec = JobRecord(uuid.uuid4().hex[:10], op, dict(args or {}), title or op, proj.name,
                        str(proj.folder), bool(gpu), "queued", time.time(),
                        estimate_min=float(estimate_min or 0.0), pid=os.getpid())
        rec.args["_job"] = rec.id                  # ops that keep things per job (makeany.Made)
        self._submit(rec)
        if wait_s:
            self.wait(rec.id, wait_s)
        return self.get(rec.id)

    def _submit(self, rec: JobRecord) -> None:
        with self.lock:
            rec.pid = os.getpid()
            rec.status = "queued"
            rec.error = ""
            self.jobs[rec.id] = rec
            self.done_event[rec.id] = threading.Event()
            self._save(rec)
            if rec.gpu:
                self.queue.append(rec.id)
                self._ensure_worker()
                self.wake.notify_all()
            else:
                threading.Thread(target=self._run, args=(rec,), daemon=True,
                                 name=f"agent-job-{rec.id}").start()

    def resume(self, ids=None) -> list[JobRecord]:
        """Queue interrupted (or failed / cancelled, when named) jobs again."""
        self.refresh()
        out = []
        for rec in list(self.jobs.values()):
            named = ids is not None and rec.id in ids
            if (rec.status == "interrupted" and (ids is None or named)) or \
                    (named and rec.status in ("failed", "cancelled")):
                rec.attempts += 0          # counted when it runs
                rec.log.append(f"resumed {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self._submit(rec)
                out.append(rec)
        return out

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._gpu_loop, daemon=True, name="agent-gpu-worker")
            self.worker.start()

    def _gpu_loop(self) -> None:
        while True:
            with self.lock:
                while not self.queue:
                    if not self.wake.wait(timeout=30):
                        if not self.queue:
                            self.worker = None
                            return
                job_id = self.queue[0]
                rec = self.jobs[job_id]
            if rec.status == "queued":
                self._run(rec)
            with self.lock:
                if self.queue and self.queue[0] == job_id:
                    self.queue.pop(0)

    # ---- running
    def _run(self, rec: JobRecord) -> None:
        if rec.status != "queued":
            self._finish_event(rec)
            return
        last_save = [0.0]

        def on_update(u: P.Update) -> None:
            with self.lock:
                rec.percent = u.percent()
                rec.stage = u.stage or rec.stage
                if u.text and u.text != rec.text:
                    rec.text = u.text
                    rec.log = (rec.log + [u.text])[-30:]
                rec.eta_s = u.eta
                now = time.monotonic()
                if now - last_save[0] > 2.0:
                    last_save[0] = now
                    self._save(rec)

        stages, estimate = (None, None)
        plan_fn = PLANS.get(rec.op)
        if plan_fn is not None:
            try:
                stages, estimate = plan_fn(rec.args)
            except Exception:  # noqa: BLE001 - a bar without a plan still works
                stages, estimate = None, None
        if estimate is None and not stages and rec.estimate_min:
            estimate = rec.estimate_min * 60
        tracker = P.Tracker(stages, estimate=estimate, on_update=on_update, title=rec.title,
                            min_interval=0.5)
        with self.lock:
            self.trackers[rec.id] = tracker
            rec.status, rec.started, rec.finished = "running", time.time(), 0.0
            rec.attempts += 1
            rec.result, rec.error = {}, ""
            rec.percent = 0
            self._save(rec)
        from .assets import using_project, Project
        try:
            with using_project(Project(rec.project, Path(rec.project_folder))):
                if rec.gpu:
                    self._rest(tracker)
                    t_wait = time.monotonic()

                    def waiting(info):
                        tracker.say(f"Waiting for the GPU ({info.get('what') or 'another job'} "
                                    f"has it)…")
                    with gpu_lock(f"{rec.title} ({rec.id})", stop=lambda: tracker.stopped,
                                  waiting=waiting):
                        rec.gpu_wait_s = time.monotonic() - t_wait
                        with P.using(tracker):
                            result = OPS[rec.op](rec.args, tracker)
                    self.last_gpu_end = time.monotonic()
                else:
                    with P.using(tracker):
                        result = OPS[rec.op](rec.args, tracker)
            tracker.finish()
            with self.lock:
                rec.status, rec.result = "done", result or {}
                rec.percent, rec.eta_s = 100, 0.0
                rec.text = "done"
        except P.Stopped as e:
            with self.lock:
                rec.status, rec.error = "cancelled", str(e)
        except Exception as e:  # noqa: BLE001 - recorded on the job
            with self.lock:
                rec.status = "failed"
                rec.error = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
                rec.log = (rec.log + traceback.format_exc().splitlines()[-6:])[-30:]
        finally:
            if rec.gpu:
                self.last_gpu_end = time.monotonic()
            with self.lock:
                rec.finished = time.time()
                self.trackers.pop(rec.id, None)
                self._save(rec)
            self._finish_event(rec)

    def _rest(self, tracker: P.Tracker) -> None:
        """Heat pacing: pause_s between GPU jobs (not before the first one)."""
        if not self.last_gpu_end or self.pause_s <= 0:
            return
        left = self.pause_s - (time.monotonic() - self.last_gpu_end)
        if left > 0.5:
            tracker.say(f"Resting the GPU {left:.0f} s between jobs (heat pacing)…")
            from .jobqueue import _rest
            if not _rest(left, lambda: tracker.stopped, time.sleep):
                raise P.Stopped("Cancelled while resting")

    def _finish_event(self, rec: JobRecord) -> None:
        ev = self.done_event.get(rec.id)
        if ev is not None:
            ev.set()

    # ---- waiting / stopping
    def wait(self, job_id: str, timeout: float) -> JobRecord:
        ev = self.done_event.get(job_id)
        if ev is not None:
            ev.wait(max(0.0, float(timeout)))
        else:                                    # another process's job: poll its record
            end = time.monotonic() + float(timeout)
            while time.monotonic() < end and self.get(job_id).status in ACTIVE:
                time.sleep(0.5)
        return self.get(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        rec = self.get(job_id)
        with self.lock:
            if rec.status == "queued" and rec.id in self.queue and self.queue[0] != rec.id:
                self.queue.remove(rec.id)
                rec.status, rec.finished = "cancelled", time.time()
                rec.text = "cancelled before it started"
                self._save(rec)
                self._finish_event(rec)
                return rec
            if rec.status == "interrupted":
                rec.status, rec.finished = "cancelled", time.time()
                self._save(rec)
                return rec
            t = self.trackers.get(rec.id)
        if t is not None:
            t.stop()               # cooperative: at the job's next message / GPU step
            rec.text = "stopping (at the next step; what is finished is kept)…"
        elif rec.status in ACTIVE and rec.pid != os.getpid():
            raise ValueError(f"Job {job_id} runs in another process (pid {rec.pid}); stop it there.")
        elif rec.status == "queued":           # first in line but not started yet
            with self.lock:
                if rec.id in self.queue:
                    self.queue.remove(rec.id)
                rec.status, rec.finished = "cancelled", time.time()
                self._save(rec)
                self._finish_event(rec)
        return rec

    def position(self, job_id: str) -> int | None:
        """0 = running/next, 1 = one job ahead..."""
        with self.lock:
            return self.queue.index(job_id) if job_id in self.queue else None


_MANAGER: Manager | None = None
_MLOCK = threading.Lock()


def manager() -> Manager:
    global _MANAGER
    with _MLOCK:
        if _MANAGER is None:
            _MANAGER = Manager()
        return _MANAGER


def reset_manager() -> None:
    """Tests: forget this process's manager (records on disk stay)."""
    global _MANAGER
    with _MLOCK:
        _MANAGER = None
