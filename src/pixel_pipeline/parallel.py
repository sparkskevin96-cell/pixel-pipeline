"""CPU work spread over worker processes, sized for the PC (a hybrid CPU and RAM that other
programs share), with the same results as doing it in one process.

    run(fn, tasks, init=..., init_args=..., per_worker_mb=...) -> [fn(task) for task in tasks]

Workers are spawned processes (numpy work holds the GIL too often for threads to help);
`init(*init_args)` runs once in each worker (load what every task shares), and fn(task) is
called there. Every task is computed by exactly the code the one-process path runs, so the
output is bit-for-bit the same; only the order work happens in changes.

How many: PIXEL_WORKERS (0 or 1 = no workers, in this process; n = at most n), else the
logical CPUs minus a few left for the app and the PC (a Core Ultra 7 155H's 22 logical CPUs
-> up to 8 workers), and never more than the free RAM allows (per_worker_mb each, keeping
`reserve_mb` free for everything else). Work too small to pay for starting workers, or a
pool that can't start (frozen app without freeze_support, out of memory), runs in this
process instead.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Iterable

MAX_WORKERS = 8               # more spawned interpreters cost RAM for little extra speed
RESERVE_MB = 1500             # left for the app, the GPU job next door and the OS


def free_ram_mb() -> float:
    """Memory the OS can hand out now (MB): available physical memory."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys / 2 ** 20
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024
        else:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except (OSError, ValueError, AttributeError):
        pass
    return 4096.0


def worker_count(tasks: int, per_worker_mb: float = 300.0, reserve_mb: float = RESERVE_MB,
                 max_workers: int | None = None) -> int:
    """Workers to use for `tasks` independent pieces (1 = none, work in this process)."""
    env = os.environ.get("PIXEL_WORKERS", "").strip()
    cap = MAX_WORKERS if max_workers is None else max_workers
    if env:
        try:
            cap = int(env)
        except ValueError:
            pass
        if cap <= 1:
            return 1
    cpus = os.cpu_count() or 1
    by_cpu = max(1, min(cpus - 2, cpus * 3 // 8 + 1))  # 22 -> 9 (P-cores' share), 8 -> 4
    by_ram = int(max(0.0, free_ram_mb() - reserve_mb) // max(per_worker_mb, 1.0))
    return max(1, min(tasks, cap, by_cpu, by_ram))


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _call_init(init, args):
    try:                                  # OpenCV's own threads would fight the other workers
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:                                  # below the app: the UI stays responsive
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(5)
    except (OSError, AttributeError):
        pass
    if init is not None:
        init(*args)


def run(fn: Callable, tasks: Iterable, *, init: Callable | None = None, init_args=(),
        per_worker_mb: float = 300.0, workers: int | None = None, on_done=None,
        cancel=None) -> list:
    """[fn(task) for task in tasks], on worker processes when that is worth it.
    on_done(i, result): called (in this process) as each task finishes. cancel(): True = stop
    (pending tasks are dropped; raises Cancel). init(*init_args) prepares each worker (and
    this process when the work runs here)."""
    tasks = list(tasks)
    n = workers if workers is not None else worker_count(len(tasks), per_worker_mb)
    pool = Pool(min(n, len(tasks)), init=init, init_args=init_args)
    try:
        return pool.map(fn, tasks, on_done=on_done, cancel=cancel)
    finally:
        pool.close()


class Cancel(Exception):
    """The work was cancelled."""


class Pool:
    """Worker processes kept for several rounds of map() (each worker runs init once).
    workers <= 1, or a pool that can't start, works in this process instead (init runs
    here then)."""

    def __init__(self, workers: int, init: Callable | None = None, init_args=()):
        self.init, self.init_args = init, tuple(init_args)
        self._pool = None
        self._here_ready = False
        self._saved = None
        if workers > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._saved = {v: os.environ.get(v) for v in _THREAD_VARS}
            for v in _THREAD_VARS:            # workers are spawned with these: one BLAS
                os.environ[v] = "1"           # thread each (n workers already fill the CPUs)
            try:
                self._pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                                 initializer=_call_init,
                                                 initargs=(init, self.init_args))
            except (OSError, ValueError, RuntimeError):
                self._restore()
                self._pool = None
        self.workers = workers if self._pool is not None else 1

    def _restore(self):
        if self._saved is not None:
            for v, old in self._saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
            self._saved = None

    def _here(self, fn, items, on_done, cancel):
        if not self._here_ready:
            if self.init is not None:
                self.init(*self.init_args)
            self._here_ready = True
        out = []
        for i, t in enumerate(items):
            if cancel is not None and cancel():
                raise Cancel()
            r = fn(t)
            out.append(r)
            if on_done is not None:
                on_done(i, r)
        return out

    def map(self, fn: Callable, items, *, on_done=None, cancel=None) -> list:
        items = list(items)
        if self._pool is None or len(items) == 0:
            return self._here(fn, items, on_done, cancel)
        from concurrent.futures import FIRST_COMPLETED, wait
        from concurrent.futures.process import BrokenProcessPool
        out: list = [None] * len(items)
        got = 0
        try:
            futs = {self._pool.submit(fn, t): i for i, t in enumerate(items)}
            pending = set(futs)
            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    i = futs[f]
                    out[i] = f.result()
                    got += 1
                    if on_done is not None:
                        on_done(i, out[i])
                if cancel is not None and cancel():
                    for f in pending:
                        f.cancel()
                    raise Cancel()
        except BrokenProcessPool:
            # a worker died (e.g. out of memory): do the rest here, results stay the same
            self.close()
            rest = [i for i in range(len(items)) if out[i] is None]
            for j, r in zip(rest, self._here(fn, [items[i] for i in rest], None, cancel)):
                out[j] = r
                if on_done is not None:
                    on_done(j, r)
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._restore()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
