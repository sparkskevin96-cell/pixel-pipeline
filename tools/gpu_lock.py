r"""One GPU job at a time across processes/agents: `with gpu_lock("what"):` waits for C:\AI\gpu.lock.
Stale locks (owner pid gone, or older than 2 h) are taken over."""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(os.environ.get("PIXEL_GPU_LOCK", r"C:\AI\gpu.lock"))


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True


@contextmanager
def gpu_lock(what: str = "", poll: float = 10.0):
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "what": what, "t": time.time()}).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                info = json.loads(LOCK.read_text())
                if not _alive(int(info.get("pid", 0))) or time.time() - info.get("t", 0) > 7200:
                    LOCK.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            time.sleep(poll)
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)
