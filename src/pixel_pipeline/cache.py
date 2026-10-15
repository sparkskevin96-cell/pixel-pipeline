"""Content-hash caches for deterministic CPU steps, so work that was done once isn't done again
("less waste"): a step's result is stored under a hash of everything it reads (input bytes,
settings, and the source code of the modules that compute it), so a hit returns exactly what
computing it again would, and any change to the inputs or to that code is a miss.

    key = digest(view_bytes, settings.to_dict(), code_digest("pipeline", "pixelize"))
    hit = load("prep", key)            # dict of arrays, or None
    save("prep", key, rgb=..., mask=...)

Where: $PIXEL_CACHE (a folder; "off" or "0" turns caching off), else ~/.pixel-pipeline/cache,
one sub-folder per kind of step. Size: $PIXEL_CACHE_MB (default 2048); past it the files used
longest ago go (a hit marks a file used). Safe with several processes: files are written to a
temporary name and renamed into place.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_MB = 2048
_FILE_DIGESTS: dict = {}
_CODE_DIGESTS: dict = {}
_last_trim = [0.0]


def enabled() -> bool:
    return os.environ.get("PIXEL_CACHE", "").strip().lower() not in ("off", "0", "no", "false")


def root() -> Path:
    env = os.environ.get("PIXEL_CACHE", "").strip()
    if env and env.lower() not in ("off", "0", "no", "false", "on", "1"):
        return Path(env)
    return Path.home() / ".pixel-pipeline" / "cache"


def max_mb() -> float:
    try:
        return float(os.environ.get("PIXEL_CACHE_MB", DEFAULT_MB))
    except ValueError:
        return float(DEFAULT_MB)


def _feed(h, x) -> None:
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(f"nd{a.dtype.str}{a.shape}".encode())
        h.update(a.view(np.uint8).reshape(-1).data if a.size else b"")
    elif isinstance(x, (bytes, bytearray, memoryview)):
        h.update(b"b%d:" % len(x))
        h.update(x)
    elif isinstance(x, Path):
        h.update(b"f" + file_digest(x).encode())
    elif isinstance(x, dict):
        h.update(b"d{")
        for k in sorted(x, key=str):
            _feed(h, str(k))
            _feed(h, x[k])
        h.update(b"}")
    elif isinstance(x, (list, tuple)):
        h.update(b"l[")
        for y in x:
            _feed(h, y)
        h.update(b"]")
    elif isinstance(x, (np.floating, float)):
        h.update(b"r" + float(x).hex().encode())
    elif isinstance(x, (np.integer, int, bool)) or x is None:
        h.update(b"i" + repr(x if x is None or isinstance(x, bool) else int(x)).encode())
    else:
        h.update(b"s" + str(x).encode())


def digest(*parts) -> str:
    """SHA-1 of the parts (arrays by dtype, shape and bytes; files by content; dicts sorted)."""
    h = hashlib.sha1()
    for p in parts:
        _feed(h, p)
    return h.hexdigest()


def file_digest(path) -> str:
    """SHA-1 of a file's bytes (remembered while its size and modification time stay)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    d = _FILE_DIGESTS.get(key)
    if d is None:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        d = _FILE_DIGESTS[key] = h.hexdigest()
    return d


def code_digest(*modules) -> str:
    """SHA-1 of these pixel_pipeline modules' source files ("pipeline", "pixelize", ...): part
    of every key, so a change to the code that computes a result never returns an old one."""
    key = tuple(sorted(modules))
    d = _CODE_DIGESTS.get(key)
    if d is None:
        here = Path(__file__).parent
        h = hashlib.sha1()
        for m in key:
            p = here / (m.replace(".", "/") + ".py")
            h.update(m.encode())
            h.update(p.read_bytes() if p.exists() else _build_salt())
        d = _CODE_DIGESTS[key] = h.hexdigest()
    return d


def _build_salt() -> bytes:
    """No source files (the frozen app): the app's version and build number stand for them."""
    import sys
    from . import __version__
    build = Path(__file__).parent / "ui" / "build.json"
    return (f"{__version__}|{sys.version}|".encode()
            + (build.read_bytes() if build.exists() else b"no-build"))


def source_of(*objs) -> bytes:
    """The source of functions / classes for a key (their compiled code, or the app's build,
    when the source isn't there)."""
    import inspect
    import marshal
    out = []
    for o in objs:
        try:
            out.append(inspect.getsource(o).encode())
        except (OSError, TypeError):
            code = getattr(o, "__code__", None)
            out.append(marshal.dumps(code) if code is not None else _build_salt())
    return b"\x00".join(out)


def _path(kind: str, key: str) -> Path:
    return root() / kind / key[:2] / f"{key}.npz"


def load(kind: str, key: str) -> dict | None:
    """The arrays saved under (kind, key), or None."""
    if not enabled():
        return None
    p = _path(kind, key)
    try:
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError, KeyError):
        return None
    try:
        os.utime(p, None)                        # used now: kept longest
    except OSError:
        pass
    return out


def save(kind: str, key: str, **arrays) -> None:
    """Store arrays under (kind, key) (compressed); silently skipped when that fails."""
    if not enabled():
        return
    p = _path(kind, key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, p)
    except OSError:
        try:
            tmp.unlink()
        except (OSError, UnboundLocalError):
            pass
        return
    if time.monotonic() - _last_trim[0] > 60:
        _last_trim[0] = time.monotonic()
        trim()


def save_json(kind: str, key: str, data) -> None:
    save(kind, key, json=np.frombuffer(json.dumps(data).encode(), np.uint8))


def load_json(kind: str, key: str):
    hit = load(kind, key)
    return None if hit is None or "json" not in hit else json.loads(hit["json"].tobytes())


def size_mb() -> float:
    r = root()
    if not r.exists():
        return 0.0
    return sum(f.stat().st_size for f in r.rglob("*.npz")) / 2 ** 20


def trim(limit_mb: float | None = None) -> int:
    """Delete the least recently used files until the cache is under limit_mb (default
    max_mb()). Returns how many went."""
    r = root()
    if not r.exists():
        return 0
    limit = (max_mb() if limit_mb is None else limit_mb) * 2 ** 20
    files = []
    for f in r.rglob("*"):
        if f.is_file():
            try:
                st = f.stat()
            except OSError:
                continue
            if f.suffix == ".tmp" and time.time() - st.st_mtime > 3600:
                files.append((0.0, st.st_size, f))       # left by a crash: first to go
            elif f.suffix == ".npz":
                files.append((st.st_mtime, st.st_size, f))
    total = sum(s for _, s, _ in files)
    gone = 0
    for _, s, f in sorted(files, key=lambda x: x[0]):
        if total <= limit:
            break
        try:
            f.unlink()
            total -= s
            gone += 1
        except OSError:
            pass
    return gone


def clear(kind: str | None = None) -> None:
    """Remove the whole cache (or one kind of step)."""
    import shutil
    shutil.rmtree(root() / kind if kind else root(), ignore_errors=True)
