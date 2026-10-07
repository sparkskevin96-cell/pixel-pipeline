"""Monocular depth: Depth Anything V2 **Small** (Apache-2.0) as ONNX, run with onnxruntime.

Only the Small model is used on purpose: the Base/Large checkpoints are CC-BY-NC-4.0
(non-commercial), Small is Apache-2.0 and fine for games you sell.

The model downloads on first use (~95 MB) to ~/.cache/pixel-pipeline/models/
($PIXEL_PIPELINE_DEPTH_MODEL can point at any compatible .onnx instead). Sources, in order:

  1. GitHub release of fabio-sim/Depth-Anything-ONNX (Apache-2.0 export, dynamic H x W)
  2. Hugging Face onnx-community/depth-anything-v2-small (transformers.js export)
  3. the same GitHub release's fixed 518 x 518 export

onnxruntime comes with the cutout extra (pip install "pixel-pipeline[cutout]"), or install
it alone with `pip install onnxruntime` (`onnxruntime-gpu` for CUDA).
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cutout import CutoutUnavailable


class DepthUnavailable(CutoutUnavailable):
    """onnxruntime missing or the depth model can't be fetched (a CutoutUnavailable, so
    callers that already handle missing optional AI pieces handle this too)."""


INSTALL_HINT = ('Depth estimation needs onnxruntime: pip install onnxruntime '
                '(or pip install "pixel-pipeline[cutout]"; onnxruntime-gpu for an NVIDIA GPU)')
CACHE_DIR = Path.home() / ".cache" / "pixel-pipeline" / "models"
ENV_MODEL = "PIXEL_PIPELINE_DEPTH_MODEL"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14  # ViT patch size: network input sides must be multiples of 14


@dataclass(frozen=True)
class ModelSource:
    file: str          # cache file name
    url: str
    sha256: str | None
    license: str


_GH = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"
_HF = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/"
SOURCES: tuple[ModelSource, ...] = (
    ModelSource("depth_anything_v2_vits_dynamic.onnx", _GH + "depth_anything_v2_vits_dynamic.onnx",
                "46c4e8eeda3a27f34701831b6a2ec7753d7b38779b215acb5633424703deed8f", "Apache-2.0"),
    ModelSource("depth_anything_v2_small_hf.onnx", _HF + "onnx/model.onnx", None, "Apache-2.0"),
    ModelSource("depth_anything_v2_vits_518.onnx", _GH + "depth_anything_v2_vits.onnx",
                "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be", "Apache-2.0"),
)

_sessions: dict = {}
_lock = threading.Lock()


def _log(msg: str, end: str = "\n") -> None:
    if sys.stderr is not None:  # None in a windowed build
        with contextlib.suppress(Exception):
            sys.stderr.write(msg + end)
            sys.stderr.flush()


def _download(src: ModelSource, dest: Path) -> None:
    """Stream to a temp file next to `dest`, verify sha256, then atomically rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    digest = hashlib.sha256()
    tty = sys.stderr is not None and sys.stderr.isatty()
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(src.url, timeout=60) as r:
            total, done, shown = int(r.headers.get("Content-Length") or 0), 0, -1
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                pct = done * 100 // total if total else -1
                if tty and pct != shown:
                    shown = pct
                    _log(f"\r  {done >> 20} / {total >> 20} MB ({pct}%)", end="")
        if tty:
            _log("")
        if done < 1 << 20:
            raise OSError(f"download too small ({done} bytes)")
        if src.sha256 and digest.hexdigest() != src.sha256:
            raise OSError(f"checksum mismatch (sha256 {digest.hexdigest()})")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def model_file() -> Path:
    """Local path of the depth model, downloading it on first use."""
    override = os.getenv(ENV_MODEL)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise DepthUnavailable(f"${ENV_MODEL} points at {path}, which doesn't exist")
        return path
    for src in SOURCES:
        if (CACHE_DIR / src.file).exists():
            return CACHE_DIR / src.file
    errors = []
    for src in SOURCES:
        dest = CACHE_DIR / src.file
        _log(f"Downloading depth model Depth Anything V2 Small (~95 MB, {src.license}) "
             f"from {src.url.split('/')[2]} to {CACHE_DIR} ...")
        try:
            _download(src, dest)
            return dest
        except OSError as e:  # URLError, timeouts, bad checksum
            errors.append(f"  {src.url}: {e}")
    raise DepthUnavailable(
        "Couldn't download the depth model:\n" + "\n".join(errors) +
        f"\nDownload one of those files by hand and save it as {CACHE_DIR / SOURCES[0].file} "
        f"(or set ${ENV_MODEL} to its path).")


def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as e:
        raise DepthUnavailable(INSTALL_HINT) from e
    return onnxruntime


def _providers(ort) -> list[str]:
    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    if "CUDAExecutionProvider" in preferred and hasattr(ort, "preload_dlls"):
        with contextlib.suppress(Exception):  # CUDA/cuDNN from nvidia-* wheels, if present
            ort.preload_dlls()
    return preferred + ["CPUExecutionProvider"]


def _session():
    ort = _onnxruntime()  # before any download: no point fetching a model we can't run
    with _lock:
        path = str(model_file())
        if path not in _sessions:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            providers = _providers(ort)
            try:
                sess = ort.InferenceSession(path, opts, providers=providers)
            except Exception:
                if providers == ["CPUExecutionProvider"]:
                    raise
                sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            _sessions[path] = sess
        return _sessions[path]


def network_size(h: int, w: int, input_shape, size: int = 518) -> tuple[int, int]:
    """(height, width) to feed the network. Fixed-shape exports get their own size; dynamic
    ones keep the photo's aspect with the short side >= `size`, both multiples of 14
    (Depth Anything V2's 'lower_bound' resize), long side capped at 2 * size."""
    fixed = input_shape[2:4] if input_shape is not None and len(input_shape) == 4 else (None, None)
    if all(isinstance(d, int) and d > 0 for d in fixed):
        return int(fixed[0]), int(fixed[1])
    scale = size / min(h, w)
    scale = min(scale, 2 * size / max(h, w))
    nh = max(PATCH, math.ceil(h * scale / PATCH) * PATCH)
    nw = max(PATCH, math.ceil(w * scale / PATCH) * PATCH)
    return nh, nw


def preprocess(rgb: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 (1, 3, nh, nw), ImageNet-normalised."""
    shrink = nw < rgb.shape[1] and nh < rgb.shape[0]
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(raw: np.ndarray, h: int, w: int) -> np.ndarray:
    """Network output (relative inverse depth: bigger = closer) -> (h, w) float32 0..1."""
    d = np.squeeze(np.asarray(raw, np.float32))
    if d.ndim != 2:
        raise ValueError(f"Unexpected depth output shape {np.shape(raw)}")
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-8:
        return np.zeros((h, w), np.float32)
    return ((d - lo) / (hi - lo)).astype(np.float32)


def estimate_depth(rgb: np.ndarray, size: int = 518) -> np.ndarray:
    """Photo (H, W, 3 or 4) uint8 -> float32 (H, W) in 0..1, where 1 = nearest.

    Relative depth only (no metres): good for ordering and grouping, which is all
    parallax layers need. `size` is the network's short side (multiple of 14; bigger =
    finer edges, slower; 518 is what the model was trained at).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    h, w = rgb.shape[:2]
    sess = _session()
    inp = sess.get_inputs()[0]
    nh, nw = network_size(h, w, inp.shape, size)
    out = sess.run(None, {inp.name: preprocess(rgb, nh, nw)})[0]
    return postprocess(out, h, w)
