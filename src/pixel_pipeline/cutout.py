"""Background removal: rembg's ONNX models, run directly with onnxruntime.

Optional extra: pip install "pixel-pipeline[cutout]" (CPU) or "[cutout-gpu]" (NVIDIA).

Same models, preprocessing and postprocessing as rembg, so masks match it, but without
rembg's scipy / scikit-image / pymatting / numba dependencies (smaller, and PyInstaller
friendly). Models download on first use from rembg's GitHub release into rembg's cache
folder, so models rembg already downloaded are reused:

  ~/.rembg/models/<name>/<name>.onnx   ($REMBG_HOME or $XDG_DATA_HOME/rembg if set,
                                        $U2NET_HOME/models/... if U2NET_HOME is set)
  ~/.u2net/<name>.onnx                 (older rembg layout, read only)

Models:
  isnet-general-use      good default for animals/objects (~170 MB)
  u2net_human_seg        people (~170 MB)
  birefnet-general       best edges, slowest (~930 MB)
  birefnet-general-lite  BiRefNet quality, faster (~210 MB)
  birefnet-portrait      people, best edges (~930 MB)
  u2net / u2netp / silueta / isnet-anime   general / tiny / small / anime and cartoons
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
import tempfile
import threading
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class CutoutUnavailable(RuntimeError):
    pass


INSTALL_HINT = ('Background removal needs the optional extra: pip install "pixel-pipeline[cutout]" '
                "(or turn background removal off)")
RELEASE_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    file: str                 # asset name on rembg's GitHub release
    md5: str
    size: int                 # square network input, pixels
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    sigmoid: bool = False     # BiRefNet outputs logits
    mb: int = 0               # download size, for messages


MODELS: dict[str, ModelSpec] = {
    "isnet-general-use": ModelSpec("isnet-general-use.onnx", "fc16ebd8b0c10d971d3513d564d01e29",
                                   1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), mb=170),
    "isnet-anime": ModelSpec("isnet-anime.onnx", "6f184e756bb3bd901c8849220a83e38e",
                             1024, IMAGENET_MEAN, (1.0, 1.0, 1.0), mb=170),
    "u2net": ModelSpec("u2net.onnx", "60024c5c889badc19c04ad937298a77b",
                       320, IMAGENET_MEAN, IMAGENET_STD, mb=170),
    "u2netp": ModelSpec("u2netp.onnx", "8e83ca70e441ab06c318d82300c84806",
                        320, IMAGENET_MEAN, IMAGENET_STD, mb=5),
    "u2net_human_seg": ModelSpec("u2net_human_seg.onnx", "c09ddc2e0104f800e3e1bb4652583d1f",
                                 320, IMAGENET_MEAN, IMAGENET_STD, mb=170),
    "silueta": ModelSpec("silueta.onnx", "55e59e0d8062d2f5d013f4725ee84782",
                         320, IMAGENET_MEAN, IMAGENET_STD, mb=45),
    "birefnet-general": ModelSpec("BiRefNet-general-epoch_244.onnx", "7a35a0141cbbc80de11d9c9a28f52697",
                                  1024, IMAGENET_MEAN, IMAGENET_STD, sigmoid=True, mb=930),
    "birefnet-general-lite": ModelSpec("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
                                       "4fab47adc4ff364be1713e97b7e66334",
                                       1024, IMAGENET_MEAN, IMAGENET_STD, sigmoid=True, mb=210),
    "birefnet-portrait": ModelSpec("BiRefNet-portrait-epoch_150.onnx", "c3a64a6abf20250d090cd055f12a3b67",
                                   1024, IMAGENET_MEAN, IMAGENET_STD, sigmoid=True, mb=930),
}

_sessions: dict = {}
_lock = threading.Lock()


def _spec(model: str) -> ModelSpec:
    try:
        return MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown background-removal model {model!r}. "
                         f"Supported: {', '.join(MODELS)}") from None


# --- model files (same locations and env vars as rembg) ----------------------------------

def legacy_home() -> Path:
    """rembg's old flat folder (~/.u2net or $U2NET_HOME); only read from."""
    default = os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    return Path(os.path.expanduser(os.getenv("U2NET_HOME", default)))


def model_home() -> Path:
    """rembg's data root: $U2NET_HOME if set, else $REMBG_HOME, else ~/.rembg."""
    if os.getenv("U2NET_HOME"):
        return legacy_home()
    xdg = os.getenv("XDG_DATA_HOME")
    default = os.path.join(xdg, "rembg") if xdg else os.path.join("~", ".rembg")
    return Path(os.path.expanduser(os.getenv("REMBG_HOME", default)))


def model_path(model: str) -> Path:
    """Where `model` lives on disk: an existing download if any, else where it will go."""
    _spec(model)
    fname = f"{model}.onnx"
    target = model_home() / "models" / model / fname
    for path in (target, legacy_home() / fname):
        if path.exists():
            return path
    return target


def _log(msg: str, end: str = "\n") -> None:
    if sys.stderr is not None:  # None in a windowed (no console) build
        with contextlib.suppress(Exception):
            sys.stderr.write(msg + end)
            sys.stderr.flush()


def _download(url: str, dest: Path, md5: str | None) -> None:
    """Stream `url` to a temp file next to `dest`, check md5, then rename into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    digest = hashlib.md5(usedforsecurity=False)
    tty = sys.stderr is not None and sys.stderr.isatty()
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=60) as r:
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
        if md5 and digest.hexdigest() != md5:
            raise CutoutUnavailable(f"Downloaded model {url} is corrupt (md5 {digest.hexdigest()}, "
                                    f"expected {md5}). Try again.")
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def ensure_model(model: str) -> Path:
    """Path to the model's .onnx file, downloading it on first use."""
    spec = _spec(model)
    path = model_path(model)
    if path.exists():
        return path
    url = RELEASE_URL + spec.file
    _log(f"Downloading background-removal model {model} (~{spec.mb} MB) to {path.parent} ...")
    md5 = None if os.getenv("MODEL_CHECKSUM_DISABLED") is not None else spec.md5
    try:
        _download(url, path, md5)
    except OSError as e:  # includes URLError / timeouts
        raise CutoutUnavailable(f"Couldn't download the {model} model from {url} ({e}). Check your "
                                f"internet connection, or save that file as {path}") from e
    return path


# --- inference ----------------------------------------------------------------------------

def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as e:
        raise CutoutUnavailable(INSTALL_HINT) from e
    return onnxruntime


def _providers(ort) -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        # onnxruntime-gpu >= 1.21 can load CUDA/cuDNN from the nvidia-* pip wheels.
        if hasattr(ort, "preload_dlls") and importlib.util.find_spec("nvidia") is not None:
            with contextlib.suppress(Exception):
                ort.preload_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(model: str):
    """New onnxruntime InferenceSession for `model` (GPU if available, else CPU)."""
    ort = _onnxruntime()  # before any download: no point fetching a model we can't run
    path = str(ensure_model(model))
    ort.set_default_logger_severity(3)  # errors only, like rembg
    opts = ort.SessionOptions()
    # Same output, about half the peak RAM (BiRefNet on CPU: ~12 GB -> ~6.7 GB), similar speed.
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    providers = _providers(ort)
    try:
        session = ort.InferenceSession(path, opts, providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(path, opts, providers=providers)
    if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in session.get_providers():
        warnings.warn("CUDA was requested but onnxruntime is running on CPU (CUDA 12 / cuDNN 9 "
                      "libraries not found), so background removal will be slower.",
                      RuntimeWarning, stacklevel=3)
    return session


def _session(model: str):
    with _lock:
        if model not in _sessions:
            _sessions[model] = _create_session(model)
        return _sessions[model]


def preprocess(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    """RGB image -> (1, 3, S, S) float32 network input, exactly as rembg's BaseSession.normalize."""
    im = image.convert("RGB").resize((spec.size, spec.size), Image.Resampling.LANCZOS)
    a = np.array(im)
    a = a / max(np.max(a), 1e-6)  # rembg scales by the image max, not 255
    a = (a - np.asarray(spec.mean)) / np.asarray(spec.std)
    return np.expand_dims(a.transpose((2, 0, 1)), 0).astype(np.float32)


def postprocess(output: np.ndarray, spec: ModelSpec, size: tuple[int, int]) -> np.ndarray:
    """First network output (1, 1, S, S) -> (H, W) uint8 mask; `size` is (W, H)."""
    pred = output[:, 0, :, :]
    if spec.sigmoid:
        pred = 1 / (1 + np.exp(-pred))
    ma, mi = np.max(pred), np.min(pred)
    pred = (pred - mi) / (ma - mi) if ma > mi else np.zeros_like(pred)
    pred = np.squeeze(pred, 0)
    mask = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))
    return np.array(mask.resize(size, Image.Resampling.LANCZOS))


def remove_background(rgb: np.ndarray, model: str = "isnet-general-use") -> np.ndarray:
    """Return an alpha mask (H, W) uint8 for the main subject of an RGB photo."""
    spec = _spec(model)
    session = _session(model)
    image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    feed = {session.get_inputs()[0].name: preprocess(image, spec)}
    return postprocess(session.run(None, feed)[0], spec, image.size)
