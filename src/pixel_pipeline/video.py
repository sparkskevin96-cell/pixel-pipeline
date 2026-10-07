"""Phone video -> animation frames -> sprites sharing one palette.

align="track"  subject stays centred and on the floor in every frame: a walk or
               run "in place", which is what game sprites want (works with a
               hand-held camera that follows the subject)
align="fixed"  one crop for all frames: keeps real movement through the frame
               (jumps, a dog crossing the shot); needs a still camera
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pipeline import Sprite, isolate, sprites_from_rgba
from .settings import Settings

ALIGN_MODES = ("track", "fixed")


@dataclass
class Clip:
    frames: list[np.ndarray]  # RGBA uint8
    duration_ms: int          # time each sampled frame represents


def read_frames(path, count: int | None = 8, start: float | None = None,
                end: float | None = None, fps: float | None = None,
                max_side: int = 1280) -> Clip:
    """Sample frames between start and end (seconds).

    fps given: take frames at that rate (count caps how many).
    otherwise: `count` frames spread evenly (end exclusive, so the clip loops).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Couldn't open video {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        first = int(round((start or 0) * src_fps))
        last = int(round(end * src_fps)) if end else total
        if total:
            last = min(last, total)
        if last <= first:
            raise ValueError(f"Empty time range in {path} ({start}s to {end}s)")
        if fps:
            step = src_fps / fps
            wanted = [int(round(first + i * step)) for i in range(int((last - first) / step) or 1)]
            if count:
                wanted = wanted[:count]
            duration = int(round(1000 / fps))
        else:
            n = count or 8
            wanted = sorted({int(first + i * (last - first) / n) for i in range(n)})
            duration = int(round(1000 * (last - first) / src_fps / n))
        want = set(wanted)
        frames, pos = [], 0
        while pos <= max(wanted):
            ok = cap.grab()
            if not ok:
                break
            if pos in want:
                ok, bgr = cap.retrieve()
                if ok:
                    frames.append(_to_rgba(bgr, max_side))
            pos += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return Clip(frames, max(1, duration))


def _to_rgba(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    return rgba


def _crop_pad(rgba: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop a box that may extend past the image; outside is transparent."""
    h, w = rgba.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgba[sy0:sy1, sx0:sx1]
    return out


def align_frames(frames: list[np.ndarray], mode: str = "track", margin: float = 0.04,
                 threshold: int = 128) -> list[np.ndarray]:
    """Crop every frame to the same-size box around the subject (see module doc)."""
    if mode not in ALIGN_MODES:
        raise ValueError(f"Unknown align mode '{mode}' (use {', '.join(ALIGN_MODES)})")
    boxes = []
    for f in frames:
        ys, xs = np.nonzero(f[..., 3] >= threshold)
        if not len(xs):
            h, w = f.shape[:2]
            boxes.append((0, 0, w, h, w / 2))
            continue
        weight = f[ys, xs, 3].astype(np.float64)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1,
                      float((xs * weight).sum() / weight.sum())))
    if mode == "fixed":
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        m = int(round(max(x1 - x0, y1 - y0) * margin))
        return [_crop_pad(f, x0 - m, y0 - m, x1 + m, y1 + m) for f in frames]
    bw = max(b[2] - b[0] for b in boxes)
    bh = max(b[3] - b[1] for b in boxes)
    m = int(round(max(bw, bh) * margin))
    bw, bh = bw + 2 * m, bh + 2 * m
    out = []
    for f, (_, _, _, by1, cx) in zip(frames, boxes):
        x0 = int(round(cx - bw / 2))
        y1 = by1 + m
        out.append(_crop_pad(f, x0, y1 - bh, x0 + bw, y1))
    return out


def video_to_sprites(path, settings: Settings, count: int | None = 8, start: float | None = None,
                     end: float | None = None, fps: float | None = None,
                     align: str = "track") -> tuple[list[Sprite], int]:
    """Returns (sprites, ms per frame). All frames share one palette and one scale."""
    clip = read_frames(path, count, start, end, fps)
    isolated = [isolate(f, settings) for f in clip.frames]
    aligned = align_frames(isolated, align, settings.margin, settings.alpha_threshold)
    # The aligned boxes already frame the subject consistently; cropping each
    # frame again would rescale frames independently and make the sprite jitter.
    s = Settings(**{**settings.to_dict(), "crop_to_subject": False})
    return sprites_from_rgba(aligned, s, shared_palette=True), clip.duration_ms


def is_video(path) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".gif"}
