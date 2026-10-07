"""Write .aseprite files directly (no Aseprite install needed).

Follows https://github.com/aseprite/aseprite/blob/main/docs/ase-file-specs.md.
Sprites whose colours all fit in <= 255 palette entries (with binary alpha) are
written in Indexed mode -- index 0 is transparent -- so palette swaps work in
Aseprite. Anything else falls back to RGBA.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

CHUNK_LAYER = 0x2004
CHUNK_CEL = 0x2005
CHUNK_COLOR_PROFILE = 0x2007
CHUNK_TAGS = 0x2018
CHUNK_PALETTE = 0x2019

LOOP_DIRECTIONS = {"forward": 0, "reverse": 1, "pingpong": 2, "pingpong_reverse": 3}


def _string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<H", len(data)) + data


def _chunk(kind: int, data: bytes) -> bytes:
    return struct.pack("<IH", len(data) + 6, kind) + data


def _palette_for(frames: list[np.ndarray], palette: np.ndarray | None) -> np.ndarray | None:
    """Palette usable for Indexed mode, or None if the frames need RGBA."""
    pixels = np.concatenate([f.reshape(-1, 4) for f in frames])
    if not np.isin(pixels[:, 3], (0, 255)).all():
        return None
    used = np.unique(pixels[pixels[:, 3] == 255][:, :3], axis=0)
    pal = palette.astype(np.uint8) if palette is not None else used
    if len(pal) > 255:
        return None
    if len(used) and not (used[:, None, :] == pal[None]).all(-1).any(1).all():
        return None  # a pixel colour isn't in the palette
    return pal


def write_aseprite(path, frames: list[np.ndarray], palette: np.ndarray | None = None,
                   durations: int | list[int] = 100,
                   tags: list[tuple[str, int, int]] | tuple = (),
                   loop: str = "forward", layer_name: str = "Sprite") -> None:
    """frames: list of (H, W, 4) uint8 RGBA arrays, all the same size.
    tags: (name, first_frame, last_frame) animation ranges."""
    if not frames:
        raise ValueError("Need at least one frame")
    h, w = frames[0].shape[:2]
    if any(f.shape != (h, w, 4) for f in frames):
        raise ValueError("All frames must be RGBA and the same size")
    if isinstance(durations, int):
        durations = [durations] * len(frames)

    pal = _palette_for(frames, palette)
    indexed = pal is not None
    if indexed:
        entries = np.vstack([[0, 0, 0], pal]).astype(np.uint8)
        alphas = [0] + [255] * len(pal)
        # rgb -> palette index lookup (first match wins)
        lut = {tuple(c): i + 1 for i, c in reversed(list(enumerate(pal.tolist())))}
    else:
        entries = (palette if palette is not None else np.zeros((1, 3))).astype(np.uint8)[:256]
        alphas = [255] * len(entries)

    pal_data = struct.pack("<III8x", len(entries), 0, len(entries) - 1)
    for (r, g, b), a in zip(entries.tolist(), alphas):
        pal_data += struct.pack("<HBBBB", 0, r, g, b, a)

    first_chunks = [
        _chunk(CHUNK_COLOR_PROFILE, struct.pack("<HHI8x", 1, 0, 0)),  # sRGB
        _chunk(CHUNK_PALETTE, pal_data),
        _chunk(CHUNK_LAYER, struct.pack("<HHHHHHB3x", 3, 0, 0, 0, 0, 0, 255) + _string(layer_name)),
    ]
    if tags:
        tag_data = struct.pack("<H8x", len(tags))
        for name, start, end in tags:
            tag_data += struct.pack("<HHBH6xBBBx", start, end, LOOP_DIRECTIONS[loop], 0, 0, 0, 0)
            tag_data += _string(name)
        first_chunks.append(_chunk(CHUNK_TAGS, tag_data))

    body = b""
    for i, (frame, duration) in enumerate(zip(frames, durations)):
        if indexed:
            flat = frame.reshape(-1, 4)
            pixels = np.zeros(len(flat), np.uint8)
            opaque = np.nonzero(flat[:, 3] == 255)[0]
            pixels[opaque] = [lut[tuple(c)] for c in flat[opaque, :3].tolist()]
            raw = pixels.tobytes()
        else:
            raw = np.ascontiguousarray(frame, np.uint8).tobytes()
        cel = struct.pack("<HhhBHh5x", 0, 0, 0, 255, 2, 0) + struct.pack("<HH", w, h)
        cel += zlib.compress(raw)
        chunks = (first_chunks if i == 0 else []) + [_chunk(CHUNK_CEL, cel)]
        data = b"".join(chunks)
        n = len(chunks)
        body += struct.pack("<IHHH2xI", 16 + len(data), 0xF1FA, min(n, 0xFFFF), duration, n) + data

    header = struct.pack(
        "<IHHHHHIHIIB3xHBBhhHH84x",
        128 + len(body), 0xA5E0, len(frames), w, h, 8 if indexed else 32,
        1, durations[0], 0, 0, 0, len(entries), 1, 1, 0, 0, 16, 16)
    Path(path).write_bytes(header + body)
