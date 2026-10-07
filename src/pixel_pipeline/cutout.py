"""Background removal via rembg (optional extra: pip install "pixel-pipeline[cutout]").

Models (downloaded on first use to ~/.u2net):
  isnet-general-use  good default for animals/objects (~170 MB)
  u2net_human_seg    people
  birefnet-general   best edges, slowest (~900 MB)
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class CutoutUnavailable(RuntimeError):
    pass


_sessions: dict = {}


def remove_background(rgb: np.ndarray, model: str = "isnet-general-use") -> np.ndarray:
    """Return an alpha mask (H, W) uint8 for the main subject of an RGB photo."""
    try:
        from rembg import new_session, remove
    except ImportError as e:
        raise CutoutUnavailable(
            'Background removal needs the optional extra: pip install "pixel-pipeline[cutout]" '
            "(or turn background removal off)") from e
    if model not in _sessions:
        _sessions[model] = new_session(model)
    mask = remove(Image.fromarray(rgb), session=_sessions[model], only_mask=True)
    return np.asarray(mask.convert("L"))
