import numpy as np
import pytest
from PIL import Image


def make_blob(w=200, h=160) -> np.ndarray:
    """Synthetic 'cut-out photo': shaded ellipse with an eye, transparent background."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2) / (w * 0.35)) ** 2 + ((yy - h / 2) / (h * 0.4)) ** 2 <= 1
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = np.clip(120 + xx * 0.6, 0, 255)
    rgba[..., 1] = np.clip(60 + yy * 0.5, 0, 255)
    rgba[..., 2] = 40
    eye = (xx - w * 0.6) ** 2 + (yy - h * 0.4) ** 2 <= (h * 0.06) ** 2
    rgba[eye, :3] = 10
    rgba[..., 3] = np.where(inside, 255, 0)
    return rgba


def make_gradient(w=240, h=120) -> np.ndarray:
    """Opaque left-dark -> right-bright photo."""
    xx = np.tile(np.linspace(0, 255, w), (h, 1))
    rgb = np.stack([xx, xx * 0.8, 255 - xx], -1).astype(np.uint8)
    return np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], -1)


@pytest.fixture
def blob():
    return make_blob()


@pytest.fixture
def gradient():
    return make_gradient()


@pytest.fixture
def blob_png(tmp_path):
    path = tmp_path / "blob.png"
    Image.fromarray(make_blob(), "RGBA").save(path)
    return path


_QAPP = []


def qt_app():
    """The one QApplication for the whole test run. Qt can't destroy its application object
    and make another: a QApplication held in a test's local variable dies with the test while
    its windows live on, and the next one then crashes at random (CI on Windows)."""
    from PySide6.QtWidgets import QApplication
    if not _QAPP:
        _QAPP.append(QApplication.instance() or QApplication([]))
    return _QAPP[0]


@pytest.fixture(autouse=True)
def _own_library(tmp_path_factory, monkeypatch):
    """Tests never touch the real Character Library."""
    monkeypatch.setenv("PIXEL_LIBRARY", str(tmp_path_factory.mktemp("library")))
