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


def pytest_configure(config):
    """Tests run at idle priority (Windows; nice 10 elsewhere) so the laptop stays usable while
    they go; child processes (Godot, worker pools) inherit it. PIXEL_TEST_PRIORITY=normal keeps
    the normal priority."""
    import os
    if os.environ.get("PIXEL_TEST_PRIORITY", "").lower() == "normal":
        return
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:       # no psutil / not allowed: run as is
        pass


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
    monkeypatch.setenv("PIXEL_DRAFT", "0")       # full-quality sampling whatever the app's switch says
    monkeypatch.setenv("PIXEL_HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("PIXEL_DRAWN", "0")       # library characters: 3D render unless asked
    # CPU caches (cache.py) per test: a test never gets another test's results (some tests
    # swap functions out), and the real ~/.pixel-pipeline/cache is never touched
    monkeypatch.setenv("PIXEL_CACHE", str(tmp_path_factory.mktemp("cache")))
    monkeypatch.setenv("PIXEL_WORKERS", "1")     # in-process (tests/test_speed.py tries workers)
    monkeypatch.setenv("PIXEL_GPU_RETRIES", "0")  # a dead fake ComfyUI fails at once (test_robust
    #                                               turns retries on with a dying fake)
    monkeypatch.setenv("PIXEL_KD_WORKERS", "-1")  # drawnrig nearest-point searches on every core
    #                                               (same answers; the app's default is 1)


_EXIT = [0]


def pytest_sessionfinish(session, exitstatus):
    _EXIT[0] = int(exitstatus)


def pytest_unconfigure(config):
    """Once the results are printed, end the process without Python tearing Qt down: with
    windows from many tests still around, that teardown crashed at random (CI, Linux and
    Windows) after every test had passed."""
    if _QAPP:
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt":     # ExitProcess still unloads Qt's DLLs, which is what crashes
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateProcess(k32.GetCurrentProcess(), _EXIT[0])
        os._exit(_EXIT[0])


@pytest.fixture(autouse=True)
def _collect_garbage_on_the_main_thread():
    """Widgets a test made and dropped (pages in reference cycles) are freed right after it, on
    this thread. Left to Python's garbage collector they can be freed on whichever thread runs
    it next, e.g. a fake ComfyUI server's, and Qt aborts deleting widgets there (seen in
    test_portraits once the import order shifted the collector's timing)."""
    yield
    import gc
    gc.collect()
