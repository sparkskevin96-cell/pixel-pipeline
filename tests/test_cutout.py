"""Background removal plumbing, without onnxruntime or model downloads (fake session)."""
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import cutout
from pixel_pipeline.cutout import CutoutUnavailable, remove_background


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: a centred disc of 'subject' logits."""

    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name="input.1")]

    def run(self, output_names, feed):
        self.feeds.append(feed)
        s = feed["input.1"].shape[-1]
        yy, xx = np.mgrid[0:s, 0:s]
        disc = ((xx - s / 2) ** 2 + (yy - s / 2) ** 2) < (s / 4) ** 2
        logits = np.where(disc, 6.0, -6.0).astype(np.float32)[None, None]
        return [logits, np.zeros_like(logits)]  # rembg uses the first output only


@pytest.fixture
def fake_sessions(monkeypatch):
    made = {}

    def factory(model):
        made[model] = FakeSession()
        return made[model]

    monkeypatch.setattr(cutout, "_sessions", {})
    monkeypatch.setattr(cutout, "_create_session", factory)
    return made


def photo(w=300, h=200):
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("model", ["isnet-general-use", "u2net_human_seg", "birefnet-general"])
def test_mask_shape_and_network_input(fake_sessions, model):
    mask = remove_background(photo(), model)
    assert mask.shape == (200, 300) and mask.dtype == np.uint8
    assert mask[100, 150] > 200 and mask[0, 0] < 50       # disc in the middle, background out
    (feed,) = fake_sessions[model].feeds
    x = feed["input.1"]
    size = cutout.MODELS[model].size
    assert x.shape == (1, 3, size, size) and x.dtype == np.float32


def test_session_is_cached_per_model(fake_sessions):
    remove_background(photo(), "u2net")
    remove_background(photo(), "u2net")
    remove_background(photo(), "isnet-general-use")
    assert len(fake_sessions["u2net"].feeds) == 2
    assert set(fake_sessions) == {"u2net", "isnet-general-use"}


def test_preprocess_matches_rembg_normalisation():
    spec = cutout.MODELS["isnet-general-use"]  # mean 0.5, std 1
    img = Image.fromarray(np.full((10, 20, 3), 100, np.uint8))
    x = cutout.preprocess(img, spec)
    assert x.shape == (1, 3, 1024, 1024) and x.dtype == np.float32
    np.testing.assert_allclose(x, 0.5)  # rembg divides by the image max: 100/100 - 0.5
    spec = cutout.MODELS["u2net"]  # ImageNet mean/std at 320 px
    img = Image.fromarray(photo())
    small = np.array(img.resize((320, 320), Image.Resampling.LANCZOS)) / 255.0
    expected = (small - spec.mean) / spec.std
    x = cutout.preprocess(img, spec)
    assert x.shape == (1, 3, 320, 320)
    np.testing.assert_allclose(x[0].transpose(1, 2, 0), expected, rtol=1e-5, atol=1e-5)


def test_postprocess_normalises_and_resizes():
    spec = cutout.MODELS["u2net"]
    out = np.full((1, 1, 320, 320), -3.0, np.float32)
    out[..., 160:] = 7.0                                    # left half background, right subject
    mask = cutout.postprocess(out, spec, (50, 40))
    assert mask.shape == (40, 50) and mask.dtype == np.uint8
    assert (mask[:, :20] == 0).all() and (mask[:, -20:] == 255).all()  # stretched to 0..255
    flat = cutout.postprocess(np.zeros((1, 1, 8, 8), np.float32), spec, (5, 5))
    assert (flat == 0).all()                                # constant output: no NaNs
    birefnet = cutout.postprocess(out, cutout.MODELS["birefnet-general"], (50, 40))
    assert birefnet.shape == (40, 50) and birefnet.max() >= 253


def test_unknown_model_lists_supported_names(fake_sessions):
    with pytest.raises(ValueError, match="isnet-general-use.*birefnet-portrait"):
        remove_background(photo(), "nope")
    assert not fake_sessions


def test_missing_onnxruntime_raises_before_download(monkeypatch):
    monkeypatch.setattr(cutout, "_sessions", {})
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(cutout, "ensure_model", lambda m: pytest.fail("downloaded without onnxruntime"))
    with pytest.raises(CutoutUnavailable, match="pixel-pipeline\\[cutout\\]"):
        remove_background(photo())


def test_model_paths_follow_rembg_env_vars(monkeypatch, tmp_path):
    for var in ("U2NET_HOME", "REMBG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REMBG_HOME", str(tmp_path / "rembg"))
    assert cutout.model_path("u2net") == tmp_path / "rembg" / "models" / "u2net" / "u2net.onnx"
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "u2"))  # wins, like rembg
    assert cutout.model_path("u2net") == tmp_path / "u2" / "models" / "u2net" / "u2net.onnx"
    (tmp_path / "u2").mkdir()
    (tmp_path / "u2" / "u2net.onnx").write_bytes(b"old layout")  # older rembg's flat folder
    assert cutout.model_path("u2net") == tmp_path / "u2" / "u2net.onnx"


def test_download_checks_md5_and_is_atomic(tmp_path):
    src = tmp_path / "src.onnx"
    src.write_bytes(b"model bytes" * 1000)
    dest = tmp_path / "cache" / "m.onnx"
    with pytest.raises(CutoutUnavailable, match="corrupt"):
        cutout._download(src.as_uri(), dest, "0" * 32)
    assert not dest.exists() and list(dest.parent.iterdir()) == []  # temp file cleaned up
    cutout._download(src.as_uri(), dest, hashlib.md5(src.read_bytes()).hexdigest())
    assert dest.read_bytes() == src.read_bytes()
    assert list(dest.parent.iterdir()) == [dest]


def test_failed_download_is_a_cutout_error(monkeypatch, tmp_path):
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.setenv("REMBG_HOME", str(tmp_path))
    monkeypatch.setattr(cutout, "RELEASE_URL", (tmp_path / "missing").as_uri() + "/")
    with pytest.raises(CutoutUnavailable, match="Couldn't download the u2netp model"):
        cutout.ensure_model("u2netp")


def test_gpu_provider_preferred_when_available():
    gpu = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    cpu = SimpleNamespace(get_available_providers=lambda: ["AzureExecutionProvider", "CPUExecutionProvider"])
    assert cutout._providers(gpu) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert cutout._providers(cpu) == ["CPUExecutionProvider"]
