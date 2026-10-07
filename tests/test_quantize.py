import numpy as np
import pytest

from pixel_pipeline.quantize import DITHER_MODES, bayer_matrix, quantize


def test_bayer_matrix_is_balanced():
    m = bayer_matrix(4)
    assert m.shape == (4, 4)
    assert len(np.unique(m)) == 16
    assert abs(m.mean()) < 1e-6 and m.min() > -0.5 and m.max() < 0.5


@pytest.mark.parametrize("method", DITHER_MODES)
def test_indices_valid_and_masked(method):
    rgb = np.tile(np.linspace(0, 255, 32).astype(np.uint8)[None, :, None], (8, 1, 3))
    mask = np.ones((8, 32), bool)
    mask[0, 0] = False
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    idx = quantize(rgb, mask, pal, method)
    assert idx[0, 0] == -1
    assert set(np.unique(idx[mask])) <= {0, 1}


def _transitions(row):
    return int((np.diff(row) != 0).sum())


@pytest.mark.parametrize("method", ["bayer4", "floyd"])
def test_dither_mixes_colours_across_a_gradient(method):
    rgb = np.tile(np.linspace(0, 255, 64).astype(np.uint8)[None, :, None], (8, 1, 3))
    mask = np.ones((8, 64), bool)
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    plain = quantize(rgb, mask, pal, "none")
    dithered = quantize(rgb, mask, pal, method)
    assert _transitions(plain[4]) == 1
    assert _transitions(dithered[4]) > 4


def test_unknown_dither_rejected():
    with pytest.raises(ValueError):
        quantize(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool),
                 np.zeros((2, 3), np.uint8), "sparkle")
