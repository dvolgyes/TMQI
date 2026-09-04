"""Golden Q/S/N values captured from a legacy-environment baseline (see FINDINGS.md and the
padding-bug fix). Full-resolution, so these are marked slow.
"""

import math

import pytest

from tmqi import TMQI, TMQIr
from tmqi.image_io import img_read

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def off_pair(data_dir):
    return img_read(data_dir / "off.png"), img_read(data_dir / "off_ldr.png")


def test_tmqi_gray_test_image(gray_png_pair):
    hdr, ldr = gray_png_pair
    result = TMQI()(hdr, ldr)
    assert round(result.Q, 4) == 0.2195
    assert round(result.S, 4) == 0.0143
    assert round(result.N, 4) == 0.0


def test_tmqi_rgb_test_image(rgb_png_pair):
    hdr, ldr = rgb_png_pair
    result = TMQI()(hdr, ldr)
    assert round(result.Q, 4) == 0.8496
    assert round(result.S, 4) == 0.8281
    assert round(result.N, 4) == 0.3429


def test_tmqi_rgb_off_image(off_pair):
    hdr, ldr = off_pair
    result = TMQI()(hdr, ldr)
    assert round(result.Q, 4) == 0.8675
    assert round(result.S, 4) == 0.8388
    assert round(result.N, 4) == 0.4229


def test_tmqir_rgb_test_image(rgb_png_pair):
    hdr, ldr = rgb_png_pair
    result = TMQIr()(hdr, ldr)
    assert round(result.Q, 4) == 0.8934
    assert round(result.S, 4) == 0.9945
    assert round(result.N, 4) == 0.3452


def test_tmqir_rgb_off_image(off_pair):
    hdr, ldr = off_pair
    result = TMQIr()(hdr, ldr)
    assert round(result.Q, 4) == 0.8998
    assert round(result.S, 4) == 0.9951
    assert round(result.N, 4) == 0.3783


@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
def test_tmqir_gray_test_image_is_nan(gray_png_pair, loguru_messages):
    # See FINDINGS.md: TMQIr's covariance formula suffers catastrophic cancellation on the
    # rescaled grayscale input, so the exact negative s_local[0] is not stable across SciPy
    # versions. Only the (stable) nan-ness of Q/S and N are pinned here. The RuntimeWarning
    # is the expected, documented consequence of raising a negative s_local to a fractional
    # power -- not a bug.
    hdr, ldr = gray_png_pair
    result = TMQIr()(hdr, ldr)
    assert math.isnan(result.Q)
    assert math.isnan(result.S)
    assert round(result.N, 4) == 0.0
    assert len(result.s_local) == 5

    assert len(loguru_messages) == 1
    assert "negative" in loguru_messages[0]
    assert "TIP.2015.2436340" in loguru_messages[0]


def test_no_warning_for_well_behaved_images(rgb_png_pair, loguru_messages):
    hdr, ldr = rgb_png_pair
    TMQI()(hdr, ldr)
    TMQIr()(hdr, ldr)
    assert loguru_messages == []
