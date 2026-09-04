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
    # Q/S use a loose absolute tolerance rather than exact precision-4 pinning: S is tiny
    # (~0.014) for this image, which makes Q sensitive enough to cross-platform BLAS
    # differences (observed directly: Linux gives Q=0.21950/S=0.01423, macOS CI gives
    # Q=0.21939/S=0.01423 for byte-identical input) that it flips the 4th decimal. See
    # FINDINGS.md.
    hdr, ldr = gray_png_pair
    result = TMQI()(hdr, ldr)
    assert result.Q == pytest.approx(0.2195, abs=0.005)
    assert result.S == pytest.approx(0.0143, abs=0.005)
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
def test_tmqir_gray_test_image(gray_png_pair, loguru_messages):
    # See FINDINGS.md: TMQIr's covariance formula suffers catastrophic cancellation on the
    # rescaled grayscale input, right at a knife-edge for this specific fixture -- whether
    # it trips into `nan` is not just SciPy-version-dependent but platform/BLAS-dependent
    # (observed directly: identical code gives `nan` on Linux, a normal float on macOS CI,
    # for byte-identical input). So this checks structural validity, and only checks the
    # warning mechanism when `nan` actually occurs here; it does not pin a specific outcome.
    # The reliable, platform-independent regression guard for the warning mechanism itself
    # is test_metric.py::test_negative_s_local_warns_for_tmqir_branch.
    hdr, ldr = gray_png_pair
    result = TMQIr()(hdr, ldr)
    assert len(result.s_local) == 5
    assert round(result.N, 4) == 0.0

    if math.isnan(result.Q):
        assert math.isnan(result.S)
        assert len(loguru_messages) >= 1
        assert "TIP.2015.2436340" in loguru_messages[0]
    else:
        assert not math.isnan(result.S)
        assert 0 <= result.Q <= 1


def test_no_warning_for_well_behaved_images(rgb_png_pair, loguru_messages):
    hdr, ldr = rgb_png_pair
    TMQI()(hdr, ldr)
    TMQIr()(hdr, ldr)
    assert loguru_messages == []
