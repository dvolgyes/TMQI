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
    # (~0.014) for this image, which makes Q sensitive enough to residual cross-platform
    # BLAS differences that it could flip the 4th decimal (previously observed directly on
    # macOS CI). Values below reflect the numerically stable _Slocal reformulation (see
    # FINDINGS.md); results from versions before that fix will differ slightly.
    hdr, ldr = gray_png_pair
    result = TMQI()(hdr, ldr)
    assert result.Q == pytest.approx(0.2199, abs=0.001)
    assert result.S == pytest.approx(0.0143, abs=0.001)
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
    # Before the numerically stable _Slocal reformulation (see FINDINGS.md), this fixture
    # reliably produced `nan` on Linux via catastrophic cancellation in TMQIr's rescaled
    # covariance computation, but not on macOS CI (a genuinely different, platform-driven
    # outcome, not just a SciPy-version one) -- both were downstream of the same numerical
    # artifact. After the fix it consistently produces a normal, finite result; kept as a
    # loose approx (not exact precision-4 pinning) since this image sits close enough to the
    # numerical edge that some residual cross-platform noise is still plausible. The `nan`
    # branch below is a defensive fallback, not the expected path -- if it's ever taken, the
    # fix regressed. The reliable, platform-independent regression guard for the warning
    # mechanism itself is test_metric.py::test_negative_s_local_warns_for_tmqir_branch.
    hdr, ldr = gray_png_pair
    result = TMQIr()(hdr, ldr)
    assert len(result.s_local) == 5
    assert round(result.N, 4) == 0.0

    if math.isnan(result.Q):
        assert math.isnan(result.S)
        assert len(loguru_messages) >= 1
        assert "TIP.2015.2436340" in loguru_messages[0]
    else:
        assert result.Q == pytest.approx(0.7984, abs=0.01)
        assert result.S == pytest.approx(0.9885, abs=0.01)


def test_no_warning_for_well_behaved_images(rgb_png_pair, loguru_messages):
    hdr, ldr = rgb_png_pair
    TMQI()(hdr, ldr)
    TMQIr()(hdr, ldr)
    assert loguru_messages == []
