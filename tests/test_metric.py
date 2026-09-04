import math

import numpy as np
import pytest

from tmqi import TMQI, TMQIr
from tmqi.metric import TMQIResult


def test_result_is_named_tuple_and_unpacks(tmqi_small_result):
    result = tmqi_small_result
    assert isinstance(result, TMQIResult)
    Q, S, N, s_local, s_maps = result
    assert (Q, S, N) == (result.Q, result.S, result.N)
    assert len(s_local) == 5
    assert len(s_maps) == 5


def test_ranges(tmqi_small_result):
    result = tmqi_small_result
    assert 0 < result.Q <= 1
    assert 0 < result.S <= 1
    assert 0 < result.N <= 1


def test_determinism(small_rgb_pair):
    hdr, ldr = small_rgb_pair
    a = TMQI()(hdr, ldr)
    b = TMQI()(hdr, ldr)
    assert (a.Q, a.S, a.N) == (b.Q, b.S, b.N)


def test_identical_images_give_perfect_structural_fidelity_under_symmetric_scaling(small_rgb_pair):
    # TMQI's "original" mode rescales only L_hdr (not L_ldr), so it treats hdr/ldr
    # asymmetrically even when they're the same array -- S is not 1 there. TMQIr rescales
    # both the same way, so identical inputs do give perfect structural fidelity.
    hdr, _ = small_rgb_pair
    result = TMQIr()(hdr, hdr)
    assert result.S == pytest.approx(1.0, abs=1e-6)


def test_rgb_path_matches_pre_converted_gray_path(small_rgb_pair):
    hdr, ldr = small_rgb_pair
    metric = TMQI()
    rgb_result = metric(hdr, ldr)
    gray_result = metric._TMQI_gray(metric._RGBtoY(hdr), metric._RGBtoY(ldr))
    assert rgb_result.Q == gray_result.Q
    assert rgb_result.S == gray_result.S


def test_tmqi_and_tmqir_differ_in_naturalness(tmqi_small_result, tmqir_small_result):
    assert tmqi_small_result.N != tmqir_small_result.N


def test_custom_window(small_rgb_pair):
    hdr, ldr = small_rgb_pair
    window = np.ones((5, 5))
    result = TMQI()(hdr, ldr, window=window)
    assert isinstance(result, TMQIResult)


def test_constructor_with_args_forwards_to_call_and_propagates_errors():
    # TMQI(*args) triggers self.__call__(*args) internally (result discarded); this proves
    # the args are actually forwarded by checking that __call__'s validation still fires.
    hdr = np.zeros((5, 5))
    ldr = np.zeros((5, 5))
    with pytest.raises(ValueError, match="10x10"):
        TMQI(hdr, ldr)


@pytest.mark.parametrize(
    ("hdr_shape", "ldr_shape"),
    [((50, 50, 3), (60, 60, 3)), ((50, 50), (50, 50, 3))],
)
def test_mismatched_shapes_raise(hdr_shape, ldr_shape):
    hdr = np.zeros(hdr_shape)
    ldr = np.zeros(ldr_shape)
    with pytest.raises(ValueError, match="same shape"):
        TMQI()(hdr, ldr)


def test_too_small_image_raises():
    hdr = np.zeros((5, 5))
    ldr = np.zeros((5, 5))
    with pytest.raises(ValueError, match="10x10"):
        TMQI()(hdr, ldr)


def test_integer_dtype_raises():
    hdr = np.zeros((50, 50), dtype=np.uint8)
    ldr = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="floating"):
        TMQI()(hdr, ldr)


def test_non_3_channel_input_raises():
    hdr = np.zeros((50, 50, 4), dtype=np.float64)
    ldr = np.zeros((50, 50, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="3 channels"):
        TMQI()(hdr, ldr)


def test_rgbtoy_requires_3d():
    with pytest.raises(ValueError, match="NxMx3"):
        TMQI()._RGBtoY(np.zeros((10, 10)))


def test_oversized_window_raises(small_gray_pair):
    hdr, ldr = small_gray_pair
    window = np.ones((300, 300))
    with pytest.raises(ValueError, match="window shape"):
        TMQI()(hdr, ldr, window=window)


def test_non_2d_window_raises(small_gray_pair):
    hdr, ldr = small_gray_pair
    window = np.ones((5, 5, 1))
    with pytest.raises(ValueError, match="2-D"):
        TMQI()(hdr, ldr, window=window)


@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
def test_negative_s_local_warns_for_original_branch(loguru_messages):
    # See FINDINGS.md: a locally inverted-contrast patch gives a genuinely negative
    # covariance ratio even in TMQI's non-revised branch (no huge rescale of L_ldr involved),
    # confirming the negative-s_local issue is inherent to the formula, not just a TMQIr
    # numerical-precision artifact.
    rng = np.random.default_rng(0)
    hdr = rng.uniform(0, 255, (200, 200))
    ldr = 255 - hdr + rng.normal(0, 1, (200, 200))

    result = TMQI()(hdr, ldr)

    assert math.isnan(result.S)
    assert len(loguru_messages) >= 1
    assert "legitimate outcome of the formula" in loguru_messages[0]
    assert "TIP.2015.2436340" in loguru_messages[0]


@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
def test_negative_s_local_warns_for_tmqir_branch(loguru_messages):
    # Same inverted-contrast construction as above, but through TMQIr. Unlike the real
    # test.png fixture (see FINDINGS.md and test_regression.py), this deliberately
    # structural inversion gives s_local values around -0.9999 -- a massive, robust
    # signal that cross-platform floating-point noise (observed: identical code gives a
    # nan on Linux but a normal float on macOS CI for the borderline real-photo case)
    # cannot flip, so this is the reliable cross-platform regression guard for TMQIr's
    # negative-s_local warning path.
    rng = np.random.default_rng(0)
    hdr = rng.uniform(0, 255, (200, 200))
    ldr = 255 - hdr + rng.normal(0, 1, (200, 200))

    result = TMQIr()(hdr, ldr)

    assert math.isnan(result.S)
    assert len(loguru_messages) >= 1
    assert "TMQIr" in loguru_messages[0]
    assert "TIP.2015.2436340" in loguru_messages[0]
