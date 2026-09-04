import re

import numpy as np
import pytest

from tmqi.image_io import RAW_DTYPES, img_read, write_map


def test_png_rgb_shape_and_dtype(rgb_png_pair):
    hdr, _ = rgb_png_pair
    assert hdr.shape == (561, 396, 3)
    assert hdr.dtype == np.float64


def test_png_gray_shape_and_range(gray_png_pair):
    hdr, _ = gray_png_pair
    assert hdr.shape == (561, 396)
    assert hdr.min() >= 0.0 and hdr.max() <= 1.0


def test_raw_gray_shape(gray_raw_pair):
    hdr, _ = gray_raw_pair
    assert hdr.shape == (561, 396)


def test_raw_rgb_shape(rgb_raw_pair):
    hdr, _ = rgb_raw_pair
    assert hdr.shape == (561, 396, 3)


def test_raw_rgb_matches_png_rgb_exactly(rgb_png_pair, rgb_raw_pair):
    # The historical Makefile invariant: raw-float32-derived and PNG-derived inputs must
    # give the same metric result. At the array level the RGB raw dump is a bit-exact
    # float32 capture of the PNG-decoded pixels.
    png_hdr, png_ldr = rgb_png_pair
    raw_hdr, raw_ldr = rgb_raw_pair
    assert np.array_equal(png_hdr, raw_hdr)
    assert np.array_equal(png_ldr, raw_ldr)


def test_raw_gray_matches_png_gray_mostly(gray_png_pair, gray_raw_pair):
    # Unlike the RGB raw dump, the historical gray .float32 fixtures are NOT a bit-exact
    # capture of skimage's rgb2hsv V-channel: measured directly, 163/222156 pixels
    # (~0.07%) differ by up to ~0.0117, while the other 99.9%+ match closely. This is a
    # pre-existing property of the 2018-era fixture files themselves (confirmed: skimage's
    # V channel is exactly max(R,G,B)/255, verified to float64 epsilon against a manual
    # computation, so this isn't a skimage/imageio version artifact) -- not a bug in this
    # code. See FINDINGS.md. The bound below catches a real regression (e.g. a reshape or
    # dtype bug) without being sensitive to this known, small, pre-existing discrepancy.
    png_hdr, png_ldr = gray_png_pair
    raw_hdr, raw_ldr = gray_raw_pair
    for png_img, raw_img in ((png_hdr, raw_hdr), (png_ldr, raw_ldr)):
        diff = np.abs(png_img - raw_img)
        assert np.mean(diff > 1e-4) < 0.005, "more than 0.5% of pixels diverge unexpectedly"
        assert diff.max() < 0.02, "an outlier pixel exceeds the known historical divergence"


def test_missing_file_raises_with_path(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        img_read(missing)


def test_missing_raw_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.float32"
    with pytest.raises(FileNotFoundError):
        img_read(missing, shape=(10, 10), dtype="float32")


def test_raw_dtype_without_shape_raises(data_dir):
    with pytest.raises(ValueError, match="shape is required"):
        img_read(data_dir / "test.float32", dtype="float32")


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8"])
def test_write_map_raw_round_trip(tmp_path, dtype):
    s_map = np.linspace(-0.1, 1.0, 20).reshape(4, 5).astype(np.float64)
    path = tmp_path / f"s_map_1.{dtype}"
    write_map(path, s_map, dtype)
    roundtripped = np.fromfile(path, dtype=dtype).reshape(4, 5)
    np.testing.assert_array_equal(roundtripped, s_map.astype(dtype))


def test_write_map_png_is_clipped_and_scaled(tmp_path):
    import imageio.v3 as iio

    s_map = np.array([[-0.5, 0.0], [0.5, 1.5]])
    path = tmp_path / "s_map_1.png"
    write_map(path, s_map, "png")
    written = iio.imread(path)
    assert written.dtype == np.uint8
    assert list(written.flatten()) == [0, 0, 127, 255]


def test_raw_dtypes_are_all_valid_numpy_dtypes():
    for dtype in RAW_DTYPES:
        np.dtype(dtype)
