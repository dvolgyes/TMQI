from pathlib import Path

import pytest
from loguru import logger

from tmqi import TMQI, TMQIr
from tmqi.image_io import img_read

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_SHAPE = (396, 561)  # (width, height), per the historical .float32 fixtures
CROP = 192  # smallest size that survives the 5-level pyramid (needs >=176)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def gray_png_pair():
    return (
        img_read(DATA_DIR / "test.png", gray=True),
        img_read(DATA_DIR / "test_ldr.png", gray=True),
    )


@pytest.fixture(scope="session")
def rgb_png_pair():
    return img_read(DATA_DIR / "test.png"), img_read(DATA_DIR / "test_ldr.png")


@pytest.fixture(scope="session")
def gray_raw_pair():
    return (
        img_read(DATA_DIR / "test.float32", gray=True, shape=RAW_SHAPE, dtype="float32"),
        img_read(DATA_DIR / "test_ldr.float32", gray=True, shape=RAW_SHAPE, dtype="float32"),
    )


@pytest.fixture(scope="session")
def rgb_raw_pair():
    return (
        img_read(DATA_DIR / "rgb_test.float32", shape=RAW_SHAPE, dtype="float32"),
        img_read(DATA_DIR / "rgb_test_ldr.float32", shape=RAW_SHAPE, dtype="float32"),
    )


@pytest.fixture(scope="session")
def small_gray_pair(gray_png_pair):
    hdr, ldr = gray_png_pair
    return hdr[:CROP, :CROP], ldr[:CROP, :CROP]


@pytest.fixture(scope="session")
def small_rgb_pair(rgb_png_pair):
    hdr, ldr = rgb_png_pair
    return hdr[:CROP, :CROP], ldr[:CROP, :CROP]


@pytest.fixture(scope="session")
def tmqi_small_result(small_rgb_pair):
    hdr, ldr = small_rgb_pair
    return TMQI()(hdr, ldr)


@pytest.fixture(scope="session")
def tmqir_small_result(small_rgb_pair):
    hdr, ldr = small_rgb_pair
    return TMQIr()(hdr, ldr)


@pytest.fixture
def loguru_messages():
    """Captures tmqi's loguru output (disabled by default for library consumers)."""
    messages: list[str] = []
    logger.enable("tmqi")
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)
    logger.disable("tmqi")
