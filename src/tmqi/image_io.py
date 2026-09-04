"""Reading HDR/LDR images and writing structural-fidelity maps."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import skimage.color

RAW_DTYPES: frozenset[str] = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


def img_read(
    source: str | Path,
    gray: bool = False,
    shape: tuple[int, int] | None = None,
    dtype: np.dtype | str | None = None,
) -> np.ndarray:
    """Read a local image file, either a regular image or a raw binary dump."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")

    if dtype is None:
        img = iio.imread(path)
        if gray and img.ndim > 2:
            img = skimage.color.rgb2hsv(img)[..., 2]
    else:
        if shape is None:
            raise ValueError("shape is required when dtype is given (raw image reads)")
        width, height = shape
        img = np.fromfile(path, dtype=dtype)
        img = img.reshape(height, width) if gray else img.reshape(height, width, -1)

    return img.astype(np.float64)


def write_map(path: str | Path, s_map: np.ndarray, maptype: str) -> None:
    """Write a structural-fidelity map either as a raw binary dump or as an image.

    Raw dtypes are written bit-exact via ``tofile``. Any other ``maptype`` is treated
    as an image format: s_map is bounded in [0, 1] per Yeganeh & Wang (brighter =
    higher fidelity), so it is clipped to that range and scaled to uint8 for writing.
    """
    if maptype in RAW_DTYPES:
        s_map.astype(maptype).tofile(path)
    else:
        scaled = (np.clip(s_map, 0.0, 1.0) * 255).astype(np.uint8)
        iio.imwrite(path, scaled)
