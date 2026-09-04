"""Tone Mapped Image Quality Index - revised."""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from TMQI.image_io import img_read
from TMQI.metric import TMQI, Metric, TMQIr, TMQIResult

try:
    __version__ = version("tmqi-revised")
except PackageNotFoundError:  # pragma: no cover - running from source, not installed
    __version__ = "0.0.0+unknown"

__title__ = "TMQIr"
__summary__ = "TMQI revised"
__uri__ = "https://github.com/dvolgyes/TMQI"

# for the Python reimplementation, original authors in 'upstream'
__author__ = "David Völgyes"
__email__ = "david.volgyes@ieee.org"

__derived__ = True  # Meaning: reimplementation with deviation
__upstream_license__ = "BSD-like"  # see the website for exact details
__upstream_uri__ = "https://ece.uwaterloo.ca/~z70wang/research/tmqi/"
__upstream_doi__ = "10.1109/TIP.2012.2221725"
__upstream_ref__ = (
    "H. Yeganeh and Z. Wang,"
    '"Objective Quality Assessment of Tone Mapped Images,"'
    "IEEE Transactions on Image Processing,"
    "vol. 22, no. 2, pp. 657-667, Feb. 2013."
)

__all__ = ["TMQI", "Metric", "TMQIResult", "TMQIr", "img_read"]

logger.disable("TMQI")
