"""The TMQI and TMQIr metrics: pure numpy/scipy, no I/O.

Identifiers such as ``Q``, ``S``, ``N``, ``L_hdr``, ``_RGBtoY`` mirror the reference
Matlab implementation and the notation of the original paper; they are kept as-is
rather than renamed to snake_case.
"""

from typing import NamedTuple, cast

import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import generic_filter
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.stats import beta, norm
from skimage.util import view_as_blocks


class TMQIResult(NamedTuple):
    """The published TMQI descriptors: quality, structural fidelity, naturalness, per-scale."""

    Q: float
    S: float
    N: float
    s_local: list[float]
    s_maps: list[np.ndarray]


def _validate_image_pair(hdr_image: np.ndarray, ldr_image: np.ndarray) -> None:
    for name, img in (("hdrImage", hdr_image), ("ldrImage", ldr_image)):
        if not np.issubdtype(img.dtype, np.floating):
            raise ValueError(f"{name} must have a floating dtype, got {img.dtype}")
        if img.ndim not in (2, 3):
            raise ValueError(f"{name} must be 2-D (NxM) or 3-D (NxMx3), got shape {img.shape}")
        if img.ndim == 3 and img.shape[2] != 3:
            raise ValueError(f"{name} 3-D input must have 3 channels, got shape {img.shape}")
    if hdr_image.shape != ldr_image.shape:
        raise ValueError(
            f"hdrImage and ldrImage must have the same shape: "
            f"{hdr_image.shape} vs {ldr_image.shape}"
        )
    n, m = hdr_image.shape[:2]
    if n <= 10 or m <= 10:
        raise ValueError(f"images must be larger than 10x10, got {hdr_image.shape[:2]}")


class Metric:
    """Base class describing an image-quality metric's descriptors and metadata."""

    name: str = "Undefined"
    descriptors: tuple[str, ...] = ()
    lists: tuple[str, ...] = ()
    maps: tuple[str, ...] = ()
    no_reference: bool = False
    full_reference: bool = False
    luminance: bool = False
    RGB: bool = False

    def __init__(self, *args, **kwargs) -> None:
        self.cache: dict[str, object] = {}

    def _RGBtoY(self, RGB: np.ndarray) -> np.ndarray:
        if RGB.ndim != 3 or RGB.shape[2] != 3:
            raise ValueError(f"expected an NxMx3 RGB array, got shape {RGB.shape}")
        weights = np.asarray([0.2126, 0.7152, 0.0722])
        return cast(np.ndarray, np.einsum("...c,c->...", RGB, weights))


class TMQI(Metric):
    """Yeganeh & Wang's TMQI (original formulation); constructing with args also calls it."""

    name = "TMQI"
    descriptors = ("Q", "S", "N")
    lists = ("s_local",)
    maps = ("s_local",)
    no_reference = False
    full_reference = True
    luminance = True
    RGB = False
    original = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if len(args) + len(kwargs) > 0:
            self.__call__(*args, **kwargs)

    def __call__(
        self, hdrImage: np.ndarray, ldrImage: np.ndarray, window: np.ndarray | None = None
    ) -> TMQIResult:
        """Compute Q, S, N (and their per-scale components) for hdrImage vs ldrImage."""
        _validate_image_pair(hdrImage, ldrImage)

        if hdrImage.ndim == 3 and ldrImage.ndim == 3:
            # Processing RGB images
            L_hdr = self._RGBtoY(hdrImage)
            L_ldr = self._RGBtoY(ldrImage)
            return self._TMQI_gray(L_hdr, L_ldr, window)

        # input is already grayscale
        return self._TMQI_gray(hdrImage, ldrImage, window)

    def _TMQI_gray(
        self, hdrImage: np.ndarray, ldrImage: np.ndarray, window: np.ndarray | None = None
    ) -> TMQIResult:
        a = 0.8012
        Alpha = 0.3046
        Beta = 0.7088
        lvl = 5  # levels
        weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        M, N = hdrImage.shape

        if window is None:
            gauss = gaussian(11, 1.5)
            window = np.outer(gauss, gauss)
        else:
            if window.ndim != 2:
                raise ValueError(f"window must be 2-D, got shape {window.shape}")
            U, V = window.shape
            if not (2 <= U < N and 2 <= V < M):
                raise ValueError(f"window shape {window.shape} must satisfy 2<=U<{N}, 2<=V<{M}")

        # unnecessary, it is just for the sake of parallels with the matlab code
        L_hdr = hdrImage
        L_ldr = ldrImage

        # Naturalness should be calculated before rescaling
        N = self._StatisticalNaturalness(ldrImage)

        # The images should have the same dynamic ranges, e.g. [0,255]

        factor = 2**32 - 1.0

        if self.original:
            L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
        else:
            # but we really should scale them similarly...
            L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
            L_ldr = factor * (L_ldr - L_ldr.min()) / (L_ldr.max() - L_ldr.min())

        S, s_local, s_maps = self._StructuralFidelity(L_hdr, L_ldr, lvl, weight, window)
        Q = a * (S**Alpha) + (1.0 - a) * (N**Beta)
        return TMQIResult(Q, S, N, s_local, s_maps)

    def _warn_negative_s_local(self, level: int, num_levels: int, sl: float) -> None:
        # See FINDINGS.md. S_local's covariance ratio (sigma_xy + C2) / (sigma_x*sigma_y + C2)
        # is not bounded below by zero -- a genuinely anti-correlated HDR/LDR patch (same as
        # SSIM's structure term) can push it negative. S = prod(s_local ** weight) then raises
        # a negative base to a fractional exponent, which is mathematically undefined (nan).
        # This is not fixed here, by design: it's an inherent property of the published TMQI
        # formula, and "fixing" it would make scores incomparable to the original/TMQIr values.
        # _Slocal computes sigma1_sq/sigma2_sq/sigma12 via a numerically stable two-pass
        # formula (see _Slocal's own comment), so unlike in earlier versions this is unlikely
        # to be a spurious floating-point artifact -- it most likely reflects a genuinely
        # anti-correlated local patch.
        cause = (
            "A negative aggregate local covariance at this pyramid level is an unusual "
            "but legitimate outcome of the formula."
        )
        logger.warning(
            "pyramid level {}/{}: local structural fidelity s_local={:.4g} is negative. {} "
            "Q and/or S will likely come out nan, since S = prod(s_local ** weight) is "
            "undefined for a negative base. See TMQI's FINDINGS.md. The official follow-up, "
            "TMQI-II (K. Ma, H. Yeganeh, K. Zeng, Z. Wang, 'High Dynamic Range Image "
            "Compression by Optimizing Tone Mapped Image Quality Index,' IEEE Trans. Image "
            "Process., vol. 24, no. 10, pp. 3086-3097, 2015, doi:10.1109/TIP.2015.2436340), "
            "revises other parts of the structural-fidelity formula but keeps this same "
            "covariance ratio unchanged.",
            level,
            num_levels,
            sl,
            cause,
        )

    def _StructuralFidelity(
        self,
        L_hdr: np.ndarray,
        L_ldr: np.ndarray,
        level: int,
        weight: list[float],
        window: np.ndarray,
    ) -> tuple[float, list[float], list[np.ndarray]]:
        f = 32.0
        s_local = []
        s_maps = []
        kernel = np.ones((2, 2)) / 4.0

        for lvl in range(level):
            f = f / 2
            sl, sm = self._Slocal(L_hdr, L_ldr, window, f)
            if sl < 0:
                self._warn_negative_s_local(lvl + 1, level, sl)

            s_local.append(sl)
            s_maps.append(sm)

            # averaging
            filtered_im1 = convolve(L_hdr, kernel, mode="valid")
            filtered_im2 = convolve(L_ldr, kernel, mode="valid")

            # downsampling
            L_hdr = filtered_im1[::2, ::2]
            L_ldr = filtered_im2[::2, ::2]

        S = np.prod(np.power(s_local, weight))
        return S, s_local, s_maps

    @staticmethod
    def _Slocal(
        img1: np.ndarray,
        img2: np.ndarray,
        window: np.ndarray,
        sf: float,
        C1: float = 0.01,
        C2: float = 10.0,
    ) -> tuple[float, np.ndarray]:
        window = window / window.sum()

        mu1 = convolve(window, img1, "valid")
        mu2 = convolve(window, img2, "valid")

        # Local variance/covariance via a genuine two-pass computation: form each window's
        # deviation from ITS OWN local mean before squaring/multiplying, instead of the
        # textbook-unstable E[X^2]-E[X]^2 / E[XY]-E[X]E[Y] shortcut. That shortcut, for
        # TMQIr's rescaled (~1e9-magnitude) images, subtracts two ~1e19-magnitude
        # quantities and can produce spuriously negative variances/covariances
        # (catastrophic cancellation; see FINDINGS.md). This is the same formula, computed
        # in a numerically stable order -- not an approximation or an algorithm change.
        win1 = sliding_window_view(img1, window.shape)
        win2 = sliding_window_view(img2, window.shape)
        d1 = win1 - mu1[..., None, None]
        d2 = win2 - mu2[..., None, None]

        sigma1_sq = np.sum(window * d1 * d1, axis=(-1, -2))
        sigma2_sq = np.sum(window * d2 * d2, axis=(-1, -2))
        sigma12 = np.sum(window * d1 * d2, axis=(-1, -2))

        sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
        sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))

        CSF = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(-((0.114 * sf) ** 1.1))
        u_hdr = 128 / (1.4 * CSF)
        sig_hdr = u_hdr / 3.0

        sigma1p = norm.cdf(sigma1, loc=u_hdr, scale=sig_hdr)

        u_ldr = u_hdr
        sig_ldr = u_ldr / 3.0

        sigma2p = norm.cdf(sigma2, loc=u_ldr, scale=sig_ldr)

        s_map = (
            (2 * sigma1p * sigma2p + C1)
            / (sigma1p**2 + sigma2p**2 + C1)
            * ((sigma12 + C2) / (sigma1 * sigma2 + C2))
        )
        s = np.mean(s_map)
        return s, s_map

    def _StatisticalNaturalness(self, L_ldr: np.ndarray, win: int = 11) -> float:
        phat1 = 4.4
        phat2 = 10.1
        muhat = 115.94
        sigmahat = 27.99
        u = np.mean(L_ldr)

        # moving window standard deviation using reflected image
        if self.original:
            W, H = L_ldr.shape
            w_extra = (11 - W % 11) % 11
            h_extra = (11 - H % 11) % 11
            # zero padding to simulate matlab's behaviour
            if w_extra > 0 or h_extra > 0:
                test = np.pad(L_ldr, pad_width=((0, w_extra), (0, h_extra)), mode="constant")
            else:
                test = L_ldr
            # block view with fixed block size, like in the original article
            view = view_as_blocks(test, block_shape=(11, 11))
            sig = np.mean(np.std(view, axis=(-1, -2)))
        else:
            # deviation: moving window with reflected borders
            sig = np.mean(generic_filter(L_ldr, np.std, size=win))

        beta_mode = (phat1 - 1.0) / (phat1 + phat2 - 2.0)
        C_0 = beta.pdf(beta_mode, phat1, phat2)
        C = beta.pdf(sig / 64.29, phat1, phat2)
        pc = C / C_0
        B = norm.pdf(u, muhat, sigmahat)
        B_0 = norm.pdf(muhat, muhat, sigmahat)
        pb = B / B_0
        N = pb * pc
        return float(N)


class TMQIr(TMQI):
    """The "revised" TMQI branch: symmetric HDR/LDR rescaling (see FINDINGS.md)."""

    name = "TMQIrev"
    original = False
