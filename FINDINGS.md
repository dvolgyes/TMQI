# Findings

2026-09-04 (updated 2026-09-04, see GitHub issue #3): `S_local`'s covariance ratio,
`(sigma_xy + C2) / (sigma_x*sigma_y + C2)`, is not bounded below by zero — a genuinely anti-correlated HDR/LDR patch
(the same as SSIM's structure term) legitimately pushes it negative. `S = prod(s_local ** weight)` then raises a
negative aggregate `s_local` to a fractional exponent (e.g. `weight[0] = 0.0448`), which is mathematically undefined
(`nan`). **This is an inherent property of the published TMQI formula, present in both `TMQI` and `TMQIr`, not a porting
bug or an implementation error** — confirmed by measuring `sigma_xy` directly on `data/off.png` (`TMQI`, non-revised,
RGB): a handful of individual pixels (5 out of 211750 at level 1) have a genuinely negative local covariance (min
-0.0034), just not enough of them, nor severe enough, to flip that level's *aggregate* `s_local` negative for this
particular image. The official follow-up, **TMQI-II** (K. Ma, H. Yeganeh, K. Zeng, Z. Wang, "High Dynamic Range Image
Compression by Optimizing Tone Mapped Image Quality Index," IEEE Trans. Image Process., vol. 24, no. 10, pp. 3086-3097,
2015, doi:10.1109/TIP.2015.2436340 — from the same research group as the original TMQI paper) revises the
contrast-visibility nonlinearity in the structural-fidelity formula but keeps this exact same covariance ratio
unchanged, so it would not fix this either. No literature erratum for this specific issue was found. A `loguru` warning
(`TMQI._warn_negative_s_local`, `metric.py`) now fires whenever an aggregate `s_local` goes negative, explaining the
cause; the formula itself is deliberately left unchanged, per explicit instruction, to keep scores comparable to
published TMQI/TMQIr values.

`TMQIr` (the "revised" branch, `original=False`) makes this dramatically more likely for a distinct, compounding reason:
it rescales `L_ldr` into the same ~4.3×10⁹ range as `L_hdr` before `_Slocal` runs, so `sigma_xy = E[XY] - E[X]E[Y]` (in
`metric.py:_Slocal`) is computed as the difference of two ~1.6×10¹⁸-magnitude quantities (confirmed by direct
measurement: `ex2 ≈ 1.63072731045629e19`, `mu2² ≈ 1.63071791096818e19` for the *variance* term, which suffers the
identical cancellation) — right at the edge of float64's ~16 significant digits. The ~11×11-tap windowed convolution
then accumulates enough rounding error to produce occasional *spurious* covariance outliers of magnitude ~10³-10⁴
(measured: `s_map` min -1842 at level 1, -1228 at level 2, versus a normal range of roughly [0, 1]) that are not a real
property of the image, just floating-point noise — and which pixels are affected depends on `scipy.signal.convolve`'s
algorithm choice (`direct` vs `fft`/`auto`), so the exact outliers (and thus the exact negative `s_local` value, though
not its sign) are not stable across SciPy versions: `s_local[0]` for `TMQIr` on `data/test.png` (gray) was -0.2138 under
the legacy stack (numpy 1.21.6/scipy 1.7.3) and is -0.014 under the current one (numpy 2.5.2/scipy 1.18.1) — both
negative, both correctly triggering the same downstream `nan`, but not bit-comparable. Practical consequences for
testing: `tests/test_regression.py` asserts `math.isnan(Q)`/`math.isnan(S)` for this case rather than pinning `s_local`,
since the exact values are provably not reproducible.

**Update, 2026-09-04, later the same day — the catastrophic-cancellation part of this is now fixed.** `_Slocal`'s
`sigma1_sq`/`sigma2_sq`/`sigma12` are now computed via a genuine two-pass local statistic: form each 11×11 window's
deviation from *its own* local mean (using `numpy.lib.stride_tricks.sliding_window_view`) before squaring/multiplying,
instead of the textbook-unstable `E[X²]-E[X]²` / `E[XY]-E[X]E[Y]` shortcut that was subtracting two ~10¹⁸–10¹⁹-magnitude
quantities above. This is the *same formula*, computed in a numerically stable order — not an approximation and not an
algorithm change (verified against a `numpy.longdouble`, i.e. higher-precision, reference computation on the exact pixel
identified above: the naive float64 formula gave -18432, the stable reformulation gives -2×10⁻¹² — indistinguishable
from zero — and the longdouble reference gives +451, all three now agreeing in sign; the surviving max error against the
longdouble reference across the whole level-1 map dropped from 24363 to 1128, roughly 22×). Concretely, on
`data/test.png` (gray), `TMQIr`'s previously-`nan` result becomes `Q≈0.7984, S≈0.9885` — a normal, finite value on every
platform tested. `TMQI` (non-revised) gray was never `nan`, but its `Q` shifted from ≈0.2195 to ≈0.2199 for the same
reason (less rounding noise in a case where `S` is tiny and `Q` is correspondingly sensitive to it) — see `README.md`'s
"Known limitations" for the user-facing note that results from before this fix will differ slightly for images that were
hitting this instability. RGB cases and `off.png` are unaffected to within existing precision (they were already
numerically well-conditioned).

This *is* a departure from the earlier "leave the formula untouched" position taken above, made deliberately at the
user's explicit request after empirically confirming the fix works and quantifying its cost: a genuinely stable,
vectorized local covariance is ~9× slower than the `scipy.signal.convolve`-based one (measured: 0.052s → 0.486s per call
for the finest pyramid level), because it materializes the full window at every output pixel instead of using
`convolve`'s FFT/BLAS-backed implementation. The `_warn_negative_s_local` warning is kept (a negative `s_local` is still
a legitimate output of the formula, per the entry above — e.g. `off.png`'s 5 genuinely negative pixels, or the synthetic
inverted-contrast test cases in `test_metric.py`), but its message no longer singles out `TMQIr`'s rescale as a distinct
amplifying cause, since that mechanism is now largely mitigated for both branches.

2026-09-04: The padding bug in `_StatisticalNaturalness` — `w_extra = (11 - W % 11)` without a second `% 11` — pads a
full spurious 11-row/column block of zeros whenever a dimension is an exact multiple of 11. Both original test fixtures
(`test.png`/`test_ldr.png`, 396×561) hit this exactly (396 = 36×11, 561 = 51×11); `off.png`/`off_ldr.png` (395×560) do
not. Fixed to `(11 - W % 11) % 11` per explicit user request; measured effect on `TMQI` (non-revised), RGB, `test.png`:
Q 0.844→0.8496, N 0.3145→0.3429. `off.png` and all grayscale/`TMQIr` cases are unaffected (grayscale N already rounds to
0.0 either way; `TMQIr`'s `_StatisticalNaturalness` uses the `generic_filter` branch, which this padding code does not
touch).

2026-09-04: `img_read()` (formerly module-level in `src/TMQI.py`) called `os`, `imread`, `skimage.color`, and `wget`,
all of which were imported *only* inside the `if __name__ == "__main__":` block. Using the documented library API
(`from TMQI import img_read`) raised `NameError`. Fixed by module-level imports in `image_io.py`.

2026-09-04: The old `setup.py` declared `packages=['TMQI'], package_dir={'TMQI': 'src'}` with no `src/__init__.py`
present — an installed copy could never actually satisfy `from TMQI import TMQI, TMQIr` as the README claims. The new
`src/TMQI/` package layout is what makes that import genuinely work for the first time.

2026-09-04: `s_map` PNG output (`-M -t png`) now uses a fixed `[0, 1]` clip-and-scale instead of the removed
`scipy.misc.imsave`'s implicit per-image min/max byte-scaling. Per Yeganeh & Wang, `S_local` is a product of two
SSIM-like ratios bounded in [0, 1], and the reference project page renders fidelity maps as grayscale where brighter =
higher fidelity — a fixed range is the behavior that matches the paper and keeps maps comparable across images/scales,
at the cost of differing from the historical byte-scaled PNGs. Raw `-t float32/float64` dumps are bit-exact and
unaffected; confirmed directly against a legacy-environment capture (min -0.0034, max 1.0 across 5 pyramid levels).

2026-09-04 (correction of an earlier, wrong measurement made while first writing the test suite): the historical gray
`.float32` fixtures (`data/test.float32`, `data/test_ldr.float32`) are **not** a close match to
`skimage.color.rgb2hsv(...)[..., 2]` computed fresh from the corresponding `.png` files — 163 of 222156 pixels (~0.07%)
differ by up to ~0.0117 (in a [0, 1] range), while the rest match closely. Triple-confirmed as a property of the
*fixture data itself*, not of any dependency or code: (a) `skimage.color.rgb2hsv`'s V-channel is exactly
`max(R, G, B) / 255` to float64 epsilon (`1.11e-16`) against a hand-computed reference, so the conversion formula hasn't
changed across versions; (b) the RGB raw dumps (`rgb_test.float32`) *are* bit-exact against their PNGs (`np.array_equal`
is `True`), ruling out an imageio/Pillow PNG-decode change; (c) the divergence is reproducible and deterministic across
repeated runs. Most likely explanation: the gray `.float32` dump was generated from a not-quite-identical earlier/later
version of the PNG file at some point in the repository's history — unknowable without the original generation script,
and not worth chasing further (Knuth's law). **Practical, concrete consequence discovered here**: this small,
pre-existing divergence was numerically invisible in the legacy environment (scipy 1.7.3/skimage 0.19.3) — the
Makefile's `diff gray1.txt gray2.txt` check passed because `Q` rounded to 0.2195 either way — but under the current
environment (scipy 1.18.1/skimage 0.26.0) it flips the 4th decimal (`Q` 0.2195 for PNG vs 0.2194 for raw), exactly the
"convolve heuristic changed between SciPy versions" risk flagged during planning, now observed concretely.
`tests/test_image_io.py` and `tests/test_cli.py` compare the raw/gray pair with a tolerance rather than exact equality;
the RGB pair, which *is* bit-exact, keeps an exact-match test.

2026-09-04: the gray `test.png` regression cases are also platform-dependent, not just SciPy-version-dependent (caught
by GitHub Actions CI on macOS, `macos-latest`/Apple Silicon, `numpy==2.5.2`/`scipy==1.18.1`, same versions as the Linux
dev environment): `test_tmqi_gray_test_image` got `Q=0.21939`/`S=0.01423` there vs `Q=0.21950`/`S=0.01423` on Linux — a
4th-decimal flip on `Q` from identical code and identical package versions — and `test_tmqir_gray_test_image` (formerly
`_is_nan`) got a normal float (`Q=0.7602`) on macOS where Linux reliably produces `nan`. Root cause: NumPy and SciPy
delegate summation-order-sensitive operations (`scipy.signal.convolve`, `np.mean`, `np.std`, all pervasive in
`_Slocal`/`_StructuralFidelity`) to BLAS/FFT backends, and IEEE 754 addition is not associative — different backends sum
the same numbers in a different order and get a different last bit. Confirmed directly: this repo's Linux wheels link
OpenBLAS built for x86 (`OpenBLAS 0.3.34 DYNAMIC_ARCH Haswell`, via `numpy.show_config()`); official NumPy wheels for
macOS (Apple Silicon, which is what `macos-latest` runners are) link Apple's Accelerate/vecLib instead — a different
vendor's BLAS tuned for a different instruction set (ARM NEON, not x86 AVX). This last-bit noise is normally 16 decimal
digits down and never visible in any reported value; it's only visible here because of the same
catastrophic-cancellation amplification documented above (`E[X²] − E[X]²`/`E[XY] − E[X]E[Y]` subtracting two huge
quantities), plus, for the non-revised `TMQI` case specifically, this particular image's unusually small `S` (~0.014)
making `Q = a·Sᵅ + …` disproportionately sensitive to small absolute changes in `S`.

Fixed at the time by loosening `test_tmqi_gray_test_image`'s `Q`/`S` assertions to `pytest.approx(..., abs=0.005)`
(verified this comfortably covers the observed macOS values) and by no longer pinning a specific `nan`/non-`nan` outcome
in `test_tmqir_gray_test_image` — it checked the warning mechanism only when `nan` actually occurred, and asserted basic
validity otherwise. The reliable, platform-independent regression guard for the negative-`s_local` warning mechanism
itself is a new pair of synthetic tests, `test_metric.py::test_negative_s_local_warns_for_ {original,tmqir}_branch`: a
deliberately, structurally inverted-contrast image pair gives `s_local` values around **-0.9999** (confirmed on Linux;
reasoned to be robust on any platform since it's not a marginal/knife-edge case like the real photo — cross-platform
BLAS noise is many orders of magnitude too small to flip a value that far from zero).

**Superseded, same day:** the numerically stable `_Slocal` reformulation described in the entry above this one directly
addresses the root cause here too (both the SciPy-version and the platform sensitivity stemmed from the same
catastrophic-cancellation amplification). The gray `test.png` regression tests now pin updated golden values (`Q≈0.2199`
for `TMQI`, `Q≈0.7984` non-`nan` for `TMQIr`); the loosened tolerances and the `nan`-fallback branch in
`test_tmqir_gray_test_image` are kept as a safety margin, not because the flip is still expected.
