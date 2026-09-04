# Findings

2026-09-04: `TMQIr._Slocal`'s variance formula (`E[X²] − (E[X])²`, `metric.py:_Slocal`) suffers
catastrophic cancellation on grayscale inputs. `TMQIr` (the "revised" branch, `original=False`)
rescales `L_ldr` into the same ~4.3×10⁹ range as `L_hdr` before `_Slocal` runs; squaring that range
pushes `E[X²]` to ~1.6×10¹⁹, right at the edge of float64's ~16 significant digits, while the true
local variance is ~5-9 orders of magnitude smaller. Confirmed by direct measurement: at one sample
pixel, `ex2 ≈ 1.63072731045629e19` and `mu2² ≈ 1.63071791096818e19` — the surviving difference
(the variance) has only ~5 significant digits left. 0.2-0.5% of pixels round to a spuriously
*negative* variance (impossible for a true variance), and which pixels do so depends on
`scipy.signal.convolve`'s method (`direct` gave 0.47% negative, `fft`/`auto` gave 0.23%, on
identical input data). Since `sigma2 = sqrt(max(sigma2_sq, 0))`, the clipped/unclipped pixels differ
between SciPy versions, and that difference propagates through `norm.cdf` and the level-1 pyramid
average enough to move `s_local[0]` from -0.2138 (legacy: numpy 1.21.6/scipy 1.7.3) to -0.014
(current: numpy 2.5.2/scipy 1.18.1) for `TMQIr` on `data/test.png` (gray). By contrast the RGB path
(`_RGBtoY`, a stable weighted sum) and `TMQI`'s (non-revised) gray path — which never rescales
`L_ldr` to that range — reproduce the legacy baseline to 4-decimal precision.

This is a latent fragility in the *original* algorithm (the formula is unchanged from the pre-port
code); it is not a porting bug, and per the surgical-changes principle it is not being "fixed" here
(that would mean adopting a numerically stable two-pass variance formula, an unrequested algorithmic
change). Practical consequences:
- `Q`/`S` are `nan` for this case in both the legacy and current environment (the NaN itself is
  stable — it comes from a negative `s_local` raised to a fractional power, a separate, reproducible
  effect — see the entry below), so `tests/test_regression.py` can and does assert that.
- The exact `s_local`/`s_maps` values for `TMQIr` on grayscale inputs are **not** pinned as golden
  regression values; they are provably not reproducible across SciPy versions. Only structural
  properties (shape, that N matches, that Q/S are nan) are asserted for that one case.

2026-09-04: `TMQIr` on `data/test.png` (gray) legitimately computes `Q = S = nan`: `s_local[0]`
(level-1 structural fidelity) goes negative — a real, non-degenerate output of the SSIM-like
covariance term in `_Slocal` — and `S = prod(power(s_local, weight))` raises a negative base to a
fractional exponent (`weight[0] = 0.0448`). This reproduces on both the legacy stack and the ported
one (see previous entry for why the exact negative value itself isn't stable). Treated as a genuine
algorithm output to preserve, not a bug to suppress.

2026-09-04: The padding bug in `_StatisticalNaturalness` — `w_extra = (11 - W % 11)` without a second
`% 11` — pads a full spurious 11-row/column block of zeros whenever a dimension is an exact multiple
of 11. Both original test fixtures (`test.png`/`test_ldr.png`, 396×561) hit this exactly (396 = 36×11,
561 = 51×11); `off.png`/`off_ldr.png` (395×560) do not. Fixed to `(11 - W % 11) % 11` per explicit user
request; measured effect on `TMQI` (non-revised), RGB, `test.png`: Q 0.844→0.8496, N 0.3145→0.3429.
`off.png` and all grayscale/`TMQIr` cases are unaffected (grayscale N already rounds to 0.0 either
way; `TMQIr`'s `_StatisticalNaturalness` uses the `generic_filter` branch, which this padding code
does not touch).

2026-09-04: `img_read()` (formerly module-level in `src/TMQI.py`) called `os`, `imread`,
`skimage.color`, and `wget`, all of which were imported *only* inside the `if __name__ ==
"__main__":` block. Using the documented library API (`from TMQI import img_read`) raised
`NameError`. Fixed by module-level imports in `image_io.py`.

2026-09-04: The old `setup.py` declared `packages=['TMQI'], package_dir={'TMQI': 'src'}` with no
`src/__init__.py` present — an installed copy could never actually satisfy `from TMQI import TMQI,
TMQIr` as the README claims. The new `src/TMQI/` package layout is what makes that import genuinely
work for the first time.

2026-09-04: `s_map` PNG output (`-M -t png`) now uses a fixed `[0, 1]` clip-and-scale instead of the
removed `scipy.misc.imsave`'s implicit per-image min/max byte-scaling. Per Yeganeh & Wang, `S_local`
is a product of two SSIM-like ratios bounded in [0, 1], and the reference project page renders
fidelity maps as grayscale where brighter = higher fidelity — a fixed range is the behavior that
matches the paper and keeps maps comparable across images/scales, at the cost of differing from the
historical byte-scaled PNGs. Raw `-t float32/float64` dumps are bit-exact and unaffected; confirmed
directly against a legacy-environment capture (min -0.0034, max 1.0 across 5 pyramid levels).
