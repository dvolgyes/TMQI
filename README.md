# Tone Mapped Image Quality Index - revised

CI:
[![CI](https://github.com/dvolgyes/TMQI/actions/workflows/ci.yml/badge.svg)](https://github.com/dvolgyes/TMQI/actions/workflows/ci.yml)
Codecov:
[![codecov](https://codecov.io/gh/dvolgyes/TMQI/branch/master/graph/badge.svg)](https://codecov.io/gh/dvolgyes/TMQI)
Python: [![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/dvolgyes/TMQI)

This is a Python 3 reimplementation of the Tone Mapped Image Quality Index. Requires Python 3.10+.

This implementation and the Matlab original have significant differences and they yield different results!

The original article can be found here: https://ieeexplore.ieee.org/document/6319406/

The reference implementation in Matlab: https://ece.uwaterloo.ca/~z70wang/research/tmqi/

The original source code does not specify license, except that the code should be referenced and the original paper
should be cited. I put this re-implementation under AGPLv3 license, hopefully this is compatible with the original
intention. The test photos are taken by me, and I donate them to public domain.

## Deviations

I disagree with some implementation choices from the original article, e.g.

- zero padding during block processing
- the rescaling of the input images dynamic range
- (maybe something else, not yet sure)

These leads to different TMQI scores, so the values from the original articles and from this implementation are NOT
comparable. Be careful before you choose one of them. You can call both the original code and my modified one, using
appropriate function calls (TMQI vs. TMQIr) or using the --revised option in CLI.

## Known limitations

- **`Q`/`S` can come out `nan`.** `S_local`'s covariance ratio is not bounded below by zero by formula design (the same
  is true of SSIM's own structure term): a strongly anti-correlated local patch between the HDR and LDR image
  legitimately pushes it negative, and the paper's multi-scale combination (`S = prod(s_local ** weight)`) is undefined
  for a negative base. This is not a bug, and it is not "fixed" here — doing so would make scores incomparable to
  published TMQI/TMQIr values. The official follow-up paper,
  [TMQI-II](https://eceweb.uwaterloo.ca/~z70wang/publications/TIP_TMO.pdf) (K. Ma, H. Yeganeh, K. Zeng, Z. Wang, "High
  Dynamic Range Image Compression by Optimizing Tone Mapped Image Quality Index," *IEEE Trans. Image Process.*, vol. 24,
  no. 10, pp. 3086-3097, 2015, doi:[10.1109/TIP.2015.2436340](https://doi.org/10.1109/TIP.2015.2436340)), revises other
  parts of the structural-fidelity formula but keeps this exact same covariance ratio — so it doesn't avoid this either.
  A warning is logged (at the default `--loglevel`) whenever it happens, explaining why.
- **`TMQIr` (the `--revised` branch) is numerically fragile.** It rescales both the HDR and LDR images into the same
  ~4.3e9 range before computing local covariance, pushing the computation to the edge of float64 precision and making
  the `nan` case above dramatically more likely. Near that edge, the exact numeric output is not reproducible across
  SciPy versions — only the qualitative behavior (e.g. whether the result is `nan`) is stable.
- Both are properties of the published algorithm(s), not implementation bugs. See `FINDINGS.md` for the full, measured
  root-cause writeups; if you need a more numerically robust structural-fidelity term, see TMQI-II above (not
  implemented here, to keep scores comparable to the original TMQI).

## Install

```
uv add git+https://github.com/dvolgyes/TMQI
```

or with pip:

```
pip install git+https://github.com/dvolgyes/TMQI
```

Afterwards, you can import it as a library:

```
from tmqi import TMQI, TMQIr
```

or call it as a command line program:

```
tmqi -h
```

(equivalently, `python -m tmqi -h`)

## Development

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```
uv sync --group dev
uv run pytest -n 8 --cov
uv run pre-commit run --all
```

## Changes since the 0.x series

This release modernizes the packaging (`pyproject.toml` + uv, Python 3.10+, no more separate Windows/Linux requirements
files) and the CLI (now built on `click` instead of `optparse`). A few behaviors changed along the way:

- The console command is now `tmqi` (or `python -m tmqi`), not `TMQI.py`. The Python package/import path is also now
  lowercase (`tmqi`), matching PEP 8 module-naming conventions; the classes inside it (`TMQI`, `TMQIr`) keep their
  original, paper-matching names.
- Images must be local files; URL inputs (and the `--keep` flag that went with them) are no longer supported.
- Missing or invalid arguments now exit with status 2 and the error on stderr (previously exit 0 with the message on
  stdout).
- `-i/--input_type` without `-W/--width` and `-H/--height` now raises a clear usage error instead of crashing.
- `--logfile` and `--loglevel` were added for status/diagnostic output (via loguru, on stderr); the single-line report
  is still the only thing printed to stdout.
- `-M`'s non-raw map output (e.g. `-t png`) now scales the fixed `[0, 1]` structural-fidelity range to `[0, 255]`,
  rather than the previous per-image min/max byte-scaling — this matches how the original paper defines and visualizes
  the structural-fidelity map, at the cost of no longer being byte-identical to historical output. Raw
  `-t float32`/`float64` dumps are unaffected.
- A padding bug in the naturalness computation was fixed: when an image's width or height was an exact multiple of 11, a
  full spurious block of zero padding was added instead of none. This changes the `N` (and therefore `Q`) score for such
  images; `S` is unaffected. See `FINDINGS.md`.
- The `PyContracts`-based runtime validation was replaced with plain `ValueError`/`TypeError` guard clauses (the library
  is unmaintained and does not support recent Python versions); the checks performed are the same.
- A warning is now logged when a run produces `Q`/`S` as `nan`, explaining why. See "Known limitations" above.

See `FINDINGS.md` for other things discovered along the way.

## Documentation

I don't have much time, take a look into the source code.
