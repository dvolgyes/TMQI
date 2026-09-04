Tone Mapped Image Quality Index - revised
=========================================

CI: [![CI](https://github.com/dvolgyes/TMQI/actions/workflows/ci.yml/badge.svg)](https://github.com/dvolgyes/TMQI/actions/workflows/ci.yml)
Codecov: [![codecov](https://codecov.io/gh/dvolgyes/TMQI/branch/master/graph/badge.svg)](https://codecov.io/gh/dvolgyes/TMQI)

This is a Python 3 reimplementation of the Tone Mapped Image Quality Index. Requires Python 3.10+.

This implementation and the Matlab original have significant differences
and they yield different results!

The original article can be found here: https://ieeexplore.ieee.org/document/6319406/

The reference implementation in Matlab: https://ece.uwaterloo.ca/~z70wang/research/tmqi/

The original source code does not specify license, except that the code should be referenced
and the original paper should be cited.
I put this re-implementation under AGPLv3 license, hopefully this is compatible
with the original intention. The test photos are taken by me, and I donate them to public domain.

Deviations
----------

I disagree with some implementation choices from the original article, e.g.

- zero padding during block processing
- the rescaling of the input images dynamic range
- (maybe something else, not yet sure)

These leads to different TMQI scores, so the values from the original articles
and from this implementation are NOT comparable. Be careful before you choose one of them.
You can call both the original code and my modified one, using appropriate
function calls (TMQI vs. TMQIr) or using the --revised option in CLI.

Install
-------

```
uv add git+https://github.com/dvolgyes/TMQI
```

or with pip:

```
pip install git+https://github.com/dvolgyes/TMQI
```

Afterwards, you can import it as a library:
```
from TMQI import TMQI, TMQIr
```

or call it as a command line program:
```
tmqi -h
```
(equivalently, `python -m TMQI -h`)

Development
-----------

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```
uv sync --group dev
uv run pytest -n 8 --cov
uv run pre-commit run --all
```

Changes since the 0.x series
-----------------------------

This release modernizes the packaging (`pyproject.toml` + uv, Python 3.10+, no more separate
Windows/Linux requirements files) and the CLI (now built on `click` instead of `optparse`). A few
behaviors changed along the way:

- The console command is now `tmqi` (or `python -m TMQI`), not `TMQI.py`.
- Images must be local files; URL inputs (and the `--keep` flag that went with them) are no longer
  supported.
- Missing or invalid arguments now exit with status 2 and the error on stderr (previously exit 0
  with the message on stdout).
- `-i/--input_type` without `-W/--width` and `-H/--height` now raises a clear usage error instead of
  crashing.
- `--logfile` and `--loglevel` were added for status/diagnostic output (via loguru, on stderr); the
  single-line report is still the only thing printed to stdout.
- `-M`'s non-raw map output (e.g. `-t png`) now scales the fixed `[0, 1]` structural-fidelity range
  to `[0, 255]`, rather than the previous per-image min/max byte-scaling — this matches how the
  original paper defines and visualizes the structural-fidelity map, at the cost of no longer being
  byte-identical to historical output. Raw `-t float32`/`float64` dumps are unaffected.
- A padding bug in the naturalness computation was fixed: when an image's width or height was an
  exact multiple of 11, a full spurious block of zero padding was added instead of none. This
  changes the `N` (and therefore `Q`) score for such images; `S` is unaffected. See `FINDINGS.md`.
- The `PyContracts`-based runtime validation was replaced with plain `ValueError`/`TypeError` guard
  clauses (the library is unmaintained and does not support recent Python versions); the checks
  performed are the same.

See `FINDINGS.md` for other things discovered along the way.

Documentation
-------------
I don't have much time, take a look into the source code.
