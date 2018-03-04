Tone Mapped Image Quality Index - revised
=========================================

Travis CI: [![Build Status](https://travis-ci.org/dvolgyes/TMQI.svg?branch=master)](https://travis-ci.org/dvolgyes/TMQI)
Semaphore: [![Build Status](https://semaphoreci.com/api/v1/dvolgyes/tmqi/branches/master/badge.svg)](https://semaphoreci.com/dvolgyes/tmqi)
CircleCI: [![CircleCI](https://circleci.com/gh/dvolgyes/TMQI.svg?style=svg)](https://circleci.com/gh/dvolgyes/TMQI)
AppVeyor: [![Build Status](https://img.shields.io/appveyor/ci/dvolgyes/TMQI.svg)](https://ci.appveyor.com/project/dvolgyes/tmqi)

Coveralls: [![Coverage Status](https://img.shields.io/coveralls/github/dvolgyes/TMQI/master.svg)](https://coveralls.io/github/dvolgyes/TMQI?branch=master)
Codecov: [![codecov](https://codecov.io/gh/dvolgyes/TMQI/branch/master/graph/badge.svg)](https://codecov.io/gh/dvolgyes/TMQI)
Code climate: [![Maintainability](https://api.codeclimate.com/v1/badges/e346fb54948ce29d1ab1/maintainability)](https://codeclimate.com/github/dvolgyes/TMQI/maintainability)

This is a Python2/3 reimplementation of the Tone Mapped Image Quality Index.

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

Install
-------

```
pip install https://github.com/dvolgyes/TMQI
```

Afterwards, you can import it as a library:
```
from TMQI import TMQI
```

or call it as a command line program:
```
TMQI.py -h
```
