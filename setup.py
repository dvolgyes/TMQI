#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.util import get_platform


short_description = 'Tone mapped image Quality Index - revised'

with open('requirements.txt', 'rt') as f:
    reqs = list(map(str.strip, f.readlines()))

setup(
    name='tmqi-revised',
    version='0.9.0',
    description=short_description,
    long_description=short_description,
    author='David Volgyes',
    author_email='david.volgyes@ieee.org',
    url='https://github.com/dvolgyes/TMQI',
    packages=['TMQI'],
    package_dir={'TMQI': 'src'},
    scripts=['src/TMQI.py'],
    data_files=[],
    keywords=['tone-mapping', 'image quality', 'metrics'],
    classifiers=[],
    license='AGPL3',
    platforms=[get_platform()],
    require=reqs,
    download_url='https://github.com/dvolgyes/TMQI/archive/latest.tar.gz',
)
