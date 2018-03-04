#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python$(PYVERSION) -m coverage
endif

RUN=python$(PYVERSION) -m coverage run -a --source src

test:
	$(COVERAGE) erase
	$(RUN) src/TMQI.py
	$(RUN) src/TMQI.py data/test.png || echo "Intentionally broken execution."
	$(RUN) src/TMQI.py data/test.png data/test_ldr.png -g -t png | tee gray1.txt
	$(RUN) src/TMQI.py data/test.png data/test_ldr.png    -t png | tee color1.txt
	$(RUN) src/TMQI.py data/test.float32 data/test_ldr.float32         -i float32 -W 396 -H 561 -g | tee gray2.txt
	$(RUN) src/TMQI.py data/rgb_test.float32 data/rgb_test_ldr.float32 -i float32 -W 396 -H 561    | tee color2.txt
	$(RUN) src/TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose
	$(RUN) src/TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose -t png
	$(RUN) src/TMQI.py data/off.png data/off_ldr.png -q -s -l -n -m --keep --quiet
	$(RUN) src/TMQI.py https://www.floridamemory.com/fpc/prints/pr76815.jpg https://www.floridamemory.com/fpc/prints/pr76815.jpg
	diff -q gray1.txt  gray2.txt
	diff -q color1.txt color2.txt
