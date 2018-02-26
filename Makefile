#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python$(PYVERSION) -m coverage
endif

test:
	@$(COVERAGE) erase
	$(COVERAGE) run -a --source . TMQI.py
	$(COVERAGE) run -a --source . TMQI.py data/test.png || echo "Intentionally broken execution."
	$(COVERAGE) run -a --source . TMQI.py data/test.png data/test_ldr.png -g -t png | tee gray1.txt
	$(COVERAGE) run -a --source . TMQI.py data/test.png data/test_ldr.png    -t png | tee color1.txt
	$(COVERAGE) run -a --source . TMQI.py data/test.float32 data/test_ldr.float32         -i float32 -W 396 -H 561 -g | tee gray2.txt
	$(COVERAGE) run -a --source . TMQI.py data/rgb_test.float32 data/rgb_test_ldr.float32 -i float32 -W 396 -H 561    | tee color2.txt
	$(COVERAGE) run -a --source . TMQI.py data/off.png data/off_ldr.png
	@diff -q gray1.txt  gray2.txt
	@diff -q color1.txt color2.txt
