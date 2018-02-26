#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python$(PYVERSION) -m coverage
endif


default:
	echo "There is nothing to do."

test: Makefile test/*
	$(COVERAGE) erase
	$(COVERAGE) run --source . TMQI.py
	$(COVERAGE) run --source . TMQI.py test/test.png test/test_ldr.png -g -t png | tee gray1.txt
	$(COVERAGE) run --source . TMQI.py test/test.png test/test_ldr.png    -t png | tee color1.txt
	$(COVERAGE) run --source . TMQI.py test/test.float32 test/test_ldr.float32         -i float32 -W 396 -H 561 -g | tee gray2.txt
	$(COVERAGE) run --source . TMQI.py test/rgb_test.float32 test/rgb_test_ldr.float32 -i float32 -W 396 -H 561    | tee color2.txt
	$(COVERAGE) run --source . TMQI.py test/off.png test/off_ldr.png
	$(COVERAGE) combine
	diff -q gray1.txt  gray2.txt
	diff -q color1.txt color2.txt
