%CMD_IN_ENV% python src/TMQI.py
%CMD_IN_ENV% python src/TMQI.py data/test.png data/test_ldr.png    -t png
%CMD_IN_ENV% python src/TMQI.py data/rgb_test.float32 data/rgb_test_ldr.float32 -i float32 -W 396 -H 561
%CMD_IN_ENV% python src/TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose
%CMD_IN_ENV% python src/TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose -t png
%CMD_IN_ENV% python src/TMQI.py data/off.png data/off_ldr.png -q -s -l -n -m --keep --quiet
%CMD_IN_ENV% python src/TMQI.py https://www.floridamemory.com/fpc/prints/pr76815.jpg https://www.floridamemory.com/fpc/prints/pr76815.jpg
