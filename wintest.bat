"%PYTHON%\\python.exe"  TMQI.py
"%PYTHON%\\python.exe"  TMQI.py data/test.png data/test_ldr.png    -t png
"%PYTHON%\\python.exe"  TMQI.py data/rgb_test.float32 data/rgb_test_ldr.float32 -i float32 -W 396 -H 561
"%PYTHON%\\python.exe"  TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose
"%PYTHON%\\python.exe"  TMQI.py data/off.png data/off_ldr.png -Q -S -L -N -M --verbose -t png
"%PYTHON%\\python.exe"  TMQI.py data/off.png data/off_ldr.png -q -s -l -n -m --keep --quiet
"%PYTHON%\\python.exe"  TMQI.py https://www.floridamemory.com/fpc/prints/pr76815.jpg https://www.floridamemory.com/fpc/prints/pr76815.jpg
