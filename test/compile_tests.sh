#!/bin/bash

if [ "$1" == "gcc" ] || [ "$1" == "g++-14" ]; then
    CXX="g++-14"
elif [ "$1" == "clang" ]; then
    CXX="clang++"
else
    CXX="clang++"
fi

echo "Using compiler: $CXX"

mkdir -p exec res

# Compile conditioning test
$CXX -std=c++20 -O3 -march=native -I.. \
    test_complex_dw_conditioning.cpp \
    -DXDW_FAST_FMA \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o exec/test_complex_dw_conditioning

# Compile function tests
$CXX -std=c++20 -O3 -march=native -I.. \
    test_complex_dw_functions.cpp \
    -DXDW_FAST_FMA \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o exec/test_complex_dw_functions

# Compile the long-running binned conditioning sweep (meant for a multi-day cluster job)
$CXX -std=c++20 -O3 -march=native -I.. \
    test_complex_dw_conditioning_binned.cpp \
    -DXDW_FAST_FMA \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o exec/test_complex_dw_conditioning_binned

# Re-runs a worst case stored in res/binned_results.csv
$CXX -std=c++20 -O3 -march=native -I.. \
    test_complex_dw_conditioning_replay.cpp \
    -DXDW_FAST_FMA \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o exec/test_complex_dw_conditioning_replay

# Examines the conditioning generator (generate_abcd_mp), writes res_cond/
$CXX -std=c++20 -O3 -march=native -I.. \
    test_conditioning_generator.cpp \
    -DXDW_FAST_FMA \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o exec/test_conditioning_generator
