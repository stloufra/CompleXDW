#!/bin/bash

CXX="clang++"

echo "Using compiler: $CXX"

OMP_INCLUDE="/opt/homebrew/Cellar/libomp/22.1.1/include"
OMP_LIB="/opt/homebrew/Cellar/libomp/22.1.1/lib"

$CXX -std=c++20 -O3 -march=native -I.. -I../test/src \
    perf_test.cpp \
    ../test/src/test_func.cpp \
    -DXDW_FAST_FMA \
    -I/opt/homebrew/include \
    -I$OMP_INCLUDE \
    -L/opt/homebrew/lib -L$OMP_LIB -lmpfr -lgmp -lm -lomp \
    -o perf_test

./perf_test
