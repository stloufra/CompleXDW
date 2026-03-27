#!/bin/bash

if [ "$1" == "gcc" ] || [ "$1" == "g++-14" ]; then
    CXX="g++-14"
elif [ "$1" == "clang" ]; then
    CXX="clang++"
else
    CXX="clang++"
fi

echo "Using compiler: $CXX"

# Compile minimal test
$CXX -std=c++17 -O3 -march=native -I.. \
    minimal_test.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o minimal_test

# Compile full test
$CXX -std=c++17 -O3 -march=native -I.. \
    test_complex_dw_random.cpp \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o test_complex_dw

# Compile conditioning test
$CXX -std=c++17 -O3 -march=native -I.. \
    test_complex_dw_conditioning.cpp \
    src/test_func.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib -lmpfr -lgmp -lm \
    -o test_complex_dw_conditioning
