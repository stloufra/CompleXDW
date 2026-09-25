#!/bin/bash
# Builds bench_elementwise with and without auto-vectorization, runs both, then plots.
# Usage: ./bench_elementwise.sh [compiler, default clang++]

set -euo pipefail
cd "$(dirname "$0")"

CXX="${1:-clang++}"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -march=native -DNDEBUG -I..)

mkdir -p exec results
$CXX "${FLAGS[@]}" -DBENCH_BUILD='"vec"' bench_elementwise.cpp -o exec/bench_elementwise_vec
$CXX "${FLAGS[@]}" -fno-tree-vectorize -fno-tree-slp-vectorize -DBENCH_BUILD='"scalar"' \
    bench_elementwise.cpp -o exec/bench_elementwise_scalar

./exec/bench_elementwise_vec
./exec/bench_elementwise_scalar

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
python3 plot_elementwise.py
