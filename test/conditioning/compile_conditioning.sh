#!/bin/bash
# Builds the binned conditioning runs and their replays into exec/. Run them from this directory:
# they read and write res/ here.
# Usage: ./compile_conditioning.sh [compiler, default clang++]

set -euo pipefail
cd "$(dirname "$0")"

CXX="${1:-clang++}"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-math-errno -march=native -I../.. -Isrc)
MPFR=(-I/opt/homebrew/include -L/opt/homebrew/lib -lmpfr -lgmp -lm)

mkdir -p exec res
for tool in xdw_binned_mul xdw_binned_div xdw_replay_mul xdw_replay_div; do
    $CXX "${FLAGS[@]}" "$tool.cpp" src/test_func.cpp "${MPFR[@]}" -o "exec/$tool"
done
echo "Built exec/: xdw_binned_{mul,div}, xdw_replay_{mul,div}"
