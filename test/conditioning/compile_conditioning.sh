#!/bin/bash
# Builds the long-running conditioning studies into exec/. Run them from this directory: they read
# and write res/ and res_cond/ here.
# Usage: ./compile_conditioning.sh [compiler, default clang++]

set -euo pipefail
cd "$(dirname "$0")"

CXX="${1:-clang++}"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-math-errno -march=native -I../.. -I../common -Isrc)
MPFR=(-I/opt/homebrew/include -L/opt/homebrew/lib -lmpfr -lgmp -lm)

mkdir -p exec res res_cond
for tool in xdw_conditioning xdw_binned_mul xdw_binned_div xdw_replay_mul xdw_replay_div xdw_conditioning_generator; do
    $CXX "${FLAGS[@]}" "$tool.cpp" ../common/test_func.cpp "${MPFR[@]}" -o "exec/$tool"
done
echo "Built exec/: xdw_conditioning, xdw_binned_{mul,div}, xdw_replay_{mul,div}, xdw_conditioning_generator"
