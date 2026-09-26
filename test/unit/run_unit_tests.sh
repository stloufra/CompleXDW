#!/bin/bash
# Builds and runs every unit test.
# Usage: ./run_unit_tests.sh [compiler, default clang++]

set -uo pipefail
cd "$(dirname "$0")"

CXX="${1:-clang++}"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-math-errno -march=native -I../..)
MPFR=(-I/opt/homebrew/include -L/opt/homebrew/lib -lmpfr -lgmp)
TESTS=(test_arith test_dw test_xdw test_xdw_span test_simd)

mkdir -p exec
failed=()
for t in "${TESTS[@]}"; do
    echo "=================== $t"
    if ! $CXX "${FLAGS[@]}" "$t.cpp" "${MPFR[@]}" -o "exec/$t"; then
        failed+=("$t (build)")
        continue
    fi
    "./exec/$t" || failed+=("$t")
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "All ${#TESTS[@]} unit tests passed ($CXX)"
else
    echo "FAILED: ${failed[*]}"
    exit 1
fi
