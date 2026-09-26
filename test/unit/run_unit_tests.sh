#!/bin/bash
# Builds and runs every unit test.
# Usage: ./run_unit_tests.sh [compiler, default clang++]

set -uo pipefail
cd "$(dirname "$0")"

CXX="${1:-clang++}"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-math-errno -march=native -I../..)
MPFR=(-I/opt/homebrew/include -L/opt/homebrew/lib -lmpfr -lgmp)
TESTS=(test_arith test_dw test_xdw test_xdw_span test_simd)

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m' GREEN=$'\033[32m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD='' GREEN='' RED='' RESET=''
fi

mkdir -p exec
failed=()
for t in "${TESTS[@]}"; do
    echo
    echo "${BOLD}==================== $t ====================${RESET}"
    if ! $CXX "${FLAGS[@]}" "$t.cpp" "${MPFR[@]}" -o "exec/$t"; then
        failed+=("$t (build)")
        continue
    fi
    "./exec/$t" || failed+=("$t")
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "${GREEN}All ${#TESTS[@]} unit tests passed ($CXX)${RESET}"
else
    echo "${RED}FAILED: ${failed[*]}${RESET}"
    exit 1
fi
