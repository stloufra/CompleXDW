#!/bin/bash
# Builds and runs every unit test.
# Usage: ./run_unit_tests.sh [compiler]  (default: $CXX, else the first of c++, clang++, g++ found)
# MPFR is taken from $MPFR_PREFIX, else Homebrew, else the system paths.

set -uo pipefail
cd "$(dirname "$0")"

if [ $# -gt 0 ]; then
    CXX="$1"
elif [ -z "${CXX:-}" ]; then
    for candidate in c++ clang++ g++; do
        if command -v "$candidate" > /dev/null; then
            CXX="$candidate"
            break
        fi
    done
fi
if [ -z "${CXX:-}" ]; then
    echo "No C++ compiler found: pass one as the first argument or set CXX" >&2
    exit 1
fi
FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-math-errno -march=native -I../..)
MPFR_PREFIX="${MPFR_PREFIX:-$(command -v brew > /dev/null && brew --prefix)}"
MPFR=(${MPFR_PREFIX:+-I$MPFR_PREFIX/include -L$MPFR_PREFIX/lib} -lmpfr -lgmp)
TESTS=(test_arith test_dw test_xdw test_xdw_span test_simd)

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m' GREEN=$'\033[32m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD='' GREEN='' RED='' RESET=''
fi

case "$(uname -s)" in
    Darwin) CPU=$(sysctl -n machdep.cpu.brand_string) ;;
    *)      CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//') ;;
esac
echo "${BOLD}Compiler${RESET}  $($CXX --version | head -n1)"
echo "${BOLD}CPU${RESET}       ${CPU:-unknown}"
echo "${BOLD}Flags${RESET}     ${FLAGS[*]}"

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
