#!/bin/bash
# Builds and runs every unit test; see Makefile for the variables.
# Usage: ./run_unit_tests.sh [compiler] [VAR=value ...], e.g. ./run_unit_tests.sh g++-16 OPT=-O3 CXXFLAGS=-g
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    set -- CXX="$1" "${@:2}"
fi
exec make -C "$(dirname "$0")" test "$@"
