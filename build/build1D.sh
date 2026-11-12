#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
target=${1:-all}
case "$target" in
    all) targets="kinetics1D kinetics1DFV mechanics1D mechanics1DFV_WENO mechanics1DFV_limiter" ;;
    kinetics1D|kinetics1DFV|mechanics1D|mechanics1DFV|mechanics1DFV_WENO|mechanics1DFV_limiter|solver1D) targets="$target" ;;
    *) echo "Expected all, kinetics1D, kinetics1DFV, mechanics1D, mechanics1DFV_WENO, mechanics1DFV_limiter, mechanics1DFV, or solver1D" >&2; exit 1 ;;
esac
if [ -z "${EIGEN_INCLUDE_DIR:-}" ]; then
    if command -v brew >/dev/null 2>&1; then
        EIGEN_INCLUDE_DIR="$(brew --prefix eigen)/include/eigen3"
    else
        EIGEN_INCLUDE_DIR=/usr/include/eigen3
    fi
fi
for name in $targets; do
    "${CXX:-c++}" -std=c++17 -O3 -I"$EIGEN_INCLUDE_DIR" -Iinclude "src/$name.C" -o "$name"
    echo "Built $name"
done
