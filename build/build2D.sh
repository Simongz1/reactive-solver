#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
EXODUS_ROOT=${EXODUS_ROOT:-"$PWD/.deps/exodus"}
if [ -z "${EIGEN_INCLUDE_DIR:-}" ]; then
    if command -v brew >/dev/null 2>&1; then
        EIGEN_INCLUDE_DIR="$(brew --prefix eigen)/include/eigen3"
    else
        EIGEN_INCLUDE_DIR=/usr/include/eigen3
    fi
fi
if [ ! -f "$EXODUS_ROOT/lib/libexodus.a" ] || [ ! -f "$EXODUS_ROOT/lib/libnetcdf.a" ]; then
    echo "Missing static Exodus/NetCDF libraries under $EXODUS_ROOT; see README_2D.md" >&2
    exit 1
fi
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra \
    -I"$EIGEN_INCLUDE_DIR" -I"$EXODUS_ROOT/include" solver2D.C \
    "$EXODUS_ROOT/lib/libexodus.a" "$EXODUS_ROOT/lib/libnetcdf.a" -o solver2D
