#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
target=${1:-solver2D}
case "$target" in
    solver2D|mechanics2DMultimaterial|mechanics2D|mechanics2DLevelSet|mechanics2DMerge|mechanics2DMergeTHINC|mechanics2DFV|mechanics2DFV_limiter|kinetics2D|kinetics2DFV) ;;
    *) echo "Expected solver2D, mechanics2DMultimaterial, mechanics2D, mechanics2DLevelSet, mechanics2DMerge, mechanics2DMergeTHINC, mechanics2DFV, mechanics2DFV_limiter, kinetics2D, or kinetics2DFV" >&2; exit 1 ;;
esac
if [ -z "${EIGEN_INCLUDE_DIR:-}" ]; then
    if command -v brew >/dev/null 2>&1; then
        EIGEN_INCLUDE_DIR="$(brew --prefix eigen)/include/eigen3"
    else
        EIGEN_INCLUDE_DIR=/usr/include/eigen3
    fi
fi

if [ "$(uname -s)" = Linux ]; then
    # Linux clusters (e.g. Negishi): static Exodus built locally, NetCDF from
    # the loaded netcdf-c module (nc-config also supplies its HDF5/zlib deps).
    EXODUS_ROOT=${EXODUS_ROOT:-"$PWD/.deps/exodus-linux"}
    if [ ! -f "$EXODUS_ROOT/lib/libexodus.a" ]; then
        echo "Missing $EXODUS_ROOT/lib/libexodus.a; build SEACAS Exodus on this machine (see README_2D.md)" >&2
        exit 1
    fi
    if [ -z "${NETCDF_LIBS:-}" ]; then
        if ! command -v nc-config >/dev/null 2>&1; then
            echo "nc-config not found; load the netcdf-c module or set NETCDF_LIBS" >&2
            exit 1
        fi
        NETCDF_LIBS="$(nc-config --libs) -Wl,-rpath,$(nc-config --libdir)"
    fi
    # shellcheck disable=SC2086
    "${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra \
        -I"$EIGEN_INCLUDE_DIR" -I"$EXODUS_ROOT/include" -Iinclude -Iutils "src/$target.C" \
        "$EXODUS_ROOT/lib/libexodus.a" $NETCDF_LIBS -o "$target"
else
    # macOS: fully static Exodus + NetCDF prefix.
    EXODUS_ROOT=${EXODUS_ROOT:-"$PWD/.deps/exodus"}
    if [ ! -f "$EXODUS_ROOT/lib/libexodus.a" ] || [ ! -f "$EXODUS_ROOT/lib/libnetcdf.a" ]; then
        echo "Missing static Exodus/NetCDF libraries under $EXODUS_ROOT; see README_2D.md" >&2
        exit 1
    fi
    "${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra \
        -I"$EIGEN_INCLUDE_DIR" -I"$EXODUS_ROOT/include" -Iinclude -Iutils "src/$target.C" \
        "$EXODUS_ROOT/lib/libexodus.a" "$EXODUS_ROOT/lib/libnetcdf.a" -o "$target"
fi
