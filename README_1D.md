# 1D propagation comparisons

Build all five comparison solvers (Eigen required, no Exodus/NetCDF dependency):

```sh
sh build/build1D.sh
```

Build only one with, for example, `sh build/build1D.sh mechanics1DFV_WENO`.
`CXX` and `EIGEN_INCLUDE_DIR` can override the compiler and Eigen location.

| 1D executable/source | Corresponding 2D solver | State and discretization |
| --- | --- | --- |
| `kinetics1D` | `kinetics2D` | Y1,Y2,Y3,T; finite differences |
| `kinetics1DFV` | `kinetics2DFV` | Y1,Y2,Y3,T; finite-volume heat fluxes |
| `mechanics1D` | `mechanics2D` | Y1,Y2,Y3,T,rho,m; finite differences |
| `mechanics1DFV_WENO` | `mechanics2DFV` | rhoY1,rhoY2,rhoY3,rho,m,rhoE; nonlinear WENO + Rusanov |
| `mechanics1DFV_limiter` | `mechanics2DFV_limiter` | Same conserved state; original limiter with piecewise-constant states |

`kinetics1D.C` is the clearly named FD kinetics counterpart of the existing `solver1D.C`.
The existing `mechanics1D.C`, `kinetics1DFV.C`, and historical `mechanics1DFV.C` sources are
retained. The historical `mechanics1DFV` uses rigid walls; use the new `_limiter` version
for comparison with the current 2D outflow solver. Build the historical executable explicitly
with `sh build/build1D.sh mechanics1DFV` if needed.

## Run a left-ignited planar problem

Common positional arguments are:

```text
N L T_init T_0 dt output.csv numSteps stride
```

Kinetics optionally accepts `left_hot_width_nm` next. Mechanics optionally accepts
`viscosity: 0|1`, followed by `left_hot_width_nm`. Examples for a 300 nm domain and
30 nm left hot interval:

```sh
./kinetics1D          201 300 1920 300 1e-5 kinetics_fd.csv   10000 100   30
./kinetics1DFV        201 300 1920 300 1e-5 kinetics_fv.csv   10000 100   30
./mechanics1D         201 300 1920 300 1e-5 mechanics_fd.csv  10000 100 0 30
./mechanics1DFV_WENO   201 300 1920 300 1e-5 mechanics_weno.csv 10000 100 0 30
./mechanics1DFV_limiter 201 300 1920 300 1e-5 mechanics_limiter.csv 10000 100 0 30
```

Set the mechanics flag to `1` to include artificial viscosity. Rusanov dissipation remains
in both FV mechanics solvers even when artificial viscosity is off.
Omit the hot-width argument for the original centered hot interval (central 20% of nodes).
There is no circular/radial option: these are planar Cartesian 1D equations, without geometric
radial source terms. Length is nm, time ns, temperature K; velocity nm/ns is numerically m/s.

All retain the original SSP RK3 and output numbering: initial state has `step=1` at time zero,
then one initial update is saved as step 2, followed by `numSteps` additional updates.
Use `time_ns=(step-1)*dt`; the final time is `(numSteps+1)*dt`.
Output is CSV (not Exodus):

- Kinetics: `step,X,Y1,Y2,Y3,T`.
- FD mechanics: `step,X,Y1,Y2,Y3,T,rho,v`.
- FV mechanics: `step,X,Y1,Y2,Y3,T,rho,v,rhoE,p`, where `p` retains the negative-gauge-pressure stress convention.

## Boundary conditions and comparisons

The new mechanics FV variants both retain half-width endpoint cells and zero-gradient
outflow, matching the current 2D FV solvers. WENO uses the same shared nonlinear weights,
face admissibility checks, and first-order closure near endpoints. The original limiter
variant uses the adjacent cell states directly. Both retain the 1D EOS/reference energy,
reaction heat source, physical conduction, and optional compression viscosity.

The existing FD mechanics solver retains rigid endpoints. The FD temperature equations
retain the one-sided endpoint conduction stencil and omit reaction heating at endpoints;
FV kinetics retains insulated boundaries with reaction heating everywhere. Thus matching
parameters alone does not make the boundary-value problems identical. Compare before
boundary disturbances reach the region of interest, or use a sufficiently large domain.
For a direct reconstruction comparison, `_WENO` and `_limiter` have matching boundaries
and equations. Removing circular geometry does not remove numerical diffusion or the
need for mesh/timestep convergence.

## Checks

```sh
python3 tests/solvers1D.py
```

`tests/mechanics1DFV.C` compares the new 1D FV mechanics residuals and RK3 updates against
the 2D implementations for a transverse-uniform state, in both directions and including
boundary nodes, with viscosity off/on. Compile with the 2D include/library paths:

```sh
c++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 -I.deps/exodus/include -Iinclude -Iutils \
    tests/mechanics1DFV.C .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a \
    -o /tmp/test_mechanics1DFV_WENO
/tmp/test_mechanics1DFV_WENO
```

Add `-DTEST_LIMITER` and change the output executable name to test the original limiter
variant. Adjust the Eigen include path for your system. Exodus is needed for these tests
because they include the corresponding 2D solver, but not for the 1D executables themselves.
