# 2D reaction–diffusion solver

Build separately from the 1D program:

```sh
./build2D.sh
```

Run:

```sh
./solver2D Nx Ny Lx Ly T_init T_0 dt output.exodus numSteps stride
./solver2D 101 101 100 100 1920 300 1e-5 solution2D.exodus 10000 100
```

Lengths are in nm, time in ns, temperature in K. Nx and Ny count endpoint-inclusive nodes, so dx=Lx/(Nx-1) and dy=Ly/(Ny-1).

The solver uses the existing chemistry and material constants from modelParams.h and computeReactionRates.h, second-order central differences, and the existing SSP RK3 stages. The interior temperature operator is the five-point Laplacian. Reflected ghost temperatures impose zero normal heat flux at all four sides. Corners include both directional contributions. Chemistry remains active at boundaries. The 1D source and executable are unchanged; the older initializeState2D.C draft is also unchanged.

The initial hot region occupies the central 20% in each direction. All other nodes begin at T_0. Species initial values match the 1D solver. A very coarse mesh may not sample the hot patch.

CSV columns: step,time_ns,X,Y,Y1,Y2,Y3,T. Initial state is labeled step=1 at time_ns=0. Exactly numSteps updates are performed; snapshots include the initial state, every stride updates, and the final state. Rows are ordered by y, then x. The 1D plotting script does not directly support these 2D snapshots.

The solver rejects timesteps violating alpha*dt*(1/dx^2+1/dy^2)<=0.5. This is a diffusion SSP restriction, not a guarantee of reaction stability or positivity; chemistry may require smaller dt. Nonpositive/nonfinite stage temperatures and nonfinite completed states are reported as errors.

Validation performed: constant nonreacting temperature, uniform reaction including boundaries, species-source balance, trapezoidally weighted insulated heat conservation, y-uniform reduction through RK stages, second-order convergence on a smooth insulated cosine mode, and a rectangular-grid integration/output smoke test.

## Exodus / ParaView output

Use `.exodus`, `.exo`, or `.e` for a single binary Exodus II file containing the static mesh and all saved times. Use `.csv` to retain the previous CSV output. No numerical equations, initial conditions, boundary conditions, or RK stages change with output format.

In ParaView: open the Exodus file, enable nodal arrays `T`, `Y1`, `Y2`, `Y3` in the reader properties, and click **Apply**. Color by the selected variable and use the animation controls. Coordinates are nm and time values are ns; the file title records the units. Fields are nodal, and adjacent nodes form counterclockwise QUAD4 elements. Snapshot indices in Exodus are consecutive starting at 1; their actual times follow the requested output stride, including initial and final states.

`build2D.sh` links static Exodus and NetCDF libraries from `.deps/exodus`. These local dependencies and generated outputs are ignored by Git. The executable has no runtime dependency on the scratch build directory or ParaView. Set `EXODUS_ROOT` to another static installation and `EIGEN_INCLUDE_DIR` if needed. A system NetCDF build may require additional link libraries; the installed local build uses classic/64-bit-offset NetCDF without HDF5 or remote-data support.

Local dependency versions: NetCDF-C 4.9.3 and the SEACAS Exodus revision recorded in `.deps/exodus/VERSIONS.txt`. On another machine, install/build those libraries or recreate the static prefix using the official NetCDF-C and SEACAS sources. CMake options used:

- NetCDF: `BUILD_SHARED_LIBS=OFF`, `NETCDF_ENABLE_HDF5=OFF`, `NETCDF_ENABLE_DAP=OFF`, `NETCDF_ENABLE_NCZARR=OFF`, `NETCDF_ENABLE_BYTERANGE=OFF`, `NETCDF_ENABLE_REMOTE_FUNCTIONALITY=OFF`, `NETCDF_ENABLE_TESTS=OFF`, `BUILD_TESTING=OFF`.
- SEACAS: `BUILD_SHARED_LIBS=OFF`, `Seacas_ENABLE_SEACASExodus=ON`, `Seacas_ENABLE_Fortran=OFF`, `Seacas_ENABLE_TESTS=OFF`, `TPL_ENABLE_Netcdf=ON`, `TPL_ENABLE_MPI=OFF`, `NetCDF_ROOT` and `netCDF_ROOT` set to the NetCDF installation prefix.

Set `CMAKE_INSTALL_PREFIX` to the same dependency prefix for both, then configure, build, and install each in that order. CMake and the source trees are needed only to rebuild dependencies, not to build/run this solver once the prefix exists.

Exodus validation: rectangular mesh dimensions and connectivity, all saved time values, all nodal arrays compared with CSV, and readback through the installed ParaView Exodus reader.


## 2D mechanical finite-volume solver

`src/mechanics2DFV.C` extends `mechanics1DFV.C` to a rectangular grid. Build and run:

```sh
sh build/build2D.sh mechanics2DFV
./mechanics2DFV 51 31 500 300 1920 300 1e-5 mechanics2DFV.e 100 10 1 20
```

Arguments are `Nx Ny Lx Ly T_init T_0 dt output numSteps stride [viscosity: 0|1] [left_hot_width_nm | --circular [radius_nm]]`.
Use `.csv`, `.e`, `.exo`, or `.exodus` output. Supply the viscosity flag before an optional hot width.
The default initial hot patch is the central 20% of nodes along each axis, as in `mechanics2D.C`.
A hot width instead initializes a left strip `x < left_hot_width_nm`, uniform in y.
To initialize a circular hotspot, replace the hot width with `--circular [radius_nm]`:

```sh
./mechanics2DFV 51 31 500 300 1920 300 1e-5 circle.e 100 10 1 --circular 50
```

The circle is centered at `(Lx/2, Ly/2)` and includes nodes whose distance from that center
is at most the radius. Omitting `radius_nm` uses `0.1*Lx`, matching `solver2D.C`.
Radius is in nm and must be finite and positive. A circle extending beyond the domain is clipped by the grid.


The seven conserved fields are `rhoY1, rhoY2, rhoY3, rho, mx, my, rhoE`.
Each direction reconstructs conserved face states using `WENO::reconstructSystem` in `include/WENO.h`,
then uses the existing `limiter::eval` Rusanov (local Lax–Friedrichs) numerical flux.
WENO5 uses the three standard finite-volume candidate polynomials with nonlinear Jiang-Shu
weights, based on the optimal weights `(0.1, 0.6, 0.3)` and stencil smoothness indicators. At face `i+1/2`, the left state uses
`i-2,...,i+2`; the mirrored right state uses `i-1,...,i+3`. The candidate polynomials are evaluated on each conserved component, with shared nonlinear
weights across all seven fields to preserve `rhoY1+rhoY2+rhoY3=rho`. Pressure, temperature, sound speed, and physical flux are
computed from each reconstructed state. The normal velocity determines the face wave speed;
the transverse momentum is transported too.
Both directional flux divergences enter the same SSP RK3 update (no directional splitting).
The EOS reference energy, temperature recovery (including both kinetic-energy components), reaction heat source, face conduction, and optional compression viscosity follow the 1D FV implementation.

All four sides use zero-gradient outflow: the exterior ghost state is copied from the adjacent
boundary cell. Both velocity components therefore have zero normal gradient at the outer face;
density, composition, and temperature are extrapolated the same way. Both momenta evolve at
boundary cells. Physical mass, momentum, and energy fluxes pass through the boundary, while
outer conductive flux is zero. This is a transmissive extrapolation, not a perfectly nonreflecting
characteristic boundary; it also allows inflow if the local velocity points inward.
Endpoint control volumes retain half width in the corresponding direction, giving quarter-area
corner volumes. The 1D FV solver still uses its original rigid walls.
The centered finite-difference mechanics equations and its shock/piston modes are not part of this FV solver.

CSV columns are `step,X,Y,Y1,Y2,Y3,T,rho,vx,vy,rhoE,p`; Exodus exposes the same nine physical fields on the existing QUAD4 mesh.
As in `mechanics1DFV.C`, the field named `p` is Cauchy stress (negative gauge pressure).
Snapshot numbering also matches that solver: initial step 1 at time zero, step 2 after one update,
then `numSteps` additional updates. Thus final time is `(numSteps+1)*dt`; Exodus stores physical times.
The timestep is fixed, with no automatic CFL selection; 2D transport and diffusion involve both grid spacings.

Numerical regression checks live in `tests/mechanics2DFV.C`. They check
smooth advective composition transport under refinement, with viscosity on/off. They check
weighted mass/energy changes against boundary fluxes, species balance, evolving boundary
momenta, a uniform moving state, axis-exchange symmetry, and all nine Exodus fields on readback. For the local dependency installation:

```sh
c++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 -I.deps/exodus/include -Iinclude -Iutils \
    tests/mechanics2DFV.C .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a \
    -o /tmp/test_mechanics2DFV
/tmp/test_mechanics2DFV /tmp/test_mechanics2DFV.e
```

Adjust the Eigen include path for other installations. Existing solvers and runner scripts are unchanged.


## Matching kinetics and finite-difference variants

The following solvers keep the same standalone `initialConditions`, `assembleResidual`,
`RK3Step`, `writeToFile`, and `main` layout, with fields stored as matrices indexed `(i,j)`:

| Source | Discretization | Evolved fields |
| --- | --- | --- |
| `mechanics2DFV.C` | Rusanov finite volumes | rhoY1, rhoY2, rhoY3, rho, mx, my, rhoE |
| `kinetics2DFV.C` | Finite-volume heat fluxes | Y1, Y2, Y3, T |
| `mechanics2D.C` | Existing finite differences | Y1, Y2, Y3, T, rho, mx, my |
| `kinetics2D.C` | Finite differences, following mechanics2D | Y1, Y2, Y3, T |

Build the added/updated variants:

```sh
sh build/build2D.sh kinetics2DFV
sh build/build2D.sh kinetics2D
sh build/build2D.sh mechanics2D
```

All four support the same hotspot configurations: no hotspot argument selects the central
20% patch in each direction; a numeric width selects a left strip; `--circular [radius_nm]`
selects a centered circle, with default radius `0.1*Lx`. Lengths are in nm.
The kinetics executables take the hotspot argument immediately after `stride`, with no viscosity flag:

```sh
# Default centered patch
./kinetics2DFV 51 31 500 300 1920 300 1e-5 kinetics_patch.e 100 10
# Left strip, width 20 nm
./kinetics2DFV 51 31 500 300 1920 300 1e-5 kinetics_left.e 100 10 20
# Circle, radius 50 nm, using either kinetics discretization
./kinetics2DFV 51 31 500 300 1920 300 1e-5 kinetics_fv_circle.e 100 10 --circular 50
./kinetics2D 51 31 500 300 1920 300 1e-5 kinetics_fd_circle.e 100 10 --circular 50
# Mechanics requires the viscosity flag before the hotspot option
./mechanics2D 51 31 500 300 1920 300 1e-5 mechanics_fd_circle.e 100 10 1 --circular 50
```

`mechanics2D.C` retains its existing finite-difference equations and shock/piston options.
Both kinetics solvers evolve only mass fractions and temperature, with no momentum, density,
pressure, or energy equation. Their CSV columns are `step,X,Y,Y1,Y2,Y3,T`; Exodus contains
`Y1,Y2,Y3,T`. Their snapshot numbering follows `kinetics1DFV.C`: an initial snapshot, one
initial RK update, then `numSteps` additional updates.

The discretizations retain their original boundary behavior:

- `kinetics2DFV.C` extends `kinetics1DFV.C` with insulated x/y face fluxes, half-width
  endpoint cells, and reaction heating at all cells, including corners.
- `kinetics2D.C` uses the same centered interior conduction and one-sided endpoint stencil
  as `mechanics2D.C` and `solver1D.C`. Species reactions are active everywhere, but temperature
  reaction heating is applied only away from all four boundaries, matching `mechanics2D.C`.
  This differs from the reflected-ghost, insulated boundary treatment in `solver2D.C`.

The FD and FV variants therefore do not impose identical thermal boundary discretizations.
Existing runner scripts are unchanged. Tests for the added kinetics operators and shared
hotspot configurations can be run after building the four solvers:

```sh
c++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 -I.deps/exodus/include -Iinclude -Iutils \
    tests/kinetics2D.C .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a \
    -o /tmp/test_kinetics2D
/tmp/test_kinetics2D
python3 tests/initial_conditions_2D.py
```


## Nonlinear WENO5 reconstruction in mechanics2DFV

Every x/y advective face reconstruction, including the outflow closure, calls `WENO::reconstructSystem`.
`limiter.h` still supplies the Rusanov flux; WENO reconstructs the states passed to that flux.
The other solvers retain their existing discretizations.

The uniform-grid coefficients require full-width cell averages. Since this solver retains
half-width endpoint volumes, faces whose six-cell stencil touches an endpoint (or extends
outside the grid) use a first-order closure through the same WENO function. Grids with fewer
than eight nodes in a direction have no full interior WENO5 stencil in that direction.
Exterior faces keep the zero-gradient outflow condition.

For each stencil, the Jiang-Shu smoothness metrics beta determine weights
`alpha_k = d_k/(epsilon+beta_k)^2`, normalized so that their sum is one.
Right-side metrics use the mirrored stencil. Each component's five stencil values are divided
by their maximum absolute magnitude before computing beta, with dimensionless `epsilon=1e-6`.
Zero-valued stencils have beta zero. The system reconstruction takes the maximum normalized
beta across scalar components for each candidate. The two momentum components are treated
as a vector: their quadratic indicators are summed and normalized by the same vector magnitude,
so rotating the velocity components does not change the sensor. The resulting common weights
are applied to all conserved fields. Thus a jump in any field can downweight a stencil without breaking the
species-density identity. The scalar `WENO::reconstruct` interface remains available too.

Nonlinear weights suppress discontinuity-crossing stencils but do not guarantee positivity.
A face falls back to its
adjacent cell states if the reconstructed state has invalid density, temperature, sound speed,
or species fractions, or its species inventory would require a negative opposite trace
relative to the cell mean. This local first-order fallback is retained in addition to the
nonlinear weights.
It does not guarantee admissibility of every time update; the timestep still matters.

Conduction and optional compression viscosity retain their original cell-based formulas.
Only smooth interior face reconstruction has the fifth-order property; the complete 2D
reactive-diffusive scheme, boundary closure, and SSP RK3 time integration are not fifth order.

The standalone reconstruction tests verify each candidate on quadratic cell averages,
quadratic reconstruction, left/right symmetry, discontinuous-stencil suppression, field-scale
invariance, species-density consistency, and fifth-order convergence on a smooth profile
away from critical points. Jiang-Shu weights can lose order near smooth critical points.
Run the header tests with:

```sh
c++ -std=c++17 -O2 -Iinclude tests/WENO.C -o /tmp/test_WENO
/tmp/test_WENO
```


## First-order mechanics FV comparison

`mechanics2DFV_limiter.C` is a separate first-order version using adjacent cell states
and the original `limiter::eval` function from `include/limiter.h` on every x/y face.
It has no WENO reconstruction or reconstruction fallbacks. The current outflow boundaries,
hotspot options, reaction sources, energy equation, conduction, optional artificial viscosity,
RK3 updates, and output conventions are retained. The WENO version remains unchanged.

```sh
sh build/build2D.sh mechanics2DFV_limiter
./mechanics2DFV_limiter 201 201 1000 1000 1920 300 1e-4 limiter_circle.e 10000 100 0 --circular 100
```

The `0` before `--circular` disables optional artificial viscosity; use `1` to enable it.
The original Rusanov dissipation in `limiter::eval` is present in either case.


## Circularity check on the Cartesian mesh

`utils/circularity.C` measures the outer threshold contour along 64 rays from the domain center,
using bilinear interpolation on the nodal Exodus grid. It reports the x-axis and 45-degree radii,
the minimum/maximum radii, and `(r_max-r_min)/mean(r)` as a dimensionless spread.
Only contours inside the domain's inscribed disk are measured; `nan` spread means not all rays
have a valid crossing. The threshold and field must match when comparing solvers.

```sh
c++ -std=c++17 -O2 -I.deps/exodus/include utils/circularity.C \
    .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a -o /tmp/circularity_check
/tmp/circularity_check fvm_30_201.e Y3 0.5
```

The supplied `fvm_30_201.e` uses a 201x201 grid with 1.5 nm spacing. At 0.10001 ns,
the Y3=0.5 contour has axis/diagonal radii about 64.38/58.20 nm. This is substantial angular
error, not merely the initial stair-step approximation to the circle. The vector-invariant
momentum sensor removes a genuine component-normalization bias, but matched full-circle
runs show that this correction alone does not remove the WENO contour distortion.
The WENO solver should not be considered validated for circular-front geometry on this mesh.
For the currently tested circle geometry, use the separate `mechanics2DFV_limiter` executable;
its first-order diffusion and front speed still require their own convergence study.


Matched comparison (201x201, 300x300 nm, radius 30 nm, 1920/300 K,
`dt=5e-5 ns`, 2000 total updates, artificial viscosity off), at 0.1 ns and Y3=0.5:

| Reconstruction | Axis radius (nm) | Diagonal radius (nm) | 64-ray relative spread |
| --- | ---: | ---: | ---: |
| Original limiter | 64.342 | 63.963 | 0.590% |
| Nonlinear WENO before vector-sensor correction | 62.098 | 56.528 | 9.345% |
| Nonlinear WENO with vector-sensor correction | 62.071 | 56.578 | 9.406% |

This comparison uses a larger timestep than the supplied Exodus file, but identical timesteps
between the three comparison runs. It measures geometry at one threshold/time, not convergence
of the propagation speed. No radial averaging or enforced symmetry was applied.


## Shock-loaded level-set cavity

The separate `mechanics2DLevelSet` Cartesian cut-cell prototype transports a circular vacuum
cavity, supports shock-speed loading on the left, outflow elsewhere, and a finite contact layer
between opposing cavity surfaces. See [the model, arguments, tests, and limitations](docs/levelset_solver.md).
