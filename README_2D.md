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
