# Contact-free material/vacuum solver

Build `sh build/build2D.sh mechanics2DMerge` and run, for example:

```sh
./mechanics2DMerge 251 251 500 500 1e-5 pore_merge.e 100000 100 5000 100
```

Arguments are `Nx Ny Lx Ly dt output numSteps stride shock_speed cavity_radius`.
There are **no contact parameters**. Chemistry is enabled in the executable.
The old `mechanics2DLevelSet` executable/source are separate and unchanged by this implementation.
Use a new output filename to preserve existing results.

## Preserved numerical ingredients

- The same nonlinear WENO-JS5 candidates, shared system weights, and momentum-vector
  smoothness indicators in `include/WENO.h`.
- Upwind local Lax-Friedrichs/Rusanov fluxes in both Cartesian directions.
- SSP RK3 coefficients 1, 3/4–1/4, 1/3–2/3 for the transport/reaction update.
- Centered two-point heat-flux stencil, weighted by material conductivity at the face.
- The same material EOS, reference energy, reaction rates and heat source.
- The existing unreacted Hugoniot left-boundary loading. Input speed is **shock speed**,
  not particle velocity. Zero speed disables the driven state for stationary tests.
- Zero-gradient outflow on right/top/bottom, and on the left when loading is disabled.
  This is approximate absorbing behavior, not an exact nonreflecting boundary.
- Cell-centered grid: dx=Lx/Nx, dy=Ly/Ny. Initial material is unreacted, 300 K, at rest;
  the circular cavity is centered at (Lx/4, Ly/2), closer to the left shock boundary.
  Its radius must be less than min(Lx/4, Ly/2). There is no hotspot.

WENO5 applies to full-material stencils. Stencils that intersect vacuum/mixed material
use `WENO::reconstructSystem(..., false, ...)`, the header's first-order closure.
Invalid face traces also fall back to cell averages. A failed RK step is retried with
first-order fluxes everywhere before reducing dt. All fluxes remain shared by the two
neighboring cells, including when fallback is used. The diagnostic counts expose these
reductions; this is **not globally fifth-order interface transport**.

## What changed

The primary interface field is the evolved material volume fraction `alpha`.
The eight unknowns are

```
alpha*rhoY1, alpha*rhoY2, alpha*rhoY3, alpha*rho,
alpha*rho*vx, alpha*rho*vy, alpha*rhoE, alpha
```

The first seven fields use conservative shared face fluxes, with chemistry as a source.
Momentum and energy fluxes contain `alpha*p` and `alpha*p*u` respectively. The volume
fraction uses `alpha_t + div(alpha*u) = alpha*div(u)`. Its numerical flux uses the same
Rusanov speed and reconstruction as the conserved fields. This preserves the relation
between material mass and occupied volume during uniform translation.

There is no evolving cut-cell aperture, cell agglomeration, contact force, contact work,
velocity penalty, temperature floor, or artificial material-density floor. Geometry is
used once to initialize the circular material fractions. Exact vacuum has zero conserved
fields and never enters the material EOS. Nearest-material velocity extension defines
transport velocities in empty cells only.

After each SSP stage and convex combination, mixed cells undergo **local zero-pressure
material/vacuum equilibration**. At fixed mass, momentum, total energy and composition,
the solver finds the occupied volume for which the material EOS gives p=0. If filling
the entire cell leaves positive pressure, alpha becomes 1 and the ordinary compressible
material equations take over. Only alpha changes in this scalar solve. No mass or energy
is inserted, deleted, clipped, or transferred to a contact potential. This lets the void
fraction disappear and material join without retaining opposing surface identities.

This is a diffuse equilibrium model, **not a repaired sharp level-set/cut-cell method**.
The equilibrium projection is a stiff local closure; the SSP RK3 coefficients do not
establish third-order accuracy for this projection/interface combination. There is no
coupled implicit PDE solve, but there is a local scalar EOS bisection in mixed cells.
Numerical mixing is present and requires refinement studies.

The zero-interface-pressure, negligible-vacuum-mass structure follows the material side
of the Baer–Nunziato equations described in [Gaburro, Castro and Dumbser,
A well balanced diffuse interface method for complex nonhydrostatic free surface flows](https://iris.univr.it/retrieve/4685ae31-9853-4a99-9d56-4093cb968e8d/5_WBDiffuseInterface.pdf),
section 2. This implementation retains energy and reactive JWL thermodynamics; its local
volume equilibration and WENO/RK discretization are not that paper's validated method.

## Outputs and checks

`rho` and `rhoE` are volume-averaged. `rho_material` is intrinsic material density.
`T`, species and velocity are material values; exact empty cells display zeros.
`p` is positive compressive gauge pressure. `material_fraction` is alpha.
`phi=0.5-alpha` is a dimensionless plotting indicator, **not a transported signed distance**.
Plot alpha=0.5 to view the diffuse interface; inspect alpha itself to confirm closure.
There are no contact-force output fields.

A companion `<output>.diagnostics.csv` records mass, total energy, integrated boundary
energy influx, integrated reaction heat, mass/momentum/energy balance errors, min/max
material temperature, location/density/alpha at the hottest cell, void area, internal
substeps, retries, first-order steps, and face reconstruction reductions. Boundary and
reaction integrals use the same RK stage weights as the solution. Min/max include every
nonempty cell, even very small volume fractions.

The command-line dt is an outer interval; internal explicit steps obey wave, conduction,
species-depletion and source-induced thermal-change bounds. Initial output is t=0;
final time is numSteps*dt. Output stride counts outer intervals.

Tests:

```sh
c++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 -I.deps/exodus/include -Iinclude -Iutils \
    tests/mechanics2DMerge.C .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a \
    -o /tmp/test_mechanics2DMerge
/tmp/test_mechanics2DMerge
```

These check uniform translating-cavity thermodynamics, shared-flux balance including
chemistry, species/mass consistency, full-material alpha invariance, axis exchange,
conservative mixed-cell compaction, appreciable high-temperature reaction heating,
and fifth-order smooth full-material spatial convergence.

Successful energy accounting does not alone validate local temperatures or the physical
closure model. Assess grid, timestep and interface diffusion sensitivity, particularly
for larger pores with strong reaction feedback. No quantitative peak-temperature or
fine-grid closure guarantee is implied by the regression tests.

## Development verification runs

The closure runs below used the original centered pore, before the default position
was moved to (Lx/4, Ly/2).

With chemistry enabled and shock speed 5000 nm/ns:

- 31x31 cells, 500x500 nm, radius 100 nm, dt=2.5e-5 ns: zero remaining
  numerical void by 0.15 ns; continued to 0.2 ns. Relative final energy balance
  error approximately 3.6e-13. Largest output-sampled temperature approximately 702 K.
- 31x31 cells, 5000x5000 nm, radius 1000 nm, dt=5e-4 ns: zero remaining
  numerical void by 1.5 ns; continued to 2 ns. Relative final energy balance
  error approximately 1.8e-13.

- 101x101 cells, 500x500 nm, radius 100 nm, dt=5e-5 ns: zero remaining
  numerical void by 0.125 ns; continued to 0.16 ns. Relative final energy balance
  error approximately 3.7e-13. Largest output-sampled temperature approximately 1321 K.
  This differs substantially from the coarse-grid peak: temperature is not yet grid-converged.

The single-step translating-cavity test changes temperature by about 1.7e-13 K.
The smooth transport residual errors decrease by factors of approximately 31.5 and
34.5 under successive refinement, consistent with fifth-order spatial convergence.
These are development checks, not a validated impact-temperature prediction.
