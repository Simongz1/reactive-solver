# Three-phase mechanics solver

```sh
sh build/build2D.sh mechanics2DMultimaterial
./mechanics2DMultimaterial 250 250 250 250 1e-5 phases.e 100000 250 7000 0 --density-map phases.csv
```

The command retains the original positional layout. The final radius is ignored
with a map. The map header is `x,y,rho,phase`, with lowercase `rdx`, `ps`, or `pore`.
Density is intrinsic density in solver units (kg/m^3 multiplied by 1e-9).
All present phases require positive density, including pore air. Zero-density
vacuum maps from the old solver must be converted to an air density and phase.
Maps use the existing rectangular-grid nearest-neighbor mapping and coverage checks.

Initialization sets the specified phase fraction to one, other fractions to zero,
T=300 K and velocity zero. Only RDX carries reaction species; its initial Y1=1.
Internal energy includes the phase EOS reference energy, not just cv*T.

## Equations and closure

The twelve fields are three RDX species partial masses, three phase partial
masses, two mixture momenta, total energy, and three volume fractions.

- Phase mass: d(alpha_k*rho_k)/dt + div(alpha_k*rho_k*u) = 0.
- Volume fraction: d(alpha_k)/dt + u.grad(alpha_k) = 0.
- RDX species sum to the RDX partial mass, not total mixture density.
- One common velocity and temperature; no mechanical pressure equilibration.
- Each phase intrinsic density is its partial mass divided by its volume fraction.
- Mixture pressure is alpha_RDX*(Y1*p_unreacted+(1-Y1)*p_reacted)
  + alpha_PS*p_PS + alpha_air*p_air.
- T is recovered from total internal energy using partial-mass-weighted cv and
  phase reference energies. Acoustic speed uses frozen fractions/composition and
  the common-temperature energy closure, not a pressure-equilibrium Wood formula.
- Chemistry acts only on RDX. The existing rho_init/rho_RDX heat-source convention
  is retained. Conduction retains the original shared k=0.361 for all phases;
  separate phase conductivities are not specified in this implementation.

PS reference density is 1020e-9 and cv=3000. Air cv=1000 and R_specific=287.
Chemistry retains its separate molar R=1.987. PS pressure law retains the supplied
K parameter: compression-positive p=K*(1-rho_PS_reference/rho_PS).
Its reference energy is K/(2*rho_PS_reference)*(1-rho_PS_reference/rho_PS)^2.
The original user definitions of E, G, and K have not been changed.

EOS headers now use pressureRDX, pressurePS, and pressureAir namespaces to coexist.
RDX and PS value() retain tensile-positive stress; the mixture EOS adapter converts
these to compression-positive pressure. Air value() is compression-positive.
Output p is compression-positive (unlike mechanics2DFV's stress-valued p).

## Numerics and boundaries

Single-composition stencils use shared-weight WENO5. Material interfaces use
THINC fractions normalized together; partial masses and energy are rebuilt from
the donor intrinsic state. Invalid traces fall back to cell values. Rusanov fluxes
transport all conserved fields. SSP RK3 uses stage checks, first-order retries,
and timestep reduction. The old solid-vacuum pressure relaxation is removed.
No HLLC, deviatoric stress or plasticity has been added.

The velocity positional argument is now **piston/particle velocity**, not shock
speed. The left face is an impermeable, adiabatic slip piston. Reflected ghost
velocity is 2*v_piston-u. Its numerical traction supplies momentum and exactly
p_wall*v_piston energy work; phase/species mass flux through the wall is zero.
No downstream shock root is calculated, so PS at the boundary is supported.

The Cartesian grid translates uniformly at v_piston (ALE flux F-v_mesh*U).
Initial material is at rest in laboratory coordinates. Cell widths remain fixed;
right/top/bottom retain zero-gradient open boundaries on the translating domain.
The right boundary therefore also moves: it is not a fixed downstream plane,
and may sweep in exterior material using the zero-gradient state. Use a long
enough domain that this boundary assumption does not control the region of interest.
A fixed downstream plane would require a different moving/deforming-grid treatment.

CSV X is the current laboratory coordinate. Exodus stores the initial mesh plus
`disp_x=v_piston*time`, `disp_y=0` fields; apply these displacements in the viewer
to see laboratory positions. Velocity and energy are laboratory quantities.
At zero piston velocity the left wall is stationary and reflecting (not outflow).

Without a map, the circle at (Lx/4,Ly/2) contains air at 1.2e-9; the outside is RDX.
This default air density is a convenience assumption, not taken from a map.
`--thinc-beta value`, `--no-chemistry`, and `--no-conduction` are supported.
The old single-material `--riemann` configuration is not supported here.

Diagnostics track each phase mass, momentum and total energy against boundary
fluxes and reaction heating, plus fraction-sum error. Output includes all three
alphas, partial densities, intrinsic densities and RDX-normalized Y1/Y2/Y3.

## Validation and limits

Tests cover pure-phase and mixed-cell thermodynamics, pressure weighting,
reference-energy derivatives, absence of chemistry in pure PS/air, phase-mass
conservation, fraction sums and piston-work and translating-state checks. A short driven
three-phase-map run also exercises the CLI and output.

This is a common-temperature, frozen-volume-fraction model. Conservative mixing
of different EOS does not in general preserve an initially uniform pressure
exactly. It is not a pressure-equilibrium five-equation model and has not yet been
validated for long-time shock/interface interaction or reactive collapse. THINC
normalization is bounded but can affect interface accuracy; assess mesh convergence.
