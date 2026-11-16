# THINC variant of mechanics2DMerge

Build and run:

```sh
sh build/build2D.sh mechanics2DMergeTHINC
./mechanics2DMergeTHINC 251 251 500 500 1e-5 pore_thinc.e 100000 100 5000 100
```

The executable accepts the same positional arguments as `mechanics2DMerge`, plus
an optional `--thinc-beta value` (default 1.6). The pore is
at (Lx/4, Ly/2). Chemistry, material/vacuum pressure equilibration, SSP RK3 stages,
Rusanov upwind flux, conduction, shock inlet, outflow boundaries, timestep limits,
output fields, and diagnostics are unchanged. There are no contact parameters.
The baseline source and executable are retained for direct comparisons.

## Reconstruction change

`include/THINC.h` implements a cell-average-preserving hyperbolic tangent profile
with beta=1.6. It reconstructs monotone mixed-cell material fractions using the two
neighboring cell averages. Pure cells, extrema and flat stencils retain their
piecewise constant traces. Coordinate directions are handled separately, using the
same beta in x and y; this is not a geometric multidimensional VOF method.

For each THINC trace, all seven extensive material quantities are reconstructed as

```
U_face = alpha_face * (U_cell / alpha_cell)
```

This keeps intrinsic density, velocity, composition and specific energy unchanged
by interface reconstruction. Sharpening alpha alone would not do so. All quantities
then enter the same shared face flux; mass, momentum and energy remain conservative.
The alpha transport equation and local pressure-equilibrium calculation are unchanged.
The same profile preserves the cell averages of all extensive variables within the
reconstructed cell, since intrinsic variables are held constant there.

Full-material stencils still use WENO5 with the existing common nonlinear weights.
Boundary stencil closure and inadmissible-state fallback are unchanged. The diagnostic
`face_fallbacks` counts non-WENO interface reconstructions (now including THINC) as
well as invalid face reconstructions; `low_order_steps` still reports whole-step
first-order fallbacks. THINC is not used during those whole-step fallbacks.

THINC background: [Xiao, Honma and Kono, A simple algebraic interface capturing scheme
using hyperbolic tangent function (2005)](https://doi.org/10.1002/fld.975).
This variant uses THINC spatial traces within the existing Rusanov/SSP RK framework,
not a replacement of the complete solver with that paper's transport algorithm.

## Verification

`tests/mechanics2DMergeTHINC.C` checks analytical THINC bounds and cell-average
preservation in both directions, then repeats the baseline conservation, translating
cavity, hot-chemistry, pressure-equilibration, symmetry and WENO convergence tests.

```sh
c++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 -I.deps/exodus/include -Iinclude -Iutils \
    tests/mechanics2DMergeTHINC.C .deps/exodus/lib/libexodus.a .deps/exodus/lib/libnetcdf.a \
    -o /tmp/test_THINC
/tmp/test_THINC
```

`tests/interfaceTransport.C` compares planar material/vacuum transport. Compile it
normally for the baseline, and with `-DTEST_THINC` for this variant. It prints the
interface-width measure, maximum temperature error, maximum pressure error and
whole-step fallback count. The width is the integral of `4*alpha*(1-alpha)` divided
by transverse length, not a 10–90% contour distance.

For 60x6 cells on a 120x12 nm domain, uniform velocity 500 nm/ns, T=300 K and 0.01 ns
of transport with chemistry enabled, this width was 21.29 nm for the baseline and
3.56 nm for THINC. Temperature errors were below 4e-12 K, pressure errors below
2e-13 GPa, and neither run needed whole-step fallback.

A 31x31 reactive pore run on 500x500 nm with radius 100 nm and shock speed 5000 nm/ns
closed by the 0.1 ns output and continued to 0.16 ns without contact. These checks
establish sharper transport and conservation, not convergence of local impact
pressure/temperature. Fine-grid and larger-pore comparisons remain necessary.

The 61x61 version of the same offset-pore test also closed by 0.1 ns and
continued to 0.12 ns, with relative total-energy balance error 1.4e-13.

## Thinner transition on the same mesh

Append `--thinc-beta 2.2` to your existing command, for example:

```sh
./mechanics2DMergeTHINC 200 100 100 50 1e-5 pore_sharper.e 100000 10 7000 10 --thinc-beta 2.2
```

The default remains 1.6. This changes only the THINC reconstruction steepness; it does
not modify the mesh, material-fraction averages directly, source terms or pressure
relaxation. All extensive material traces use the same reconstructed fraction as before.
The program prints beta at startup. Valid inputs are finite numbers with 0 < beta <= 10;
this parser range is not a guarantee of stability or accuracy throughout that range.

In the existing uniform planar translation test on the same mesh:

| beta | Interface-width measure (nm) |
|---:|---:|
| 1.6 | 3.5619 |
| 2.2 | 2.5221 |
| 3.0 | 2.1588 |

These runs preserved temperature within 4e-12 K and pressure within 5e-13 GPa and
needed no whole-step first-order fallback. Start with 2.2: additional steepening gives
diminishing returns in this test. This does not remove all small-volume-fraction tails,
resolve subcell geometry, or guarantee perfect circular symmetry. Monitor closure and
`low_order_steps` on your particular run; frequent fallback can negate sharpening.
`tests/interfaceTransport.C`, compiled with `-DTEST_THINC`, now accepts beta as its
optional first command-line argument for reproducible comparisons.
