# Density-map initialization

Build and run the existing THINC solver:

```sh
sh build/build2D.sh mechanics2DMergeTHINC
./mechanics2DMergeTHINC 250 250 100 100 1e-5 mapped.e 100000 250 7000 0 --density-map density.csv --thinc-beta 1.6
```

The positional pore radius (zero above) is ignored with `--density-map`.
Without the option, circular-pore initialization is unchanged. Riemann mode and
density-map initialization cannot be combined. Chemistry remains enabled.

CSV format (unquoted header, solver units):

```csv
x,y,rho
0,0,0.00000182
100,0,0.00000182
0,100,0
100,100,0.00000182
```

This small example only illustrates the format; supply enough spatial samples
to resolve your geometry. Samples may appear in any order but must form a
complete rectangular grid with at least two distinct coordinates per axis.
Duplicates, negative/nonfinite densities, and missing grid points are rejected.

The importer samples at FV centers `(i+0.5)*Lx/Nx, (j+0.5)*Ly/Ny` by nearest
neighbor. A tie selects the lower coordinate. Sample coverage extends half the
outer sample spacing beyond each outer coordinate; centers outside coverage
are rejected. Use matching FV cell centers for exact assignment. Nearest-neighbor
resampling of a different grid is not a conservative volume-overlap remap and
can change represented pore area/mass. No interpolation smoothing is applied.

Density means **intrinsic solid density**, as requested:

- Exactly zero: vacuum, zero fraction/mass/momentum/energy.
- Any positive density: full solid (fraction one), that intrinsic density,
  300 K, zero velocity, and the solver's existing unreacted composition.
- Initial energy includes the density-dependent EOS reference energy.
- Invalid EOS temperature/sound-speed states fail with cell coordinates.

There is no small-density cutoff. Very small positive values are material, not
vacuum, and must be admissible for the solid EOS. The material's reference density
is not changed by the map. Non-reference density at 300 K can therefore create
initial pressure and motion even without a driven shock.

For a nonzero shock speed, each left-boundary row uses the density mapped into
its first FV cell as the upstream state. A Rankine-Hugoniot solve including its
upstream pressure and internal energy gives that row's imposed downstream state.
Zero-density inlet rows impose vacuum. These inlet states are held fixed during
the run, as in the original driven boundary. This does not pre-shock the interior
map or reset the inlet using evolving interior density. Shock speed must exceed
the local upstream sound speed. Other boundaries retain the existing outflow BCs.

Validation: importer coverage/duplicate tests; shock jump conditions at 0.98,
1.0 and 1.02 times reference density; short driven-map run with vacuum and varying
solid density; existing mechanics2DMergeTHINC regression suite.
