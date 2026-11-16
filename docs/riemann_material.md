# Four-quadrant test of the existing THINC solver

Edit `include/riemannMaterial2D.h` for the four density ratios, temperatures,
and velocities. Pressure and energy use the existing material EOS, so this is
not the standard ideal-gas quadrant benchmark. Split coordinates are fractions
of the domain. Units remain nm, ns, K and the existing material units.

```sh
sh build/build2D.sh mechanics2DMergeTHINC
./mechanics2DMergeTHINC 100 100 1000 1000 1e-4 riemann.e 1000 100 0 0 --riemann
```

The last two positional arguments (shock speed and pore radius) are ignored in
this mode. All boundaries are zero-gradient outflow; no driven inlet or pore
is initialized. Chemistry and conduction are disabled. Pure unreacted material
fills all quadrants. Species and material-fraction storage remain in place but
represent fixed composition and full material; the evolving physical equations
are mass, two momenta, and total energy. Existing WENO/Rusanov, SSP RK3,
adaptive stepping, and admissibility fallbacks are reused. THINC is inactive
because there is no material-vacuum interface. Normal cavity runs are unchanged.

This is an implementation test, not a verified analytical reference solution.
Inspect results before outgoing waves reach boundaries, and compare refinements.
