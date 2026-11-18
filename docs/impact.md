# Film impact initialization

Generate the cell-center map with:

    python3 density_maps/generate_impact.py

Default dimensions (nm): domain 20000 x 12000, PS film from y=2000 to
2250 spanning the domain, RDX circle radius 3750 centered at (10000,6250).
The initial gap is 250. Everything else is air (CSV phase "pore").
The 25 nm spacing gives 800 x 480 cells and 10 cells through the film.
Use --spacing, --height, --film-bottom, and --gap to change these choices.

Build and run:

    sh build/build2D.sh mechanics2DMultimaterial
    ./mechanics2DMultimaterial 800 480 20000 12000 1e-4 impact.e 10000 100 0 0 --density-map density_maps/impact.csv --rdx-velocity 0 -750 --open-boundaries --fixed-ps-ends 250

The example runs to 1 ns; absent deceleration, first contact is approximately
250/750 = 0.333 ns. Internal adaptive substeps still enforce the solver limits.
This is an example run configuration, not a completed impact validation.

--rdx-velocity vx vy assigns laboratory velocities in nm/ns to every RDX
map cell. PS and air start at rest. Total energy includes the corresponding
kinetic energy, so all phases initially remain at 300 K. No additional CSV
columns are needed. Chemistry remains enabled unless explicitly disabled.

--open-boundaries uses zero-gradient outflow on all four boundaries and a
stationary mesh. It requires positional piston velocity 0. These boundaries
are not perfectly nonreflecting.

--fixed-ps-ends 250 selects initial PS cells within 250 nm of the left or
right domain edge. These stationary grid patches constrain both momentum
components to zero after each forward-Euler stage of SSP RK3. Removed kinetic
energy is transferred to the support, not converted into heat; thermal and
phase transport remain active. Conservation diagnostics include this support
exchange in the existing boundary-exchange accumulator.

These are fixed velocity patches, not resolved mechanical grips. The current
pressure-only constitutive model does not provide elastic bending or shear
resistance. Full clamped-solid film behavior requires the planned deviatoric
stress/constitutive evolution. Likewise, common initial temperature and each
phase's reference density do not automatically ensure pressure equilibrium.

The existing piston setup and reconstruction are unchanged when these options
are omitted. A Cartesian 2D circle represents an infinitely long cylinder.
