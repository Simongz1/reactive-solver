# MOOSE input matching kinetics1D.C

`kinetics1D.i` is a standalone input using standard MOOSE objects; no custom application
kernels are required. Run with a MOOSE-based executable that registers the framework objects:

```sh
/path/to/moose-app-opt -i moose/kinetics1D.i --check-input
/path/to/moose-app-opt -i moose/kinetics1D.i
```

MOOSE was not available for execution during development. The object names and syntax
were checked against the official documentation; the generated expressions and RK3
updates were numerically compared with the actual `src/kinetics1D.C` implementation.
This is **not** a claim of a successful MOOSE runtime test.

## Why this input uses scalar variables

An ordinary finite-element heat equation input would not exactly reproduce the source
solver's nodal equations. In particular, the code's temperature endpoint equations are
not a conventional Robin boundary condition, despite the source comment, and they
**overwrite reaction heating** at the two endpoints.

This input instead represents the finite-difference method-of-lines system directly:
there are four scalar unknowns per original grid node, `Y1_i`, `Y2_i`, `Y3_i`, `T_i`.
Each has an `ODETimeDerivative` and a `ParsedODEKernel`. Scalar kernels have no spatial
quadrature or finite-element mass matrix approximation. The dummy one-element mesh is
MOOSE bookkeeping only: physical positions are `x_i=i*L/(N-1)`.

This is an exact nodal-equation verification input, not a scalable finite-element PDE
implementation. Large N creates many scalar variables/kernels and may have significant
startup/memory cost. The CSV contains one column per scalar variable, rather than
spatial Exodus fields. A conventional four-field FE model would be a separate
approximation and would require an explicit choice for the endpoint treatment.

## Equations and units

For every node:

```
dY1/dt = -r1(T)*Y1
dY2/dt =  r1(T)*Y1 - r2(T)*Y2
dY3/dt =  r2(T)*Y2
r1(T) = Z1*exp(-E1/(R*T))
r2(T) = Z2*exp(-E2/(R*T))
```

At interior nodes:

```
dT_i/dt = (Q1(T_i)*r1(T_i)*Y1_i + Q2(T_i)*r2(T_i)*Y2_i)/cv
          + alpha*(T_(i+1)-2*T_i+T_(i-1))/dx^2
```

`Qj(T)=aj` for `T<=1736.6`; otherwise `Qj(T)=aj+bj*(T-1736.6)`.
`alpha=k/(rho*cv)`. All constants are read by the generator from `include/modelParams.h`.
The units remain those of the original solver: length in nm, time in ns, temperature in K.
No mechanics, advection, density evolution or species diffusion is added.

The endpoint equations are exactly:

```
dT_0/dt     = alpha/(2*dx^2)*(-7*T_0 + 8*T_1     - T_2)
dT_(N-1)/dt = alpha/(2*dx^2)*(-7*T_(N-1) + 8*T_(N-2) - T_(N-3))
```

Species still react at the endpoints; their temperature equations omit the reaction
source, just as the C++ code does. This may not be the intended physical boundary
model, but it has deliberately not been corrected in this transcription.
`ParsedODEKernel` takes the **left-hand-side** source, so each generated expression
is the negative of the corresponding C++ RHS. Keep its default `implicit=true`:
`ExplicitSSPRungeKutta` evaluates the current stage itself, as in MOOSE's scalar SSPRK test.
Order 3 with a diagonal scalar time-derivative matrix gives the source's SSP RK3 update.

## Default case and regeneration

The included input uses N=101, L=250 nm, T_hot=1920 K, T_cold=300 K,
dt=1e-5 ns, source numSteps=100000, stride=250, left hot width=20 nm.
These are representative values from `run_solver1D.py`, with N=101 rather than its
resolution sweep. Generate the first N=1000 case, for example:

```sh
python3 moose/generate_kinetics1d.py --nodes 1000 --length 250 \
  --hot 1920 --cold 300 --dt 1e-5 --steps 100000 --stride 250 \
  --left-hot-width 20 --output moose/kinetics1D_1000.i
```

Use `--left-hot-width 0` for the original centered IC: hot indices are
`floor(2*N/5) <= i < floor(3*N/5)`. Positive width uses the strict condition `x_i<width`.
Initial composition is Y1=1, Y2=Y3=0 using the header defaults.
Regenerate after changing header constants. The emitted `.i` is self-contained;
it does not read the C++ headers at runtime.

The C++ main advances one step before its numSteps loop. Consequently, the MOOSE
input advances `--steps+1` steps as well. Its initial time is zero; source CSV step s
represents time `(s-1)*dt`. MOOSE CSV sampling uses ordinary multiples of stride plus
initial/final output, while the C++ CSV additionally writes the first step and then
uses an offset sampling schedule. Compare physical times, not row or step numbers.
No adaptive timestep cutting is requested; dtmin=dt. The explicit stability restrictions
of the original solver still apply. Output base can be overridden with
`Outputs/file_base=my_case`.

## Verification

```sh
python3 tests/test_moose_kinetics1d.py
```

This compiles a small harness including the original C++ source, then compares every
generated nodal residual and 20 SSP RK3 steps on a nonuniform test state. It checks both
reaction-heat branches, the exact transition temperature, endpoint source omission,
centered/left initial conditions, and reproducibility of the included input.
It evaluates the generated mathematical expressions in Python, not in MOOSE's parser.

Official references:

- [ParsedODEKernel](https://mooseframework.inl.gov/moose/source/scalarkernels/ParsedODEKernel.html)
- [ExplicitSSPRungeKutta](https://mooseframework.inl.gov/source/timeintegrators/ExplicitSSPRungeKutta.html)
