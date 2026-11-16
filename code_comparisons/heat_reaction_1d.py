"""
1D heat conduction coupled to a first-order, temperature-dependent reaction
(no species diffusion, D = 0):

    dT/dt = alpha * d2T/dx2 + beta * k(T) * C
    dC/dt = -k(T) * C
    k(T)  = k0 * exp(-Ea / (R*T)),   beta = -dH / (rho*cp)

Method of lines: central differences in x, stiff BDF integrator in time.
Edit the PARAMETERS block for your problem.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat, identity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- PARAMETERS (SI units) ----------------
L      = 0.01      # domain length [m]
N      = 101       # grid points
alpha  = 1e-7      # thermal diffusivity [m^2/s]
k0     = 1e8       # pre-exponential factor [1/s]
Ea     = 1e5       # activation energy [J/mol]
R      = 8.314     # gas constant [J/(mol K)]
dH     = -1e5      # heat of reaction [J/mol] (negative = exothermic)
rho_cp = 2e6       # volumetric heat capacity [J/(m^3 K)]
C_init = 1000.0    # initial concentration [mol/m^3]
T_init = 300.0     # initial temperature [K]
T_wall = 500.0     # wall temperature [K] (used if BC = "dirichlet")
BC     = "dirichlet"   # "dirichlet" (fixed T at walls) or "insulated"
t_end  = 2000.0    # simulation time [s]
# -------------------------------------------------------

beta = -dH / rho_cp
x = np.linspace(0, L, N)
dx = x[1] - x[0]

def k(T):
    return k0 * np.exp(-Ea / (R * T))

def rhs(t, y):
    T, C = y[:N], y[N:]
    r = k(T) * C
    lap = np.empty(N)
    lap[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / dx**2
    if BC == "insulated":                     # ghost-node zero flux
        lap[0]  = 2*(T[1]  - T[0])  / dx**2
        lap[-1] = 2*(T[-2] - T[-1]) / dx**2
    dT = alpha * lap + beta * r
    if BC == "dirichlet":
        dT[0] = dT[-1] = 0.0                  # walls held at T_wall
    return np.concatenate([dT, -r])

# Sparsity pattern: T couples to neighbours and to local C; C only locally.
tri = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(N, N))
I = identity(N)
sparsity = bmat([[tri, I], [I, I]])

T0 = np.full(N, T_init)
if BC == "dirichlet":
    T0[0] = T0[-1] = T_wall
y0 = np.concatenate([T0, np.full(N, C_init)])

t_eval = np.linspace(0, t_end, 9)
sol = solve_ivp(rhs, (0, t_end), y0, method="BDF", t_eval=t_eval,
                jac_sparsity=sparsity, rtol=1e-6, atol=1e-8)
print(sol.message)

T_sol, C_sol = sol.y[:N], sol.y[N:]
print(f"Adiabatic temperature rise beta*C0 = {beta*C_init:.1f} K")
print(f"Max temperature reached: {T_sol.max():.1f} K")
print(f"Remaining reactant at t_end: {100*C_sol[:, -1].mean()/C_init:.1f} %")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for i, t in enumerate(sol.t):
    a1.plot(x*1e3, T_sol[:, i], label=f"t = {t:.0f} s")
    a2.plot(x*1e3, C_sol[:, i])
a1.set(xlabel="x [mm]", ylabel="T [K]", title="Temperature")
a2.set(xlabel="x [mm]", ylabel="C [mol/m³]", title="Concentration")
a1.legend(fontsize=8)
fig.tight_layout()
fig.savefig("heat_reaction_1d.png", dpi=150)
print("Saved heat_reaction_1d.png")
