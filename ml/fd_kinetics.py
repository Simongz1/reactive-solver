"""Port of the 2D finite-difference kinetics solver (src/kinetics2D.C).

Same residual, boundary closure, initial condition and SSP-RK3 step as the C++ code,
written so the functions accept numpy arrays or torch tensors.

Layout: state U has shape (..., 4, nx, ny) with channels (Y1, Y2, Y3, T); axis -2 is x
(index i), axis -1 is y (index j), nodes at x = i*dx, as in kinetics2D.C. Units: nm, ns, K.
"""
import numpy as np

# include/modelParams.h
R = 1.987
CV = 2355.592
K_COND = 0.361
RHO = 1820e-9
ALPHA = K_COND / RHO / CV  # nm^2/ns
E1, E2 = 24400.0, 23500.0
Z1, Z2 = 32103.943853583012, 11810.38092425548
A1, B1 = -32002.15963286241, 591.0987132187528
A2, B2 = 7831116.710159273, -722.1193197156482
T_SWITCH = 1736.6  # include/computeReactionRates.h


def _xp(a):
    if type(a).__module__.startswith("torch"):
        import torch
        return torch
    return np


def _cat(xp, parts, axis):
    return xp.cat(parts, dim=axis) if xp is not np else np.concatenate(parts, axis=axis)


def _stack(xp, parts, axis):
    return xp.stack(parts, dim=axis) if xp is not np else np.stack(parts, axis=axis)


def reaction_heat(a, b, T):
    """computeReactionHeat: a below T_SWITCH, a + b (T - T_SWITCH) above."""
    xp = _xp(T)
    return xp.where(T > T_SWITCH, a + b * (T - T_SWITCH), a + 0 * T)


def _second_difference(T, axis, h):
    """conduction() in kinetics2D.C along one axis (-2 for x, -1 for y)."""
    xp = _xp(T)
    n = T.shape[axis]

    def take(start, stop):
        index = [slice(None)] * T.ndim
        index[axis] = slice(start, stop)
        return T[tuple(index)]

    c = ALPHA / (h * h)
    interior = c * (take(2, n) - 2 * take(1, n - 1) + take(0, n - 2))
    low = 0.5 * c * (-7 * take(0, 1) + 8 * take(1, 2) - take(2, 3))
    high = 0.5 * c * (-7 * take(n - 1, n) + 8 * take(n - 2, n - 1) - take(n - 3, n - 2))
    return _cat(xp, [low, interior, high], axis)


def residual(U, dx, dy):
    """assembleResidual in kinetics2D.C: dU/dt for (Y1, Y2, Y3, T)."""
    xp = _xp(U)
    Y1, Y2, T = U[..., 0, :, :], U[..., 1, :, :], U[..., 3, :, :]
    r1 = Z1 * xp.exp(-E1 / (R * T))
    r2 = Z2 * xp.exp(-E2 / (R * T))
    w1, w2 = r1 * Y1, r2 * Y2
    heat = (reaction_heat(A1, B1, T) * w1 + reaction_heat(A2, B2, T) * w2) / CV
    # No reaction heating on wall nodes (C++: only when !xWall && !yWall).
    heat = heat * interior_mask(T)
    dT = _second_difference(T, -2, dx) + _second_difference(T, -1, dy) + heat
    return _stack(xp, [-w1, w1 - w2, w2, dT], -3)


def interior_mask(T):
    xp = _xp(T)
    mask = xp.zeros_like(T)
    mask[..., 1:-1, 1:-1] = 1
    return mask


def rk3_step(U, dt, dx, dy):
    """RK3Step in kinetics2D.C (Shu-Osher SSP-RK3)."""
    U1 = U + dt * residual(U, dx, dy)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * residual(U1, dx, dy)
    return U / 3.0 + 2.0 / 3.0 * U2 + 2.0 / 3.0 * dt * residual(U2, dx, dy)


def advance(U, time, dt, dx, dy):
    """Advance by `time` with fixed RK3 steps of at most dt (last step shortened)."""
    steps = max(1, int(np.ceil(time / dt - 1e-9)))
    h = time / steps
    for _ in range(steps):
        U = rk3_step(U, h, dx, dy)
    return U


def circular_hotspot(nx, ny, dx, dy, radius, T_hot, T_cold=300.0, burnt_core=False, center=None):
    """initialConditions in kinetics2D.C with --circular: hot unreacted disc (Y1 = 1).

    burnt_core=True instead starts the disc fully reacted (Y3 = 1), like an ignited
    region from a subgrid surrogate. center is an optional (x, y) offset in nm from the
    domain center.
    """
    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    ox = i * dx - 0.5 * (nx - 1) * dx - (center[0] if center else 0.0)
    oy = j * dy - 0.5 * (ny - 1) * dy - (center[1] if center else 0.0)
    hot = ox * ox + oy * oy <= radius * radius
    U = np.zeros((4, nx, ny))
    U[0] = 1.0
    if burnt_core:
        U[0][hot] = 0.0
        U[2][hot] = 1.0
    U[3] = np.where(hot, T_hot, T_cold)
    return U
