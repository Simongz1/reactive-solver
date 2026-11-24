"""Markovian coarse kinetics surrogate: (T, Y) averaged on any uniform grid -> (T, Y) after dt.

The model has no latent or history state. Its inputs are the cell averages of T and Y1..Y3,
the requested step dt and the cell size (hx, hy). It returns the averages at t + dt.
Use it through surrogate.py:

    from surrogate import KineticsSurrogate
    model = KineticsSurrogate("increment_surrogate.pt")
    T_new, Y_new = model.predict(T, Y, dt, h=(hx, hy))       # or extent=(Lx, Ly)

Training data, two sources (the target is always the resolved solution, never coarse FD):
  A. Fine FD runs (src/kinetics2D.C, ported in fd_kinetics.py; float64, stable --dt-fine) of
     random hotspots (discs or slabs, unreacted or burnt cores, background --t-cold), box-
     averaged every --dt-snap onto every box shape (fx, fy) from --factors with aspect
     <= --max-aspect. These teach ignition, burnt cores, merging and curved small fronts, for
     cells of a few to tens of nm.
  B. Synthetic fronts for large cells (--synth-h, default 64 nm - 4 um), which fine FD cannot
     reach. A planar 1D front is run with the same FD (one per background temperature) until it
     is a steady traveling wave; its profile f(u) (u = distance ahead of the front) and speed
     s are kept. A fine field is then f(signed distance to a front geometry), with geometries
     that move exactly at s along their normal (growing burnt discs and their unions, shrinking
     unburnt holes, planar fronts at any angle). Box averages at t and t + dt are the training
     pair. This is accurate when cells and radii are much larger than the flame (~ nm), which
     is exactly the large-cell regime. Fraction of batches: --synth-fraction.

Model (version 2), fully convolutional, conditioned by input channels and FiLM on
  log(dt / dt_ref), log(S_REF dt / hx), log(S_REF dt / hy), delta / hx, delta / hy
(front Courant numbers, i.e. cells crossed per step, and the flame-to-cell ratio,
delta = alpha / S_REF ~ 1 nm, which goes to 0 for large cells instead of growing without
bound like log h). Update in finite-volume form:
  Y1 -> Y1 - xi1, Y2 -> Y2 + xi1 - xi2, Y3 -> Y3 + xi2   (0 <= xi1 <= Y1, 0 <= xi2 <= Y2 + xi1)
  T  -> T + sum_faces c_f (T_neighbour - T) + (Q1 xi1 + Q2 xi2)/cv * (1 + bounded correction)
Face coefficient c_f = lam / (1 + 4 lam) with lam = alpha dt / h^2 + (learned, >= 0), so
c_f <= 1/4 and the heat update is a convex combination of neighbours: heat only flows from hot
to cold, T never drops below its coldest neighbour, and it exceeds the hottest one only by
the heat of reaction released. Zero flux on the domain boundary; species sum and bounds hold
exactly; no heat without reaction. (Version 1 used an unbounded additive learned flux, which
could carry heat from cold to hot and drifted out of bounds; surrogate.py still loads it.)

Training windows of up to --rollout steps feed the network its own output, each sample with
its own dt, plus a consistency loss step(dt1 + dt2) == step(dt1) step(dt2). dt is capped per
cell size so a front at S_REF crosses at most --courant cells per step (the network sees depth
cells per side); surrogate.py sub-steps any larger dt.

Example (FD generation dominates the cost; an existing --data cache is reused as is):
  python ml/train_increment.py --n 512 --factors 4 8 16 32 64 --t-end 3 \
      --trajectories 64 --data increment_data.pt --out increment_surrogate_v2.pt

Units nm, ns, K; 1 nm/ns = 1 m/s.
"""
import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fd_kinetics as fd

T_REF = 300.0      # K, normalization offset
T_SCALE = 1000.0   # K, normalization scale and T loss scale
FLUX_SCALE = 10.0  # version 1 only: K per reference step of the additive learned flux
LAM_SCALE = 0.25   # version 2: scale of the learned (nonnegative) face conductance
REACT_EPS = 0.01   # reacted fraction = max(0, sigmoid - eps) / (1 - eps): exactly 0 when off
HEAT_CORRECTION = 0.2  # largest relative correction of the heat of reaction (Q at the cell mean T)
S_REF = 75.0       # nm/ns, front speed used for the Courant numbers
DELTA = fd.ALPHA / S_REF  # nm, flame length scale alpha / S
MODEL_KIND = "increment"
MODEL_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("A: fine FD runs")
    g.add_argument("--n", type=int, default=256, help="fine nodes per side (divisible by every factor)")
    g.add_argument("--dx", type=float, default=1.0, help="fine node spacing, nm")
    g.add_argument("--dt-fine", type=float, default=0.0005, help="fine RK3 step, ns (stable at dx = 1 nm)")
    g.add_argument("--dt-snap", type=float, default=0.005, help="snapshot spacing, ns; every dt is a multiple of it")
    g.add_argument("--t-end", type=float, default=1.5, help="length of each run, ns")
    g.add_argument("--trajectories", type=int, default=32, help="training runs")
    g.add_argument("--holdout", type=int, default=2, help="extra runs kept for evaluation")
    g.add_argument("--radius", type=float, nargs=2, default=[10.0, 40.0], help="hotspot radius / slab half-width, nm")
    g.add_argument("--t-hot", type=float, nargs=2, default=[1300.0, 2000.0], help="hotspot temperature, K")
    g.add_argument("--t-cold", type=float, nargs=2, default=[300.0, 600.0], help="background temperature, K")
    g.add_argument("--burnt-fraction", type=float, default=0.25, help="fraction of runs whose hotspot starts reacted")
    g.add_argument("--t-hot-burnt", type=float, nargs=2, help="temperature of reacted hotspots, K (default --t-hot)")
    g.add_argument("--planar-fraction", type=float, default=0.25,
                   help="fraction of runs with a planar slab instead of a disc (1D-like fronts)")
    g.add_argument("--factors", type=int, nargs="+", default=[4, 8, 16, 32],
                   help="coarsening factors; box shapes are all pairs (fx, fy) of them")
    g.add_argument("--max-aspect", type=float, default=2.0, help="largest hx/hy of a cell (FD and synthetic)")
    g.add_argument("--gen-device", default="cpu", help="device for the float64 FD runs (cpu or cuda)")
    g.add_argument("--data", type=Path, help="cache file: loaded if it exists, else written after generation")
    g = p.add_argument_group("B: synthetic large-cell fronts")
    g.add_argument("--synth-fraction", type=float, default=0.5, help="fraction of batches from synthetic fronts (0: off)")
    g.add_argument("--synth-h", type=float, nargs=2, default=[64.0, 4000.0], help="cell size range, nm (log-uniform)")
    g.add_argument("--synth-dt-max", type=float, default=5.0, help="largest synthetic step, ns")
    g.add_argument("--synth-cells", type=int, default=32, help="cells per side scored (plus depth cells of margin)")
    g.add_argument("--synth-sub", type=int, default=8, help="sub-samples per cell side for the box averages")
    g.add_argument("--profiles", type=int, default=4, help="1D traveling fronts across --t-cold")
    g.add_argument("--profile-length", type=float, default=800.0, help="1D FD domain for the profiles, nm")
    g = p.add_argument_group("model and training")
    g.add_argument("--width", type=int, default=48)
    g.add_argument("--depth", type=int, default=6, help="3x3 conv layers (receptive field: depth cells per side)")
    g.add_argument("--dt-range", type=float, nargs=2, default=[0.005, 0.5], help="FD training steps, ns")
    g.add_argument("--dt-ref", type=float, default=0.05, help="reference step for normalization, ns")
    g.add_argument("--h-ref", type=float, default=16.0, help="version 1 only: reference cell size, nm")
    g.add_argument("--courant", type=float, default=2.0,
                   help="largest S_REF dt / h trained (front cells crossed per step); larger dt is sub-stepped")
    g.add_argument("--rollout", type=int, default=4, help="steps per training window (reached by curriculum)")
    g.add_argument("--consistency", type=float, default=0.1, help="weight of the step-composition loss")
    g.add_argument("--noise-T", type=float, default=0.0, help="std of temperature noise on window starts, K")
    g.add_argument("--epochs", type=int, default=4000)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, default=Path("increment_surrogate.pt"))
    return p.parse_args()


def box_shapes(factors, max_aspect, n):
    shapes = [(a, b) for a, b in itertools.product(sorted(set(factors)), repeat=2)
              if max(a, b) / min(a, b) <= max_aspect + 1e-9]
    bad = [s for s in shapes if n % s[0] or n % s[1]]
    if bad:
        raise SystemExit(f"--n {n} is not divisible by the box shapes {bad}")
    return shapes


def restrict(U, fx, fy):
    """Box average over fx x fy nodes: (..., C, nx, ny) -> (..., C, nx/fx, ny/fy)."""
    lead = U.shape[:-3]
    x = F.avg_pool2d(U.reshape(-1, *U.shape[-3:]), (fx, fy))
    return x.reshape(*lead, *x.shape[-3:])


def normalize(U):
    return torch.cat([U[:, :3], (U[:, 3:] - T_REF) / T_SCALE], dim=1)


def snaps(time_ns, dt_snap):
    m = round(time_ns / dt_snap)
    if m < 1 or abs(m * dt_snap - time_ns) > 1e-9 * max(1.0, time_ns):
        raise SystemExit(f"{time_ns:g} ns is not a positive multiple of --dt-snap {dt_snap:g} ns")
    return m


def log_uniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


# ---------------------------------------------------------------- A: fine FD data

def initial_state(n, dx, radius, t_hot, t_cold, burnt, planar, center):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    ox = i * dx - 0.5 * (n - 1) * dx - center[0]
    oy = j * dx - 0.5 * (n - 1) * dx - center[1]
    hot = (np.abs(ox) + 0 * oy <= radius) if planar else (ox * ox + oy * oy <= radius * radius)
    U = np.zeros((4, n, n))
    U[0] = 1.0
    if burnt:
        U[0][hot], U[2][hot] = 0.0, 1.0
    U[3] = np.where(hot, t_hot, t_cold)
    return U


def run_trajectory(args, rng, shapes):
    """One hotspot; for every box shape, coarse snapshots every dt_snap."""
    n, dx = args.n, args.dx
    radius = rng.uniform(*args.radius)
    burnt = rng.uniform() < args.burnt_fraction
    planar = rng.uniform() < args.planar_fraction
    t_hot = rng.uniform(*(args.t_hot_burnt if burnt and args.t_hot_burnt else args.t_hot))
    t_cold = rng.uniform(*args.t_cold)
    # random position inside the largest box, so fronts sit anywhere relative to cell edges
    center = tuple(rng.uniform(-0.5, 0.5, size=2) * max(args.factors) * dx)
    U = torch.tensor(initial_state(n, dx, radius, t_hot, t_cold, burnt, planar, center),
                     dtype=torch.float64, device=args.gen_device)
    coarse = {s: [restrict(U, *s).float().cpu()] for s in shapes}
    for k in range(1, snaps(args.t_end, args.dt_snap) + 1):
        U = fd.advance(U, args.dt_snap, args.dt_fine, dx, dx)
        if not torch.isfinite(U).all():
            raise RuntimeError(f"fine FD run became non-finite at t = {k * args.dt_snap} ns; lower --dt-fine")
        for s in shapes:
            coarse[s].append(restrict(U, *s).float().cpu())
    return {"coarse": {s: torch.stack(v) for s, v in coarse.items()}, "radius": radius, "t_hot": t_hot,
            "t_cold": t_cold, "burnt": burnt, "planar": planar}


def load_or_generate(args, shapes):
    if args.data and args.data.exists():
        data = torch.load(args.data)
        if abs(data["dt_snap"] - args.dt_snap) > 1e-12 or abs(data["dx"] - args.dx) > 1e-12:
            raise SystemExit(f"{args.data} was made with dt_snap {data['dt_snap']}, dx {data['dx']}; use a new --data file")
        missing = [s for s in shapes if s not in data["train"][0]["coarse"]]
        if missing:
            raise SystemExit(f"{args.data} has no box shapes {missing}; use a new --data file")
        print(f"loaded {len(data['train'])} + {len(data['test'])} runs from {args.data}")
        return data
    snapshots = snaps(args.t_end, args.dt_snap) + 1
    gb = (args.trajectories + args.holdout) * snapshots * sum(4 * 4 * args.n ** 2 / (a * b) for a, b in shapes) / 1e9
    print(f"generating {args.trajectories + args.holdout} runs, {len(shapes)} box shapes, "
          f"about {gb:.2f} GB of coarse snapshots", flush=True)
    rng = np.random.default_rng(args.seed)
    runs = []
    for k in range(args.trajectories + args.holdout):
        start = time.time()
        runs.append(run_trajectory(args, rng, shapes))
        r = runs[-1]
        print(f"run {k + 1}/{args.trajectories + args.holdout}: {'slab' if r['planar'] else 'disc'} "
              f"r = {r['radius']:.1f} nm, T = {r['t_hot']:.0f} K on {r['t_cold']:.0f} K"
              f"{', burnt core' if r['burnt'] else ''}, {time.time() - start:.0f} s", flush=True)
    data = {"train": runs[:args.trajectories], "test": runs[args.trajectories:],
            "dt_snap": args.dt_snap, "dx": args.dx}
    if args.data:
        torch.save(data, args.data)
    return data


# ---------------------------------------------------------------- B: synthetic large-cell fronts

def residual_1d(U, dx):
    """The kinetics2D.C residual in 1D (no y direction): U (4, n) -> dU/dt."""
    Y1, Y2, T = U[0], U[1], U[3]
    w1 = fd.Z1 * np.exp(-fd.E1 / (fd.R * T)) * Y1
    w2 = fd.Z2 * np.exp(-fd.E2 / (fd.R * T)) * Y2
    heat = (fd.reaction_heat(fd.A1, fd.B1, T) * w1 + fd.reaction_heat(fd.A2, fd.B2, T) * w2) / fd.CV
    heat[0] = heat[-1] = 0.0  # no reaction heating on wall nodes, as in 2D
    return np.stack([-w1, w1 - w2, w2, fd._second_difference(T, -1, dx) + heat])


def traveling_front(t_cold, args):
    """Planar front from a hot slab at x = 0, run until steady; returns its profile and speed."""
    dx, L = args.dx, args.profile_length
    x = np.arange(int(round(L / dx)) + 1) * dx
    U = np.zeros((4, len(x)))
    U[0] = 1.0
    U[3] = np.where(x <= 30.0, 1800.0, t_cold)
    h = args.dt_fine
    t, times, pos = 0.0, [], []
    while True:
        for _ in range(int(round(0.05 / h))):
            U1 = U + h * residual_1d(U, dx)
            U2 = 0.75 * U + 0.25 * U1 + 0.25 * h * residual_1d(U1, dx)
            U = U / 3.0 + 2.0 / 3.0 * U2 + 2.0 / 3.0 * h * residual_1d(U2, dx)
        t += 0.05
        times.append(t)
        pos.append(U[2].sum() * dx)
        if not np.isfinite(U).all():
            raise RuntimeError("1D front run became non-finite; lower --dt-fine")
        if pos[-1] > 0.75 * L:
            break
        if t > 3 * L / 30.0:
            raise RuntimeError(f"no steady front at T_cold = {t_cold:g} K after {t:g} ns")
    times, pos = np.array(times), np.array(pos)
    use = (pos > 0.35 * L) & (pos < 0.7 * L)
    speed = np.polyfit(times[use], pos[use], 1)[0]
    u = x - pos[-1]
    keep = u >= -0.4 * pos[-1]  # drop the region near the initial slab
    u, f = u[keep], U[:, keep]
    # cumulative integral of each channel along u, for averages over a sub-sample width
    C = np.concatenate([np.zeros((4, 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dx, axis=1)], axis=1)
    return {"t_cold": t_cold, "speed": speed, "u0": u[0], "du": dx, "C": C, "f_lo": f[:, 0], "f_hi": f[:, -1],
            "T_burnt": f[3, 0]}


def make_profiles(args):
    profiles = []
    for t_cold in np.linspace(*args.t_cold, args.profiles):
        start = time.time()
        p = traveling_front(t_cold, args)
        profiles.append(p)
        print(f"1D front on {t_cold:.0f} K: {p['speed']:.1f} m/s, burnt T {p['T_burnt']:.0f} K "
              f"({time.time() - start:.0f} s)", flush=True)
    return profiles


class Profile:
    """Box average of a 1D front profile over [u - a, u + a] for any u (torch, on device)."""

    def __init__(self, p, device):
        self.speed, self.t_cold, self.T_burnt = p["speed"], p["t_cold"], p["T_burnt"]
        self.u0, self.du = p["u0"], p["du"]
        self.C = torch.tensor(p["C"], dtype=torch.float32, device=device)
        self.f_lo = torch.tensor(p["f_lo"], dtype=torch.float32, device=device).view(4, 1)
        self.f_hi = torch.tensor(p["f_hi"], dtype=torch.float32, device=device).view(4, 1)
        self.u1 = self.u0 + (self.C.shape[1] - 1) * self.du

    def _cumulative(self, u):
        idx = ((u - self.u0) / self.du).clamp(0, self.C.shape[1] - 1)
        i0 = idx.floor().long().clamp(max=self.C.shape[1] - 2)
        w = idx - i0
        inside = self.C[:, i0] * (1 - w) + self.C[:, i0 + 1] * w
        return inside + self.f_lo * (u - self.u0).clamp(max=0) + self.f_hi * (u - self.u1).clamp(min=0)

    def __call__(self, u, a):
        """u (P,) signed distance ahead of the front (burnt where u < 0) -> (4, P)."""
        return (self._cumulative(u + a) - self._cumulative(u - a)) / (2 * a)


def random_geometry(rng, Dx, Dy, h_min):
    """A front geometry: function (X, Y, t, s) -> signed distance ahead of the front (nm)."""
    D = min(Dx, Dy)
    kind = rng.choice(["discs", "line", "hole", "discs+line"])
    parts = []
    if "discs" in kind:
        for _ in range(rng.integers(1, 4)):
            c = rng.uniform([0, 0], [Dx, Dy])
            r0 = rng.uniform(0.5 * h_min, 0.3 * D)
            parts.append(lambda X, Y, t, s, c=c, r0=r0: torch.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2) - (r0 + s * t))
    if "line" in kind:
        th = rng.uniform(0, 2 * np.pi)
        n = (math.cos(th), math.sin(th))
        p0 = n[0] * Dx / 2 + n[1] * Dy / 2 + rng.uniform(-0.3, 0.3) * D
        parts.append(lambda X, Y, t, s, n=n, p0=p0: n[0] * X + n[1] * Y - p0 - s * t)
    if kind == "hole":
        c = rng.uniform([0.3 * Dx, 0.3 * Dy], [0.7 * Dx, 0.7 * Dy])
        r0 = rng.uniform(0.15, 0.45) * D
        parts.append(lambda X, Y, t, s, c=c, r0=r0: (r0 - s * t) - torch.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2))

    def distance(X, Y, t, s):
        d = parts[0](X, Y, t, s)
        for part in parts[1:]:
            d = torch.minimum(d, part(X, Y, t, s))  # union of burnt regions
        return d
    return distance


def render(geometries, profile, times, N, hx, hy, sub, device):
    """Cell averages (B, 4, N, N) of the profile laid on each geometry at its time."""
    xs = (torch.arange(N * sub, device=device, dtype=torch.float32) + 0.5) * (hx / sub)
    ys = (torch.arange(N * sub, device=device, dtype=torch.float32) + 0.5) * (hy / sub)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    a = 0.5 * max(hx, hy) / sub
    out = []
    for geom, t in zip(geometries, times):
        u = geom(X, Y, float(t), profile.speed).reshape(-1)
        fine = profile(u, a).view(1, 4, N * sub, N * sub)
        out.append(F.avg_pool2d(fine, sub)[0])
    return torch.stack(out)


def synth_dt_bounds(args, h_min):
    hi = min(args.synth_dt_max, args.courant * h_min / S_REF)
    lo = min(max(args.dt_range[0], 0.01 * h_min / S_REF), hi / 4)
    return lo, hi


def synth_batch(args, rng, profiles, K, device):
    """A batch of synthetic large-cell windows: initial state and K (dt, target) steps."""
    profile = profiles[rng.integers(len(profiles))]
    hx = log_uniform(rng, *args.synth_h)
    hy = float(np.clip(hx * log_uniform(rng, 1 / args.max_aspect, args.max_aspect), *args.synth_h))
    m = args.depth
    N = args.synth_cells + 2 * m
    B = args.batch
    geometries = [random_geometry(rng, N * hx, N * hy, min(hx, hy)) for _ in range(B)]
    lo, hi = synth_dt_bounds(args, min(hx, hy))
    dts = log_uniform(rng, lo, hi, (K, B))
    times = np.concatenate([np.zeros((1, B)), np.cumsum(dts, axis=0)])
    frames = [render(geometries, profile, times[k], N, hx, hy, args.synth_sub, device) for k in range(K + 1)]
    steps = [(torch.tensor(dts[k], dtype=torch.float32, device=device), frames[k + 1],
              torch.ones(B, device=device)) for k in range(K)]
    return {"U0": frames[0], "hx": hx, "hy": hy, "steps": steps, "dt_lo": lo, "dt_hi": hi, "crop": m}


# ---------------------------------------------------------------- model

def conv(c_in, c_out, k=3):
    return nn.Conv2d(c_in, c_out, k, padding=k // 2, padding_mode="replicate")


class FiLM(nn.Module):
    """Conditioning vector (B, C) -> a scale and shift per hidden layer, starting at identity."""

    def __init__(self, cond, width, layers, hidden=64):
        super().__init__()
        self.width, self.layers = width, layers
        self.mlp = nn.Sequential(nn.Linear(cond, hidden), nn.GELU(), nn.Linear(hidden, 2 * width * layers))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c):
        out = self.mlp(c).view(len(c), self.layers, 2, self.width, 1, 1)
        return [(out[:, k, 0], out[:, k, 1]) for k in range(self.layers)]


def per_sample(x, B, like):
    x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return x.expand(B) if x.dim() == 0 else x


class Stepper(nn.Module):
    """forward(U, dt, hx, hy): U (B, 4, nx, ny) = cell averages (Y1, Y2, Y3, T) -> U at t + dt.
    dt, hx, hy are floats or (B,) tensors, in ns and nm. nx or ny may be 1 (1D)."""

    def __init__(self, width, depth, dt_ref, h_ref, version=MODEL_VERSION):
        super().__init__()
        self.version, self.dt_ref, self.h_ref = version, dt_ref, h_ref
        n_cond = 3 if version == 1 else 5
        self.convs = nn.ModuleList([conv(4 + n_cond if k == 0 else width, width) for k in range(depth)])
        self.film = FiLM(n_cond, width, depth)
        self.head = nn.Conv2d(width, 5, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[:2] = -4.0  # reaction gates just above their cut-off
            if version >= 2:
                self.head.bias[2:4] = -6.0  # learned conductance starts near zero

    def conditioning(self, dt, hx, hy):
        if self.version == 1:
            return torch.stack([torch.log(dt / self.dt_ref), torch.log(hx / self.h_ref), torch.log(hy / self.h_ref)], 1)
        return torch.stack([torch.log(dt / self.dt_ref), torch.log(S_REF * dt / hx), torch.log(S_REF * dt / hy),
                            DELTA / hx, DELTA / hy], 1)

    def forward(self, U, dt, hx, hy):
        B, _, nx, ny = U.shape
        dt, hx, hy = (per_sample(v, B, U) for v in (dt, hx, hy))
        cond = self.conditioning(dt, hx, hy)
        x = torch.cat([normalize(U), cond.view(B, -1, 1, 1).expand(B, cond.shape[1], nx, ny)], dim=1)
        for layer, (gamma, beta) in zip(self.convs, self.film(cond)):
            x = F.gelu(layer(x) * (1 + gamma) + beta)
        out = self.head(x)

        Y1, Y2, Y3, T = U.unbind(1)
        frac1 = torch.relu(torch.sigmoid(out[:, 0]) - REACT_EPS) / (1 - REACT_EPS)
        frac2 = torch.relu(torch.sigmoid(out[:, 1]) - REACT_EPS) / (1 - REACT_EPS)
        xi1 = frac1 * Y1
        xi2 = frac2 * (Y2 + xi1)

        kx = (fd.ALPHA * dt / hx ** 2).view(B, 1, 1)
        ky = (fd.ALPHA * dt / hy ** 2).view(B, 1, 1)
        if self.version == 1:
            # explicit-ish prior plus an unbounded additive flux (can go against the gradient)
            denom = 1 + 2 * kx * (nx > 1) + 2 * ky * (ny > 1)
            scale = (FLUX_SCALE * dt / self.dt_ref).view(B, 1, 1)
            gx, gy = scale * out[:, 2], scale * out[:, 3]
            Fx = kx / denom * (T[:, :-1] - T[:, 1:]) + 0.5 * (gx[:, :-1] + gx[:, 1:])
            Fy = ky / denom * (T[:, :, :-1] - T[:, :, 1:]) + 0.5 * (gy[:, :, :-1] + gy[:, :, 1:])
        else:
            # monotone: face conductance lam >= 0, coefficient lam / (1 + 4 lam) <= 1/4
            lam_x = kx + LAM_SCALE * F.softplus(0.5 * (out[:, 2, :-1] + out[:, 2, 1:]))
            lam_y = ky + LAM_SCALE * F.softplus(0.5 * (out[:, 3, :, :-1] + out[:, 3, :, 1:]))
            Fx = lam_x / (1 + 4 * lam_x) * (T[:, :-1] - T[:, 1:])
            Fy = lam_y / (1 + 4 * lam_y) * (T[:, :, :-1] - T[:, :, 1:])
        Fx = F.pad(Fx, (0, 0, 1, 1))  # zero flux through the domain boundary
        Fy = F.pad(Fy, (1, 1))
        conduction = Fx[:, :-1] - Fx[:, 1:] + Fy[:, :, :-1] - Fy[:, :, 1:]

        heat = (fd.reaction_heat(fd.A1, fd.B1, T) * xi1 + fd.reaction_heat(fd.A2, fd.B2, T) * xi2) / fd.CV
        T_new = T + conduction + heat * (1 + HEAT_CORRECTION * torch.tanh(out[:, 4]))
        return torch.stack([Y1 - xi1, Y2 + xi1 - xi2, Y3 + xi2, T_new], dim=1)


def field_loss(pred, target, weight=None, crop=0):
    if crop:
        pred, target = pred[..., crop:-crop, crop:-crop], target[..., crop:-crop, crop:-crop]
    err = (pred[:, :3] - target[:, :3]).pow(2).mean(dim=(1, 2, 3)) + \
          ((pred[:, 3] - target[:, 3]) / T_SCALE).pow(2).mean(dim=(1, 2))
    if weight is None:
        return err.mean()
    return (err * weight).sum() / weight.sum().clamp(min=1)


# ---------------------------------------------------------------- training

def sample_multiples(rng, m_min, m_max, size):
    """Integers in [m_min, m_max], log-uniform."""
    return np.clip(np.rint(np.exp(rng.uniform(np.log(m_min), np.log(m_max + 0.5), size))), m_min, m_max).astype(int)


def max_multiple(args, shape):
    """Largest dt / dt_snap trained on this box shape: --dt-range and the Courant cap."""
    h_min = min(shape) * args.dx
    by_courant = math.floor(args.courant * h_min / (S_REF * args.dt_snap) + 1e-9)
    return min(snaps(args.dt_range[1], args.dt_snap), by_courant)


def augment(x, flip_x, flip_y, swap):
    if flip_x:
        x = x.flip(-2)
    if flip_y:
        x = x.flip(-1)
    return x.transpose(-1, -2) if swap else x


def fd_batch(args, rng, runs, usable, K, device):
    """A batch of FD windows: initial state and K (dt, target, active) steps."""
    m_min = snaps(args.dt_range[0], args.dt_snap)
    shape = usable[rng.integers(len(usable))]
    m_max = max_multiple(args, shape)
    last = len(runs[0]["coarse"][shape]) - 1
    flip_x, flip_y, swap = rng.integers(2, size=3).astype(bool)
    hx, hy = shape[0] * args.dx, shape[1] * args.dx
    if swap:
        hx, hy = hy, hx
    B = args.batch
    series = [runs[r]["coarse"][shape] for r in rng.integers(len(runs), size=B)]
    position = rng.integers(0, last - m_min + 1, size=B)

    def frames(pos):
        return augment(torch.stack([series[b][pos[b]] for b in range(B)]), flip_x, flip_y, swap).to(device)

    U0 = frames(position)
    steps = []
    for _ in range(K):
        m = np.minimum(sample_multiples(rng, m_min, m_max, B), last - position)
        active = m >= m_min
        m = np.where(active, m, m_min)  # finished samples still step, but are not scored
        steps.append((torch.tensor(m * args.dt_snap, dtype=torch.float32, device=device),
                      frames(np.minimum(position + m, last)),
                      torch.tensor(active, dtype=torch.float32, device=device)))
        position = np.where(active, position + m, position)
    return {"U0": U0, "hx": hx, "hy": hy, "steps": steps,
            "dt_lo": m_min * args.dt_snap, "dt_hi": m_max * args.dt_snap, "crop": 0}


def train(args, data, shapes, profiles):
    device = torch.device(args.device)
    m_min = snaps(args.dt_range[0], args.dt_snap)
    usable = [s for s in shapes if max_multiple(args, s) >= m_min]
    for s in shapes:
        if s not in usable:
            print(f"box {s}: Courant cap leaves no dt >= {args.dt_range[0]:g} ns; not trained")
    model = Stepper(args.width, args.depth, args.dt_ref, args.h_ref).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed + 1)
    synth_profiles = [Profile(p, device) for p in profiles]

    for epoch in range(1, args.epochs + 1):
        K = min(args.rollout, 1 + (args.rollout - 1) * 2 * (epoch - 1) // max(1, args.epochs))
        synthetic = synth_profiles and rng.uniform() < args.synth_fraction
        batch = synth_batch(args, rng, synth_profiles, K, device) if synthetic else \
            fd_batch(args, rng, data["train"], usable, K, device)
        hx, hy, crop = batch["hx"], batch["hy"], batch["crop"]
        U0 = batch["U0"]
        if args.noise_T > 0:
            U0 = U0.clone()
            U0[:, 3] += args.noise_T * torch.randn_like(U0[:, 3])

        U, loss = U0, 0.0
        for dts, target, active in batch["steps"]:
            U = model(U, dts, hx, hy)
            loss = loss + field_loss(U, target, active, crop)
        loss = loss / K

        lo, hi = batch["dt_lo"], batch["dt_hi"]
        if args.consistency > 0 and 2 * lo <= hi:
            B = len(U0)
            d1 = log_uniform(rng, lo, hi / 2, B)
            d2 = np.array([log_uniform(rng, lo, hi - a) for a in d1])
            d1, d2 = (torch.tensor(a, dtype=torch.float32, device=device) for a in (d1, d2))
            loss = loss + args.consistency * field_loss(model(model(U0, d1, hx, hy), d2, hx, hy),
                                                        model(U0, d1 + d2, hx, hy), crop=crop)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  rollout {K:2d}  {'synthetic' if synthetic else 'FD':9s} "
                  f"h {hx:7.1f} x {hy:7.1f} nm  loss {loss.item():.3e}", flush=True)
    return model


# ---------------------------------------------------------------- evaluation

def front_position(U, hx, hy, planar):
    """Burnt slab half-width (sum Y3 hx hy / (2 Ly)) or burnt disc radius sqrt(A / pi), nm."""
    area = U[:, 2].sum(dim=(-2, -1)) * hx * hy
    return area / (2 * U.shape[-1] * hy) if planar else torch.sqrt(area / math.pi)


def front_speed(times, pos, lo, hi):
    t, r = np.asarray(times), np.asarray(pos)
    use = np.isfinite(r) & (r > lo) & (r < hi)
    return np.polyfit(t[use], r[use], 1)[0] if use.sum() >= 2 else float("nan")


@torch.no_grad()
def evaluate(args, data, shapes, model):
    """Held-out FD runs on the square box shapes, rolled out from t = 0 at three dt values."""
    device = torch.device(args.device)
    m_min = snaps(args.dt_range[0], args.dt_snap)
    for k, run in enumerate(data["test"] or data["train"][:1]):
        print(f"\nheld-out run {k + 1}: {'slab' if run['planar'] else 'disc'} r0 = {run['radius']:.1f} nm, "
              f"T0 = {run['t_hot']:.0f} K on {run['t_cold']:.0f} K{', burnt core' if run['burnt'] else ''}")
        for shape in (s for s in shapes if s[0] == s[1]):
            m_max = max_multiple(args, shape)
            if m_max < m_min:
                continue
            h = shape[0] * args.dx
            half = args.n * args.dx / (2 if run["planar"] else math.sqrt(math.pi))
            for m in sorted({m_min, int(round(math.sqrt(m_min * m_max))), m_max}):
                dt = m * args.dt_snap
                ref = run["coarse"][shape][::m]
                U = ref[0:1].to(device)
                Ufd = ref[0:1].double()
                times, p_ref, p_net, p_fd = [], [], [], []
                for s in range(len(ref)):
                    if s > 0:
                        U = model(U, dt, h, h)
                        Ufd = fd.advance(Ufd, dt, args.dt_fine, h, h)
                    times.append(s * dt)
                    p_ref.append(front_position(ref[s:s + 1], h, h, run["planar"]).item())
                    p_net.append(front_position(U.cpu(), h, h, run["planar"]).item())
                    p_fd.append(front_position(Ufd, h, h, run["planar"]).item()
                                if torch.isfinite(Ufd).all() else float("nan"))
                print(f"  h = {h:4g} nm, dt = {dt:6.3g} ns ({len(ref) - 1:3d} steps): front speed (m/s) "
                      f"fine->avg {front_speed(times, p_ref, 0.2 * half, 0.7 * half):6.1f}, "
                      f"network {front_speed(times, p_net, 0.2 * half, 0.7 * half):6.1f}, "
                      f"coarse FD {front_speed(times, p_fd, 0.2 * half, 0.7 * half):6.1f}; "
                      f"T range at end net [{U[0, 3].min().item():.0f}, {U[0, 3].max().item():.0f}] / "
                      f"ref [{ref[-1, 3].min().item():.0f}, {ref[-1, 3].max().item():.0f}] K")


@torch.no_grad()
def evaluate_large(args, profiles, model, cells=48, hs=(128.0, 500.0, 2000.0), courants=(0.1, 1.0)):
    """Synthetic growing disc on large cells: network front speed and T range vs the 1D front."""
    device = torch.device(args.device)
    p = Profile(profiles[0], device)
    print(f"\nlarge cells, burning disc on {p.t_cold:.0f} K (1D front {p.speed:.1f} m/s, burnt T {p.T_burnt:.0f} K):")
    for h in hs:
        D = cells * h
        r0 = 6 * h
        geom = [lambda X, Y, t, s: torch.sqrt((X - D / 2) ** 2 + (Y - D / 2) ** 2) - (r0 + s * t)]
        U = render(geom, p, [0.0], cells, h, h, args.synth_sub, device)
        for C in courants:
            dt = C * h / S_REF
            steps = int(math.ceil(12 * h / (p.speed * dt)))  # radius 6h -> 18h
            V, times, radius = U, [0.0], [front_position(U, h, h, False).item()]
            for s in range(1, steps + 1):
                V = model(V, dt, h, h)
                times.append(s * dt)
                radius.append(front_position(V, h, h, False).item())
            print(f"  h = {h:6g} nm, dt = {dt:7.3g} ns (front Courant {C:g}, {steps} steps): network "
                  f"{front_speed(times, radius, 7 * h, 17 * h):6.1f} m/s; T range "
                  f"[{V[0, 3].min().item():.0f}, {V[0, 3].max().item():.0f}] K")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    shapes = box_shapes(args.factors, args.max_aspect, args.n)
    snaps(args.t_end, args.dt_snap)
    print(f"fine {args.n}^2 at {args.dx:g} nm; box shapes {shapes}; dt {args.dt_range[0]:g}-{args.dt_range[1]:g} ns, "
          f"front Courant <= {args.courant:g}", flush=True)
    data = load_or_generate(args, shapes)
    profiles = make_profiles(args) if args.synth_fraction > 0 else []
    model = train(args, data, shapes, profiles)

    trained = [s for s in shapes if max_multiple(args, s) >= snaps(args.dt_range[0], args.dt_snap)]
    h_range = [min(min(s) for s in trained) * args.dx, max(max(s) for s in trained) * args.dx]
    dt_max = args.dt_range[1]
    if profiles:
        h_range[1] = max(h_range[1], args.synth_h[1])
        dt_max = max(dt_max, args.synth_dt_max)
    torch.save({"model": model.state_dict(),
                "args": {**vars(args), "out": str(args.out), "data": str(args.data),
                         "model_kind": MODEL_KIND, "model_version": MODEL_VERSION,
                         "h_range": h_range, "dt_max": dt_max,
                         "front_speeds": {p["t_cold"]: p["speed"] for p in profiles}}}, args.out)
    print(f"saved {args.out}")
    evaluate(args, data, shapes, model)
    if profiles:
        evaluate_large(args, profiles, model)


if __name__ == "__main__":
    main()
