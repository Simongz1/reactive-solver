"""Markovian coarse kinetics surrogate: (T, Y) averaged on any uniform grid -> (T, Y) after dt.

The model has no latent or history state. Its inputs are the cell averages of T and Y1..Y3,
the requested step dt and the cell size (hx, hy). It returns the averages at t + dt. It is
trained on box averages of fine FD runs (src/kinetics2D.C, ported in fd_kinetics.py), so the
target is always the resolved solution, not coarse FD, and never FD with a large step.
Use it through surrogate.py:

    from surrogate import KineticsSurrogate
    model = KineticsSurrogate("increment_surrogate.pt")
    T_new, Y_new = model.predict(T, Y, dt, h=(hx, hy))       # or extent=(Lx, Ly)

Pipeline:
  1. Fine FD runs (float64, stable step --dt-fine) of random hotspots: discs or planar slabs,
     unreacted or burnt cores, on a cold background of random temperature.
  2. Every --dt-snap ns each fine state is box-averaged onto several coarse grids, one per
     box shape (fx, fy) built from --factors with aspect ratio <= --max-aspect. A box shape
     is a cell size (hx, hy) = (fx dx, fy dy), so one run trains many cell sizes, including
     anisotropic cells (what a compressed mesh produces).
  3. A fully convolutional network, conditioned on log(dt / dt_ref), log(hx / h_ref) and
     log(hy / h_ref) (input channels plus FiLM after every hidden conv), predicts the
     update in finite-volume form:
       Y1 -> Y1 - xi1, Y2 -> Y2 + xi1 - xi2, Y3 -> Y3 + xi2   (0 <= xi1 <= Y1, 0 <= xi2 <= Y2 + xi1)
       T  -> T + (net face heat flux) + (Q1 xi1 + Q2 xi2)/cv * (1 + bounded correction)
     The conduction prior is one Jacobi sweep of backward Euler,
       c_x = kx / (1 + 2 kx + 2 ky),  kx = alpha dt / hx^2,
     which equals explicit conduction for small k and stays monotone for any dt, so there is
     no conduction limit on dt. Face fluxes are zero on the domain boundary, species sum and
     bounds hold exactly, and no heat appears without reaction.

Training pairs: (snapshot s, snapshot s + m) of one run on one coarse grid, dt = m dt_snap,
m log-uniform in --dt-range, but capped per box shape so that a front at S_REF (75 m/s)
crosses at most --courant cells per step (the network sees depth cells on each side, so
the information it needs must be inside its receptive field). surrogate.py sub-steps any
larger dt automatically. Windows of up to --rollout steps feed the network its own output
(each sample with its own dt), plus a consistency loss step(dt1 + dt2) == step(dt1) step(dt2).

Example (generation dominates the cost; run it on a cluster node):
  python ml/train_increment.py --n 512 --factors 4 8 16 32 64 --t-end 3 \
      --trajectories 64 --data increment_data.pt --out increment_surrogate.pt

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
FLUX_SCALE = 10.0  # K per reference step, scale of the learned flux outputs
REACT_EPS = 0.01   # reacted fraction = max(0, sigmoid - eps) / (1 - eps): exactly 0 when off
HEAT_CORRECTION = 0.2  # largest relative correction of the heat of reaction (Q at the cell mean T)
S_REF = 75.0       # nm/ns, front speed used for the Courant cap on dt
MODEL_KIND = "increment"
MODEL_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("fine FD runs")
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
    g.add_argument("--max-aspect", type=float, default=2.0, help="largest fx/fy of a box shape")
    g.add_argument("--gen-device", default="cpu", help="device for the float64 FD runs (cpu or cuda)")
    g.add_argument("--data", type=Path, help="cache file: loaded if it exists, else written after generation")
    g = p.add_argument_group("model and training")
    g.add_argument("--width", type=int, default=48)
    g.add_argument("--depth", type=int, default=6, help="3x3 conv layers (receptive field: depth cells per side)")
    g.add_argument("--dt-range", type=float, nargs=2, default=[0.005, 0.5], help="training steps, ns")
    g.add_argument("--dt-ref", type=float, default=0.05, help="reference step for normalization, ns")
    g.add_argument("--h-ref", type=float, default=16.0, help="reference cell size for normalization, nm")
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


# ---------------------------------------------------------------- data

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

    def __init__(self, width, depth, dt_ref, h_ref):
        super().__init__()
        self.dt_ref, self.h_ref = dt_ref, h_ref
        self.convs = nn.ModuleList([conv(7 if k == 0 else width, width) for k in range(depth)])
        self.film = FiLM(3, width, depth)
        self.head = nn.Conv2d(width, 5, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[:2] = -4.0  # reaction gates just above their cut-off

    def forward(self, U, dt, hx, hy):
        B, _, nx, ny = U.shape
        dt, hx, hy = (per_sample(v, B, U) for v in (dt, hx, hy))
        cond = torch.stack([torch.log(dt / self.dt_ref), torch.log(hx / self.h_ref), torch.log(hy / self.h_ref)], 1)
        x = torch.cat([normalize(U), cond.view(B, 3, 1, 1).expand(B, 3, nx, ny)], dim=1)
        for layer, (gamma, beta) in zip(self.convs, self.film(cond)):
            x = F.gelu(layer(x) * (1 + gamma) + beta)
        out = self.head(x)

        Y1, Y2, Y3, T = U.unbind(1)
        frac1 = torch.relu(torch.sigmoid(out[:, 0]) - REACT_EPS) / (1 - REACT_EPS)
        frac2 = torch.relu(torch.sigmoid(out[:, 1]) - REACT_EPS) / (1 - REACT_EPS)
        xi1 = frac1 * Y1
        xi2 = frac2 * (Y2 + xi1)

        # conduction prior: one Jacobi sweep of backward Euler (monotone for any dt)
        kx = fd.ALPHA * dt / hx ** 2 * (nx > 1)
        ky = fd.ALPHA * dt / hy ** 2 * (ny > 1)
        denom = 1 + 2 * kx + 2 * ky
        cx, cy = (kx / denom).view(B, 1, 1), (ky / denom).view(B, 1, 1)
        scale = (FLUX_SCALE * dt / self.dt_ref).view(B, 1, 1)
        gx, gy = scale * out[:, 2], scale * out[:, 3]
        Fx = cx * (T[:, :-1] - T[:, 1:]) + 0.5 * (gx[:, :-1] + gx[:, 1:])
        Fy = cy * (T[:, :, :-1] - T[:, :, 1:]) + 0.5 * (gy[:, :, :-1] + gy[:, :, 1:])
        Fx = F.pad(Fx, (0, 0, 1, 1))  # zero flux through the domain boundary
        Fy = F.pad(Fy, (1, 1))
        conduction = Fx[:, :-1] - Fx[:, 1:] + Fy[:, :, :-1] - Fy[:, :, 1:]

        heat = (fd.reaction_heat(fd.A1, fd.B1, T) * xi1 + fd.reaction_heat(fd.A2, fd.B2, T) * xi2) / fd.CV
        T_new = T + conduction + heat * (1 + HEAT_CORRECTION * torch.tanh(out[:, 4]))
        return torch.stack([Y1 - xi1, Y2 + xi1 - xi2, Y3 + xi2, T_new], dim=1)


def field_loss(pred, target, weight=None):
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


def train(args, data, shapes):
    device = torch.device(args.device)
    m_min = snaps(args.dt_range[0], args.dt_snap)
    usable = [s for s in shapes if max_multiple(args, s) >= m_min]
    for s in shapes:
        if s not in usable:
            print(f"box {s}: Courant cap leaves no dt >= {args.dt_range[0]:g} ns; not trained")
    model = Stepper(args.width, args.depth, args.dt_ref, args.h_ref).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    runs = data["train"]
    last = len(runs[0]["coarse"][usable[0]]) - 1
    rng = np.random.default_rng(args.seed + 1)

    for epoch in range(1, args.epochs + 1):
        K = min(args.rollout, 1 + (args.rollout - 1) * 2 * (epoch - 1) // max(1, args.epochs))
        shape = usable[rng.integers(len(usable))]
        m_max = max_multiple(args, shape)
        flip_x, flip_y, swap = rng.integers(2, size=3).astype(bool)
        hx, hy = shape[0] * args.dx, shape[1] * args.dx
        if swap:
            hx, hy = hy, hx
        B = args.batch
        run_ids = rng.integers(len(runs), size=B)
        position = rng.integers(0, last - m_min + 1, size=B)
        series = [runs[r]["coarse"][shape] for r in run_ids]

        def frames(pos):
            return augment(torch.stack([series[b][pos[b]] for b in range(B)]), flip_x, flip_y, swap).to(device)

        U0 = frames(position)
        if args.noise_T > 0:
            U0 = U0.clone()
            U0[:, 3] += args.noise_T * torch.randn_like(U0[:, 3])

        U, loss = U0, 0.0
        for _ in range(K):
            m = np.minimum(sample_multiples(rng, m_min, m_max, B), last - position)
            active = m >= m_min
            m = np.where(active, m, m_min)  # finished samples still step, but are not scored
            dts = torch.tensor(m * args.dt_snap, dtype=torch.float32, device=device)
            U = model(U, dts, hx, hy)
            target = frames(np.minimum(position + m, last))
            loss = loss + field_loss(U, target, torch.tensor(active, dtype=torch.float32, device=device))
            position = np.where(active, position + m, position)
        loss = loss / K

        if args.consistency > 0 and 2 * m_min <= m_max:
            m1 = sample_multiples(rng, m_min, m_max // 2, B)
            m2 = np.array([sample_multiples(rng, m_min, m_max - a, 1)[0] for a in m1])
            d1, d2 = (torch.tensor(a * args.dt_snap, dtype=torch.float32, device=device) for a in (m1, m2))
            loss = loss + args.consistency * field_loss(model(model(U0, d1, hx, hy), d2, hx, hy),
                                                        model(U0, d1 + d2, hx, hy))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  rollout {K:2d}  box {shape}  loss {loss.item():.3e}", flush=True)
    return model


# ---------------------------------------------------------------- evaluation

def front_position(U, hx, hy, planar):
    """Burnt slab half-width (sum Y3 hx hy / (2 Ly)) or burnt disc radius sqrt(A / pi), nm."""
    area = U[:, 2].sum(dim=(-2, -1)) * hx * hy
    return area / (2 * U.shape[-1] * hy) if planar else torch.sqrt(area / math.pi)


def front_speed(times, pos, half):
    t, r = np.asarray(times), np.asarray(pos)
    use = np.isfinite(r) & (r > 0.2 * half) & (r < 0.7 * half)
    return np.polyfit(t[use], r[use], 1)[0] if use.sum() >= 2 else float("nan")


@torch.no_grad()
def evaluate(args, data, shapes, model):
    """Held-out runs on the square box shapes, rolled out from t = 0 at three dt values."""
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
                err_T = (U.cpu()[0, 3] - ref[-1, 3]).abs().max().item()
                print(f"  h = {h:4g} nm, dt = {dt:6.3g} ns ({len(ref) - 1:3d} steps): front speed (m/s) "
                      f"fine->avg {front_speed(times, p_ref, half):6.1f}, network {front_speed(times, p_net, half):6.1f}, "
                      f"coarse FD {front_speed(times, p_fd, half):6.1f}; max |T error| at end {err_T:.0f} K")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    shapes = box_shapes(args.factors, args.max_aspect, args.n)
    snaps(args.t_end, args.dt_snap)
    print(f"fine {args.n}^2 at {args.dx:g} nm; box shapes {shapes}; dt {args.dt_range[0]:g}-{args.dt_range[1]:g} ns, "
          f"front Courant <= {args.courant:g}", flush=True)
    data = load_or_generate(args, shapes)
    model = train(args, data, shapes)
    trained = [s for s in shapes if max_multiple(args, s) >= snaps(args.dt_range[0], args.dt_snap)]
    h_range = [min(min(s) for s in trained) * args.dx, max(max(s) for s in trained) * args.dx]
    torch.save({"model": model.state_dict(),
                "args": {**vars(args), "out": str(args.out), "data": str(args.data),
                         "model_kind": MODEL_KIND, "model_version": MODEL_VERSION, "h_range": h_range}}, args.out)
    print(f"saved {args.out}")
    evaluate(args, data, shapes, model)


if __name__ == "__main__":
    main()
