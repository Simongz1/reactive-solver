"""Coarse-grid time stepper for the 2D kinetics, trained from fine FD runs, conditional on dt.

Pipeline (no mechanics):
  1. Run circular hotspots with the exact FD residual and RK3 of src/kinetics2D.C
     (ported in fd_kinetics.py) on a fine grid, always at the stable fine step --dt-fine.
     The box-averaged state is saved every --dt-snap ns.
  2. Restrict to the target grid: physical fields (Y1, Y2, Y3, T) by fixed box averages
     over f x f nodes, f = 2**levels (conservative, since rho and cv are constant).
     A learned encoder (levels x [conv -> GELU -> avg-pool 2]) maps the fine field to
     extra latent channels per coarse cell: the subgrid information the averages lose.
  3. One coarse CNN marches (averages, latent) from t to t + dt for a requested dt. dt is a
     per-sample input (a (B,) tensor): log(dt / dt_ref) is an input channel and also sets a
     FiLM scale and shift after every hidden conv. The update has a finite-volume form:
       Y1 -> Y1 - xi1, Y2 -> Y2 + xi1 - xi2, Y3 -> Y3 + xi2   (0 <= xi1 <= Y1, 0 <= xi2 <= Y2 + xi1)
       T  -> T + (net face heat flux) + (Q1 xi1 + Q2 xi2)/cv * (1 + correction)
     with face fluxes zero on the domain boundary, reacted fractions exactly 0 unless the
     network turns them on, and no heat source other than conduction and reaction.

Training pairs for a step dt = m * dt_snap are (snapshot s, snapshot s + m) of the same
fine trajectory: the target is always the resolved FD solution over that interval, never
FD taken with a large step. Each window starts from an encoded fine snapshot and takes up
to --rollout steps, every sample with its own random dt (log-uniform in --dt-range). A
consistency loss makes one step of dt1 + dt2 agree with a step of dt1 then dt2.

Model versions (stored in the checkpoint as model_version):
  1  sigmoid fractions and a free heat source, fixed dt
  2  exact-zero fractions, no free heat source, fixed dt
  3  version 2 conditioned on dt (this script); Stepper(version=1/2) still loads 1 and 2.

Loss (--loss):
  fine            target = box average of the fine FD solution at t + dt (default).
  coarse-residual target = the same FD residual and RK3 run on the coarse grid from the
                  previous coarse state (the unresolved answer, for comparison).

Example (generation dominates the cost; run it on a cluster node, float64 on CPU or CUDA):
  python ml/train_coarse.py --n 256 --dx 1 --levels 4 --dt-snap 0.005 --t-end 1.5 \
      --dt-range 0.005 0.5 --trajectories 48 --data hotspots_v3.pt --out coarse_stepper_v3.pt

Units are nm and ns, so 1 nm/ns = 1 m/s: a 76 m/s front crosses a 16 nm coarse cell in
about 0.2 ns and reaches the edge of a 256 nm domain in under 2 ns. Size --t-end so the
front stays inside the domain for most of the run.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fd_kinetics as fd

T_COLD = 300.0
T_SCALE = 1000.0  # K, for normalization and the T loss
FLUX_SCALE = 10.0  # K per reference step, scale of the learned flux outputs
REACT_EPS = 0.01   # reacted fraction = max(0, sigmoid - eps) / (1 - eps): exactly 0 when off
HEAT_CORRECTION = 0.2  # largest relative correction of the heat of reaction (Q at the cell mean T)
MODEL_VERSION = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("fine FD runs")
    g.add_argument("--n", type=int, default=256, help="fine nodes per side (divisible by 2**levels)")
    g.add_argument("--dx", type=float, default=1.0, help="fine node spacing, nm")
    g.add_argument("--dt-fine", type=float, default=0.0005, help="fine RK3 step, ns (at dx = 1 nm, 0.001 is marginal and 0.002 unstable)")
    g.add_argument("--dt-snap", type=float, default=0.005, help="spacing of the saved coarse snapshots, ns; every dt is a multiple of it")
    g.add_argument("--t-end", type=float, default=1.5, help="length of each trajectory, ns")
    g.add_argument("--trajectories", type=int, default=32, help="training trajectories")
    g.add_argument("--holdout", type=int, default=1, help="extra trajectories kept for evaluation")
    g.add_argument("--radius", type=float, nargs=2, default=[10.0, 40.0], help="hotspot radius range, nm")
    g.add_argument("--t-hot", type=float, nargs=2, default=[1500.0, 2000.0], help="hotspot temperature range, K")
    g.add_argument("--burnt-core", action="store_true", help="start every hotspot reacted (Y3 = 1)")
    g.add_argument("--burnt-fraction", type=float, default=0.0,
                   help="fraction of hotspots that start reacted (Y3 = 1), the rest unreacted")
    g.add_argument("--t-hot-burnt", type=float, nargs=2,
                   help="temperature range for reacted hotspots, K (default: --t-hot)")
    g.add_argument("--start-every", type=float, default=0.25,
                   help="keep a fine snapshot (a possible window start) every this many ns")
    g.add_argument("--gen-device", default="cpu", help="device for the float64 FD runs (cpu or cuda; MPS has no float64)")
    g.add_argument("--data", type=Path, help="cache file: loaded if it exists, else written after generation")
    g = p.add_argument_group("model and training")
    g.add_argument("--levels", type=int, default=4, help="conv + avg-pool levels; coarsening factor 2**levels")
    g.add_argument("--latent", type=int, default=8, help="latent channels per coarse cell")
    g.add_argument("--width", type=int, default=32)
    g.add_argument("--depth", type=int, default=4, help="conv layers in the coarse stepper")
    g.add_argument("--dt-range", type=float, nargs=2, default=[0.005, 0.5],
                   help="range of training steps, ns (multiples of --dt-snap, sampled log-uniformly)")
    g.add_argument("--dt-coarse", type=float, default=0.05,
                   help="reference step, ns: normalizes log(dt) and the learned fluxes; the default step for prediction")
    g.add_argument("--loss", choices=["fine", "coarse-residual"], default="fine")
    g.add_argument("--rollout", type=int, default=8, help="steps per training window (reached by curriculum)")
    g.add_argument("--consistency", type=float, default=0.1,
                   help="weight of the loss |step(dt1) then step(dt2) - step(dt1 + dt2)|")
    g.add_argument("--noise-T", type=float, default=0.0,
                   help="std of random temperature noise (K) added to each window's starting state, "
                        "so the stepper learns to correct its own errors on long rollouts")
    g.add_argument("--epochs", type=int, default=2000)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, default=Path("coarse_stepper.pt"))
    return p.parse_args()


def restrict(U, f):
    """Box average over f x f nodes: (..., 4, n, n) -> (..., 4, n/f, n/f)."""
    lead = U.shape[:-2]
    x = F.avg_pool2d(U.reshape(-1, *U.shape[-2:]).unsqueeze(1), f).squeeze(1)
    return x.reshape(*lead, *x.shape[-2:])


def normalize(U):
    return torch.cat([U[:, :3], (U[:, 3:] - T_COLD) / T_SCALE], dim=1)


def snaps(time_ns, dt_snap):
    """Number of snapshot intervals in time_ns (must be a whole multiple of dt_snap)."""
    m = round(time_ns / dt_snap)
    if m < 1 or abs(m * dt_snap - time_ns) > 1e-9 * max(1.0, time_ns):
        raise SystemExit(f"{time_ns:g} ns is not a positive multiple of --dt-snap {dt_snap:g} ns")
    return m


# ---------------------------------------------------------------- data

def run_trajectory(args, rng, f):
    """One hotspot: coarse snapshots every dt_snap, fine snapshots every start_every."""
    n, dx = args.n, args.dx
    radius = rng.uniform(*args.radius)
    burnt = args.burnt_core or rng.uniform() < args.burnt_fraction
    t_hot = rng.uniform(*(args.t_hot_burnt if burnt and args.t_hot_burnt else args.t_hot))
    # random sub-cell position so fronts do not always sit at the same place in a cell
    center = tuple(rng.uniform(-0.5, 0.5, size=2) * f * dx)
    U = torch.tensor(fd.circular_hotspot(n, n, dx, dx, radius, t_hot, T_COLD, burnt, center),
                     dtype=torch.float64, device=args.gen_device)
    count, fine_every = snaps(args.t_end, args.dt_snap), snaps(args.start_every, args.dt_snap)
    coarse, fine = [restrict(U, f).float().cpu()], {0: U.float().cpu()}
    for s in range(1, count + 1):
        U = fd.advance(U, args.dt_snap, args.dt_fine, dx, dx)
        if not torch.isfinite(U).all():
            raise RuntimeError(f"fine FD run became non-finite at t = {s * args.dt_snap} ns; lower --dt-fine")
        coarse.append(restrict(U, f).float().cpu())
        if s % fine_every == 0:
            fine[s] = U.float().cpu()
    return {"coarse": torch.stack(coarse), "fine": fine, "radius": radius, "t_hot": t_hot, "burnt": burnt}


def load_or_generate(args, f):
    if args.data and args.data.exists():
        data = torch.load(args.data)
        saved = data.get("dt_snap")
        if saved is None or abs(saved - args.dt_snap) > 1e-12:
            raise SystemExit(f"{args.data} has snapshots every {saved} ns, not --dt-snap {args.dt_snap:g}; "
                             "use a new --data file")
        print(f"loaded {len(data['train'])} + {len(data['test'])} trajectories from {args.data}")
        return data
    rng = np.random.default_rng(args.seed)
    trajs = []
    for k in range(args.trajectories + args.holdout):
        start = time.time()
        trajs.append(run_trajectory(args, rng, f))
        t = trajs[-1]
        print(f"trajectory {k + 1}/{args.trajectories + args.holdout}: r = {t['radius']:.1f} nm, "
              f"T = {t['t_hot']:.0f} K{', burnt core' if t.get('burnt') else ''}, {time.time() - start:.0f} s",
              flush=True)
    data = {"train": trajs[:args.trajectories], "test": trajs[args.trajectories:], "dt_snap": args.dt_snap}
    if args.data:
        torch.save(data, args.data)
    return data


# ---------------------------------------------------------------- model

def conv(c_in, c_out, k=3):
    return nn.Conv2d(c_in, c_out, k, padding=k // 2, padding_mode="replicate")


class Encoder(nn.Module):
    """Fine field -> latent channels on the coarse grid: levels x [conv -> GELU -> avg-pool 2]."""

    def __init__(self, levels, width, latent):
        super().__init__()
        layers, c = [], 4
        for _ in range(levels):
            layers += [conv(c, width), nn.GELU(), nn.AvgPool2d(2)]
            c = width
        layers.append(nn.Conv2d(c, latent, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, U_fine):
        return self.net(normalize(U_fine))


class FiLM(nn.Module):
    """log(dt / dt_ref) of each sample -> a scale (gamma) and shift (beta) per hidden layer."""

    def __init__(self, width, layers, hidden=64):
        super().__init__()
        self.width, self.layers = width, layers
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 2 * width * layers))
        nn.init.zeros_(self.mlp[-1].weight)  # start as the identity (gamma = beta = 0)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, log_dt):
        """log_dt: (B,) -> list of `layers` pairs (gamma, beta), each (B, width, 1, 1)."""
        out = self.mlp(log_dt[:, None]).view(len(log_dt), self.layers, 2, self.width, 1, 1)
        return [(out[:, k, 0], out[:, k, 1]) for k in range(self.layers)]


class Stepper(nn.Module):
    """(coarse averages, latent) at t -> at t + dt, in finite-volume form.

    version 3: forward(U, z, dt) with dt a float or a (B,) tensor of steps in ns.
    versions 1 and 2 (older checkpoints) were trained at the single step `dt` given here.
    """

    def __init__(self, latent, width, depth, dt, h, version=MODEL_VERSION):
        super().__init__()
        self.version, self.dt_ref, self.h = version, dt, h
        if version < 3:
            layers, c = [], 4 + latent
            for _ in range(depth):
                layers += [conv(c, width), nn.GELU()]
                c = width
            self.body = nn.Sequential(*layers)
        else:
            # one extra input channel: log(dt / dt_ref)
            self.convs = nn.ModuleList([conv(5 + latent if k == 0 else width, width) for k in range(depth)])
            self.film = FiLM(width, depth)
        self.head = nn.Conv2d(width, 5 + latent, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            # start with almost no reaction per step (versions 2 and 3: just above the
            # cut-off, so the gate still has gradients at the start of training)
            self.head.bias[:2] = -6.0 if version == 1 else -4.0

    def diffusion(self, dt):
        """alpha dt / h^2 of the coarse explicit conduction prior (stable below 0.25)."""
        return fd.ALPHA * dt / self.h ** 2

    def forward(self, U, z, dt=None):
        B = U.shape[0]
        if self.version < 3:
            if dt is not None and not torch.allclose(torch.as_tensor(dt, dtype=torch.float64),
                                                     torch.tensor(self.dt_ref, dtype=torch.float64)):
                raise ValueError(f"model_version {self.version} was trained at dt = {self.dt_ref:g} ns only")
            out = self.head(self.body(torch.cat([normalize(U), z], dim=1)))
            dt = torch.full((B,), self.dt_ref, dtype=U.dtype, device=U.device)
        else:
            dt = torch.as_tensor(self.dt_ref if dt is None else dt, dtype=U.dtype, device=U.device)
            dt = dt.expand(B) if dt.dim() == 0 else dt
            log_dt = torch.log(dt / self.dt_ref)
            x = torch.cat([normalize(U), z, log_dt.view(B, 1, 1, 1).expand(B, 1, *U.shape[-2:])], dim=1)
            for layer, (gamma, beta) in zip(self.convs, self.film(log_dt)):
                x = F.gelu(layer(x) * (1 + gamma) + beta)
            out = self.head(x)

        Y1, Y2, Y3, T = U.unbind(1)
        if self.version == 1:
            frac1, frac2 = torch.sigmoid(out[:, 0]), torch.sigmoid(out[:, 1])
        else:
            frac1 = torch.relu(torch.sigmoid(out[:, 0]) - REACT_EPS) / (1 - REACT_EPS)
            frac2 = torch.relu(torch.sigmoid(out[:, 1]) - REACT_EPS) / (1 - REACT_EPS)
        xi1 = frac1 * Y1
        xi2 = frac2 * (Y2 + xi1)

        # face heat fluxes (K): coarse explicit conduction over dt + learned correction
        k = self.diffusion(dt).view(B, 1, 1)
        scale = FLUX_SCALE if self.version < 3 else FLUX_SCALE * (dt / self.dt_ref).view(B, 1, 1)
        gx, gy = scale * out[:, 2], scale * out[:, 3]
        Fx = k * (T[:, :-1] - T[:, 1:]) + 0.5 * (gx[:, :-1] + gx[:, 1:])
        Fy = k * (T[:, :, :-1] - T[:, :, 1:]) + 0.5 * (gy[:, :, :-1] + gy[:, :, 1:])
        Fx = F.pad(Fx, (0, 0, 1, 1))  # zero flux through the domain boundary
        Fy = F.pad(Fy, (1, 1))
        conduction = Fx[:, :-1] - Fx[:, 1:] + Fy[:, :, :-1] - Fy[:, :, 1:]

        heat = (fd.reaction_heat(fd.A1, fd.B1, T) * xi1 + fd.reaction_heat(fd.A2, fd.B2, T) * xi2) / fd.CV
        if self.version == 1:
            T_new = T + conduction + heat + FLUX_SCALE * out[:, 4]
        else:
            # Q is evaluated at the cell-mean T; the subgrid temperature changes it, so the
            # heat of reaction gets a bounded correction. No heat appears without reaction.
            T_new = T + conduction + heat * (1 + HEAT_CORRECTION * torch.tanh(out[:, 4]))
        U_new = torch.stack([Y1 - xi1, Y2 + xi1 - xi2, Y3 + xi2, T_new], dim=1)
        return U_new, z + out[:, 5:]


def field_loss(pred, target, weight=None):
    """MSE of Y and T / T_SCALE; weight (B,) masks or weights samples."""
    err = (pred[:, :3] - target[:, :3]).pow(2).mean(dim=(1, 2, 3)) + \
          ((pred[:, 3] - target[:, 3]) / T_SCALE).pow(2).mean(dim=(1, 2))
    if weight is None:
        return err.mean()
    return (err * weight).sum() / weight.sum().clamp(min=1)


# ---------------------------------------------------------------- training

def coarse_fd_step(U, args, h, dts):
    """The FD residual and RK3 on the coarse grid, sample b advanced by dts[b] (ns)."""
    with torch.no_grad():
        return torch.stack([fd.advance(U[b].double(), float(dts[b]), args.dt_fine, h, h)
                            for b in range(len(U))]).to(U.dtype)


def sample_multiples(rng, m_min, m_max, size):
    """Integers in [m_min, m_max], log-uniform."""
    return np.clip(np.rint(np.exp(rng.uniform(np.log(m_min), np.log(m_max + 0.5), size))), m_min, m_max).astype(int)


def train(args, data, f):
    device = torch.device(args.device)
    h = f * args.dx
    m_min, m_max = snaps(args.dt_range[0], args.dt_snap), snaps(args.dt_range[1], args.dt_snap)
    k_max = fd.ALPHA * args.dt_range[1] / h ** 2
    if k_max >= 0.25:
        raise SystemExit(f"alpha dt / h^2 = {k_max:.2f} at the largest dt: the explicit conduction prior is "
                         f"unstable above dt = {0.25 * h * h / fd.ALPHA:.2f} ns; lower --dt-range")
    encoder = Encoder(args.levels, args.width, args.latent).to(device)
    stepper = Stepper(args.latent, args.width, args.depth, args.dt_coarse, h).to(device)
    params = list(encoder.parameters()) + list(stepper.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    trajs = data["train"]
    starts = [(t, s) for t, traj in enumerate(trajs) for s in traj["fine"] if s + m_min < len(traj["coarse"])]
    rng = np.random.default_rng(args.seed + 1)

    for epoch in range(1, args.epochs + 1):
        # curriculum: rollout length grows from 1 to --rollout over the first half
        K = min(args.rollout, 1 + (args.rollout - 1) * 2 * (epoch - 1) // max(1, args.epochs))
        picks = [starts[i] for i in rng.choice(len(starts), size=min(args.batch, len(starts)), replace=False)]
        B = len(picks)
        U_fine = torch.stack([trajs[t]["fine"][s] for t, s in picks]).to(device)
        U0 = torch.stack([trajs[t]["coarse"][s] for t, s in picks]).to(device)
        z0 = encoder(U_fine)
        if args.noise_T > 0:
            U0 = U0.clone()
            U0[:, 3] += args.noise_T * torch.randn_like(U0[:, 3])

        # rollout: every sample takes its own random dt at every step, until its trajectory ends
        position = np.array([s for _, s in picks])
        last = np.array([len(trajs[t]["coarse"]) - 1 for t, _ in picks])
        U, z, loss = U0, z0, 0.0
        for _ in range(K):
            m = np.minimum(sample_multiples(rng, m_min, m_max, B), last - position)
            active = m >= m_min
            m = np.where(active, m, m_min)  # finished samples still step, but are not scored
            dts = torch.tensor(m * args.dt_snap, dtype=torch.float32, device=device)
            prev = U
            U, z = stepper(U, z, dts)
            if args.loss == "fine":
                target = torch.stack([trajs[t]["coarse"][min(p + mk, l)]
                                      for (t, _), p, mk, l in zip(picks, position, m, last)]).to(device)
            else:
                target = coarse_fd_step(prev.detach().cpu(), args, h, dts.cpu()).to(device)
            loss = loss + field_loss(U, target, torch.tensor(active, dtype=torch.float32, device=device))
            position = np.where(active, position + m, position)
        loss = loss / K

        # consistency: one step of dt1 + dt2 == a step of dt1 followed by a step of dt2
        if args.consistency > 0 and 2 * m_min <= m_max:
            m1 = sample_multiples(rng, m_min, m_max // 2, B)
            m2 = np.array([sample_multiples(rng, m_min, m_max - a, 1)[0] for a in m1])
            d1, d2 = (torch.tensor(a * args.dt_snap, dtype=torch.float32, device=device) for a in (m1, m2))
            Ua, _ = stepper(*stepper(U0, z0, d1), d2)
            Ub, _ = stepper(U0, z0, d1 + d2)
            loss = loss + args.consistency * field_loss(Ua, Ub)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  rollout {K:2d}  loss {loss.item():.3e}", flush=True)
    return encoder, stepper


# ---------------------------------------------------------------- evaluation

def burnt_radius(U, h):
    """Equivalent radius of the reacted region, sqrt(sum(Y3) h^2 / pi), nm."""
    return torch.sqrt(U[:, 2].sum(dim=(-2, -1)) * h * h / math.pi)


def front_speed(times, radii, r_domain):
    """Slope of burnt radius vs time while 0.2 < r / r_domain < 0.7, i.e. after ignition
    and before the front reaches the boundary. nm/ns == m/s."""
    t, r = np.asarray(times), np.asarray(radii)
    use = np.isfinite(r) & (r > 0.2 * r_domain) & (r < 0.7 * r_domain)
    return np.polyfit(t[use], r[use], 1)[0] if use.sum() >= 2 else float("nan")


@torch.no_grad()
def evaluate(args, data, f, encoder, stepper):
    """Held-out hotspots, rolled out from t = 0 at a small, a middle and the largest trained dt."""
    device = torch.device(args.device)
    h = f * args.dx
    m_min, m_max = snaps(args.dt_range[0], args.dt_snap), snaps(args.dt_range[1], args.dt_snap)
    multiples = sorted({m_min, int(round(math.sqrt(m_min * m_max))), m_max})
    for k, traj in enumerate(data["test"] or data["train"][:1]):
        ref_all = traj["coarse"]
        r_domain = ref_all.shape[-1] * h / math.sqrt(math.pi)  # radius of a disc with the domain's area
        print(f"\nheld-out trajectory {k + 1}: r0 = {traj['radius']:.1f} nm, T0 = {traj['t_hot']:.0f} K"
              f"{' (burnt core)' if traj.get('burnt') else ''}, coarse cell {h:g} nm")
        for m in multiples:
            dt = m * args.dt_snap
            ref = ref_all[::m]
            U = ref[0:1].to(device)
            z = encoder(traj["fine"][0][None].to(device))
            Ufd = ref[0:1].double()
            times, r_ref, r_net, r_fd = [], [], [], []
            for s in range(len(ref)):
                if s > 0:
                    U, z = stepper(U, z, dt)
                    Ufd = fd.advance(Ufd, dt, args.dt_fine, h, h)
                times.append(s * dt)
                r_ref.append(burnt_radius(ref[s:s + 1], h).item())
                r_net.append(burnt_radius(U.cpu(), h).item())
                r_fd.append(burnt_radius(Ufd, h).item() if torch.isfinite(Ufd).all() else float("nan"))
            err_T = (U.cpu()[0, 3] - ref[-1, 3]).abs().max().item()
            print(f"  dt = {dt:g} ns ({len(ref) - 1} steps to {times[-1]:g} ns): front speed (m/s) "
                  f"fine->avg {front_speed(times, r_ref, r_domain):.1f}, network {front_speed(times, r_net, r_domain):.1f}, "
                  f"coarse FD {front_speed(times, r_fd, r_domain):.1f}; final burnt radius {r_ref[-1]:.1f} / "
                  f"{r_net[-1]:.1f} / {r_fd[-1]:.1f} nm; max |T error| {err_T:.0f} K")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    f = 2 ** args.levels
    if args.n % f:
        raise SystemExit(f"--n must be divisible by 2**levels = {f}")
    snaps(args.t_end, args.dt_snap)
    snaps(args.start_every, args.dt_snap)
    print(f"fine {args.n}^2 at {args.dx:g} nm -> coarse {args.n // f}^2 at {f * args.dx:g} nm; snapshots every "
          f"{args.dt_snap:g} ns to {args.t_end:g} ns; training dt {args.dt_range[0]:g}-{args.dt_range[1]:g} ns",
          flush=True)
    data = load_or_generate(args, f)
    encoder, stepper = train(args, data, f)
    torch.save({"encoder": encoder.state_dict(), "stepper": stepper.state_dict(),
                "args": {**vars(args), "model_version": MODEL_VERSION}}, args.out)
    print(f"saved {args.out}")
    evaluate(args, data, f, encoder, stepper)


if __name__ == "__main__":
    main()
