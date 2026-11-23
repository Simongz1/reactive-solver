"""Advance a random field of hotspots with a trained coarse stepper (train_coarse.py).

The network only works at the resolution it was trained at (coarse cell h = 2**levels * dx,
step dt_coarse, from the checkpoint). This script builds a domain of Lx x Ly nm on that
coarse grid, seeds random circular hotspots, and marches the network:

  1. The coarse state starts as unreacted material at --t-cold (Y1 = 1).
  2. Every hotspot is drawn on a fine patch (spacing dx) around it, averaged onto the coarse
     grid, and encoded into the latent channels, exactly as in training. Patches extend
     --margin coarse cells beyond the hotspot so the encoder sees its full neighbourhood;
     away from hotspots the latent is that of uniform cold material. So no fine field of
     the whole domain is ever built.
  3. The stepper advances the whole coarse field --steps times (dt_coarse each).

Outputs (prefix --output): <prefix>.npz with Y3 and T every --save-every steps (float32),
<prefix>_maps.png (T and Y3 at a few times) and <prefix>_burnt.png (burnt area vs time).

Example (4 x 2 um, 40 hotspots, 10 ns):
  python ml/run_field.py --model coarse_stepper_v2.pt --lx 4000 --ly 2000 --hotspots 40 \
      --steps 200 --save-every 20 --output field

Cost: the state is (4 + latent) float32 values per coarse cell and the stepper keeps a few
(width)-channel activations, so memory ~ cells * width * 4 B * ~4. The script prints the
estimate and stops above --max-gb.
"""
import argparse
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from predict_coarse import load_model
from train_coarse import restrict

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", type=Path, default=Path("coarse_stepper_v2.pt"))
parser.add_argument("--lx", type=float, required=True, help="domain width, nm (rounded to whole coarse cells)")
parser.add_argument("--ly", type=float, required=True, help="domain height, nm")
parser.add_argument("--hotspots", type=int, default=20, help="number of random hotspots")
parser.add_argument("--radius", type=float, nargs=2, default=[10.0, 40.0], help="radius range, nm")
parser.add_argument("--t-hot", type=float, nargs=2, default=[1500.0, 2000.0], help="temperature range, K")
parser.add_argument("--burnt-fraction", type=float, default=0.0, help="fraction of hotspots that start reacted (Y3 = 1)")
parser.add_argument("--t-hot-burnt", type=float, nargs=2, help="temperature range of reacted hotspots, K")
parser.add_argument("--t-cold", type=float, default=300.0, help="background temperature, K")
parser.add_argument("--steps", type=int, default=100, help="coarse steps")
parser.add_argument("--dt", type=float, help="network step, ns (model_version 3; default: the reference step)")
parser.add_argument("--save-every", type=int, default=10)
parser.add_argument("--margin", type=int, default=4, help="extra coarse cells around each hotspot patch")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max-gb", type=float, default=8.0, help="refuse to run above this memory estimate")
parser.add_argument("--output", default="field")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device(args.device)
encoder, stepper, cfg = load_model(args.model, device)
f, dx = 2 ** cfg["levels"], cfg["dx"]
h = f * dx
dt = args.dt or cfg["dt_coarse"]
if cfg.get("model_version", 1) < 3 and abs(dt - cfg["dt_coarse"]) > 1e-12:
    raise SystemExit(f"{args.model} (model_version {cfg.get('model_version', 1)}) only steps by {cfg['dt_coarse']:g} ns")
if "dt_range" in cfg and not cfg["dt_range"][0] - 1e-12 <= dt <= cfg["dt_range"][1] + 1e-12:
    print(f"warning: dt = {dt:g} ns is outside the trained range {cfg['dt_range'][0]:g}-{cfg['dt_range'][1]:g} ns")
nx, ny = round(args.lx / h), round(args.ly / h)
latent = cfg["latent"]
memory_gb = nx * ny * cfg["width"] * 4 * 4 / 1e9
print(f"model {args.model}: coarse cell {h:g} nm, dt {dt:g} ns, latent {latent}")
print(f"domain {nx * h:g} x {ny * h:g} nm = {nx} x {ny} cells ({nx * ny:.3g}); "
      f"{args.steps} steps = {args.steps * dt:g} ns; estimated memory {memory_gb:.2f} GB")
if memory_gb > args.max_gb:
    raise SystemExit(f"estimated {memory_gb:.1f} GB > --max-gb {args.max_gb:g}; use a smaller domain")

# ---- random hotspots (circles in nm, domain origin at the lower-left corner)
rng = np.random.default_rng(args.seed)
spots = []
for _ in range(args.hotspots):
    r = rng.uniform(*args.radius)
    burnt = rng.uniform() < args.burnt_fraction
    T = rng.uniform(*(args.t_hot_burnt if burnt and args.t_hot_burnt else args.t_hot))
    cx, cy = rng.uniform(r, nx * h - r), rng.uniform(r, ny * h - r)
    spots.append((cx, cy, r, T, burnt))


def fine_patch(i0, i1, j0, j1):
    """Fine field (4, ni*f, nj*f) of coarse cells [i0, i1) x [j0, j1), all hotspots drawn.
    Fine node k sits at k*dx, so coarse cell c averages nodes c*f ... c*f + f - 1 (as in training)."""
    x = (i0 * f + np.arange((i1 - i0) * f)) * dx
    y = (j0 * f + np.arange((j1 - j0) * f)) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.zeros((4,) + X.shape)
    U[0] = 1.0
    U[3] = args.t_cold
    for cx, cy, r, T, burnt in spots:
        if cx + r < x[0] or cx - r > x[-1] or cy + r < y[0] or cy - r > y[-1]:
            continue
        hot = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        if burnt:
            U[0][hot], U[2][hot] = 0.0, 1.0
        else:
            U[0][hot], U[2][hot] = 1.0, 0.0
        U[3][hot] = np.maximum(U[3][hot], T)
    return torch.tensor(U, dtype=torch.float32, device=device)


@torch.no_grad()
def initial_state():
    U = torch.zeros(4, nx, ny, device=device)
    U[0] = 1.0
    U[3] = args.t_cold
    # latent of uniform cold material (constant: the encoder is translation invariant)
    uniform = torch.tensor([1.0, 0, 0, args.t_cold], device=device).view(1, 4, 1, 1).expand(1, 4, 2 * f, 2 * f)
    background = encoder(uniform)[0, :, 0, 0]
    z = background.view(latent, 1, 1).expand(latent, nx, ny).clone()
    m = args.margin
    for cx, cy, r, _, _ in spots:
        # coarse cells touched by the hotspot, and the encoder patch around them
        a0, a1 = max(0, int((cx - r) // h)), min(nx, int((cx + r) // h) + 1)
        b0, b1 = max(0, int((cy - r) // h)), min(ny, int((cy + r) // h) + 1)
        i0, i1, j0, j1 = max(0, a0 - m), min(nx, a1 + m), max(0, b0 - m), min(ny, b1 + m)
        patch = fine_patch(i0, i1, j0, j1)
        U[:, a0:a1, b0:b1] = restrict(patch, f)[:, a0 - i0:a1 - i0, b0 - j0:b1 - j0]
        z[:, a0:a1, b0:b1] = encoder(patch[None])[0, :, a0 - i0:a1 - i0, b0 - j0:b1 - j0]
    return U[None], z[None]


start = time.time()
U, z = initial_state()
print(f"seeded {len(spots)} hotspots ({sum(s[4] for s in spots)} reacted) in {time.time() - start:.1f} s")

times, burnt_area, snaps_T, snaps_Y3, snap_times = [], [], [], [], []
start = time.time()
with torch.no_grad():
    for s in range(args.steps + 1):
        if s > 0:
            U, z = stepper(U, z, dt)
        times.append(s * dt)
        burnt_area.append(U[0, 2].sum().item() * h * h)
        if s % args.save_every == 0 or s == args.steps:
            snaps_T.append(U[0, 3].cpu().numpy().copy())
            snaps_Y3.append(U[0, 2].cpu().numpy().copy())
            snap_times.append(s * dt)
            print(f"t = {s * dt:7.3f} ns: burnt area {burnt_area[-1] / 1e6:.4f} um^2, "
                  f"T [{U[0, 3].min().item():.0f}, {U[0, 3].max().item():.0f}] K, "
                  f"{time.time() - start:.1f} s elapsed")

np.savez_compressed(f"{args.output}.npz", times=np.array(snap_times), T=np.array(snaps_T),
                    Y3=np.array(snaps_Y3), h=h, hotspots=np.array(spots, dtype=float))
print(f"saved {args.output}.npz")

# ---- figures
frames = np.unique(np.linspace(0, len(snap_times) - 1, min(4, len(snap_times))).astype(int))
extent = [0, nx * h / 1000, 0, ny * h / 1000]
fig, axes = plt.subplots(len(frames), 2, figsize=(11, 2.8 * len(frames) * ny / nx + 1.2 * len(frames)),
                         squeeze=False, constrained_layout=True)
Tmax = max(a.max() for a in snaps_T)
for row, k in enumerate(frames):
    imT = axes[row, 0].imshow(snaps_T[k].T, origin="lower", extent=extent, cmap="inferno",
                              vmin=args.t_cold, vmax=Tmax, interpolation="nearest")
    imY = axes[row, 1].imshow(snaps_Y3[k].T, origin="lower", extent=extent, cmap="viridis",
                              vmin=0, vmax=1, interpolation="nearest")
    axes[row, 0].set_ylabel(f"t = {snap_times[k]:.3g} ns\ny (um)")
for ax in axes[-1]:
    ax.set_xlabel("x (um)")
axes[0, 0].set_title("T (K)")
axes[0, 1].set_title("Y3")
fig.colorbar(imT, ax=axes[:, 0], shrink=0.8)
fig.colorbar(imY, ax=axes[:, 1], shrink=0.8)
fig.savefig(f"{args.output}_maps.png", dpi=200)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(times, np.array(burnt_area) / 1e6)
ax.set_xlabel("time (ns)")
ax.set_ylabel("burnt area (um$^2$)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{args.output}_burnt.png", dpi=200)
print(f"saved {args.output}_maps.png, {args.output}_burnt.png")
if args.show:
    plt.show()
