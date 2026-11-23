"""Use a trained coarse stepper (train_coarse.py) to predict a hotspot, and compare it with FD.

For one circular hotspot this script
  1. builds the fine initial field (same as kinetics2D.C --circular, optionally burnt core),
  2. predicts: encodes the fine initial field once, then marches the coarse stepper,
  3. (unless --no-fd) solves the same hotspot with the exact FD model on the fine grid,
     and with the same FD model directly on the coarse grid,
  4. writes figures: T and Y3 maps at several times (fine FD, fine FD averaged to the
     coarse grid, network, coarse FD) and the burnt radius vs time.

The fine grid spacing and coarse factor come from the checkpoint; the domain size
(--n) and the hotspot can be anything, including cases outside the training ranges.

Example:
  python ml/predict_coarse.py --model coarse_stepper.pt --radius 30 --t-hot 1800 --output hs30
  python ml/predict_coarse.py --model coarse_stepper.pt --radius 15 --t-hot 1600 --steps 40 --n 512
  python ml/predict_coarse.py --model coarse_stepper_v3.pt --radius 25 --t-hot 1800 --dt 0.25   # dt-conditional

Use from Python:
  from predict_coarse import load_model, predict
  encoder, stepper, cfg = load_model("coarse_stepper.pt")
  coarse = predict(encoder, stepper, cfg, U0_fine, steps, dt=0.1)   # (steps+1, 4, n/f, n/f)
"""
import argparse
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

import fd_kinetics as fd
from train_coarse import Encoder, Stepper, burnt_radius, front_speed, restrict, T_COLD

FIELDS = {"T": 3, "Y3": 2}


def load_model(path, device="cpu"):
    """Rebuild encoder and stepper from a train_coarse.py checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["args"]
    h = 2 ** cfg["levels"] * cfg["dx"]
    encoder = Encoder(cfg["levels"], cfg["width"], cfg["latent"])
    # model_version 1: checkpoints trained before exact-zero reactions / no free heat source
    stepper = Stepper(cfg["latent"], cfg["width"], cfg["depth"], cfg["dt_coarse"], h,
                      version=cfg.get("model_version", 1))
    encoder.load_state_dict(checkpoint["encoder"])
    stepper.load_state_dict(checkpoint["stepper"])
    return encoder.eval().to(device), stepper.eval().to(device), cfg


@torch.no_grad()
def predict(encoder, stepper, cfg, U0_fine, steps, dt=None, device="cpu"):
    """Coarse trajectory (steps + 1, 4, n/f, n/f) from a fine initial field (4, n, n).
    dt: step in ns (model_version 3 only; older models use their training step)."""
    f = 2 ** cfg["levels"]
    U0 = torch.as_tensor(U0_fine, dtype=torch.float32, device=device)
    U = restrict(U0, f)[None]
    z = encoder(U0[None])
    out = [U[0].cpu()]
    for _ in range(steps):
        U, z = stepper(U, z, dt)
        out.append(U[0].cpu())
    return torch.stack(out)


@torch.no_grad()
def fd_trajectory(U0, steps, dt, dt_fine, h, f=None):
    """Exact FD (fd_kinetics) every dt for `steps` steps; returns fine snapshots and,
    if f is given, their box averages."""
    U = torch.as_tensor(U0, dtype=torch.float64)
    fine, coarse = [U.float()], [restrict(U, f).float()] if f else None
    for s in range(steps):
        U = fd.advance(U, dt, dt_fine, h, h)
        if not torch.isfinite(U).all():
            raise RuntimeError(f"FD run became non-finite at t = {(s + 1) * dt} ns; lower --dt-fine")
        fine.append(U.float())
        if f:
            coarse.append(restrict(U, f).float())
    return torch.stack(fine), (torch.stack(coarse) if f else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("coarse_stepper.pt"))
    p.add_argument("--radius", type=float, default=25.0, help="hotspot radius, nm")
    p.add_argument("--t-hot", type=float, default=1800.0, help="hotspot temperature, K")
    p.add_argument("--burnt-core", action="store_true", help="start the hotspot reacted (Y3 = 1)")
    p.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], help="offset from the domain center, nm")
    p.add_argument("--n", type=int, help="fine nodes per side (default: as in training)")
    p.add_argument("--steps", type=int, help="coarse steps (default: the training trajectory length)")
    p.add_argument("--dt", type=float, help="network step, ns (model_version 3; default: the reference step)")
    p.add_argument("--dt-fine", type=float, help="fine FD step, ns (default: as in training)")
    p.add_argument("--frames", type=int, default=4, help="times shown in the map figures")
    p.add_argument("--no-fd", action="store_true", help="prediction only: skip both FD solutions")
    p.add_argument("--output", default="prediction", help="prefix for the figures and the .npz")
    p.add_argument("--show", action="store_true", help="also open the figures")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoder, stepper, cfg = load_model(args.model)
    f, dx = 2 ** cfg["levels"], cfg["dx"]
    version = cfg.get("model_version", 1)
    dt = args.dt or cfg["dt_coarse"]
    if version < 3 and abs(dt - cfg["dt_coarse"]) > 1e-12:
        raise SystemExit(f"{args.model} (model_version {version}) only steps by {cfg['dt_coarse']:g} ns")
    if version >= 3 and not cfg["dt_range"][0] - 1e-12 <= dt <= cfg["dt_range"][1] + 1e-12:
        print(f"warning: dt = {dt:g} ns is outside the trained range {cfg['dt_range'][0]:g}-{cfg['dt_range'][1]:g} ns")
    h = f * dx
    n = args.n or cfg["n"]
    t_end = cfg["t_end"] if "t_end" in cfg else cfg["coarse_steps"] * cfg["dt_coarse"]
    steps = args.steps or max(1, round(t_end / dt))
    dt_fine = args.dt_fine or cfg["dt_fine"]
    if n % f:
        raise SystemExit(f"--n must be divisible by the coarse factor {f}")
    nc, L = n // f, n * dx
    times = np.arange(steps + 1) * dt
    print(f"fine {n}^2 at {dx:g} nm -> coarse {nc}^2 at {h:g} nm; {steps} steps of {dt:g} ns; "
          f"hotspot r = {args.radius:g} nm, T = {args.t_hot:g} K{' (burnt core)' if args.burnt_core else ''}")

    U0 = fd.circular_hotspot(n, n, dx, dx, args.radius, args.t_hot, T_COLD, args.burnt_core, tuple(args.center))
    start = time.time()
    net = predict(encoder, stepper, cfg, U0, steps, dt)
    print(f"network: {time.time() - start:.2f} s")

    runs = {"network": net}
    fine = None
    if not args.no_fd:
        start = time.time()
        fine, ref = fd_trajectory(U0, steps, dt, dt_fine, dx, f)
        print(f"fine FD: {time.time() - start:.1f} s")
        start = time.time()
        coarse_fd, _ = fd_trajectory(restrict(torch.as_tensor(U0), f), steps, dt, dt_fine, h)
        print(f"coarse FD: {time.time() - start:.1f} s")
        runs = {"fine FD (averaged)": ref, "network": net, "coarse FD": coarse_fd}

    # ---- burnt radius and front speed
    r_domain = L / math.sqrt(math.pi)
    radii = {name: [burnt_radius(U[s:s + 1], h).item() for s in range(steps + 1)] for name, U in runs.items()}
    print("front speed (m/s): " + ", ".join(f"{name} {front_speed(times, r, r_domain):.1f}"
                                            for name, r in radii.items()))
    if "fine FD (averaged)" in runs:
        err = (net[-1, 3] - runs["fine FD (averaged)"][-1, 3]).abs().max().item()
        print(f"max |T network - T fine FD averaged| at t = {times[-1]:g} ns: {err:.1f} K")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    styles = {"fine FD (averaged)": "k-", "network": "C0--", "coarse FD": "C3:"}
    for name, r in radii.items():
        ax.plot(times, r, styles[name], lw=2, label=f"{name}: {front_speed(times, r, r_domain):.0f} m/s")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("burnt radius (nm)")
    ax.set_title(f"r0 = {args.radius:g} nm, T0 = {args.t_hot:g} K, coarse cell {h:g} nm")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.output}_radius.png", dpi=200)
    print(f"saved {args.output}_radius.png")

    # ---- maps
    frames = np.unique(np.linspace(0, steps, args.frames).astype(int))
    columns = ([("fine FD", fine, dx)] if fine is not None else []) + [(name, U, h) for name, U in runs.items()]
    extent = [-L / 2, L / 2, -L / 2, L / 2]
    for field, k in FIELDS.items():
        vmax = max(U[:, k].max().item() for _, U, _ in columns)
        vmin, cmap = (T_COLD, "inferno") if field == "T" else (0.0, "viridis")
        vmax = vmax if field == "T" else 1.0
        fig, axes = plt.subplots(len(frames), len(columns), figsize=(2.6 * len(columns) + 0.8, 2.5 * len(frames)),
                                 squeeze=False, constrained_layout=True)
        for row, s in enumerate(frames):
            for col, (name, U, _) in enumerate(columns):
                ax = axes[row, col]
                im = ax.imshow(U[s, k].numpy().T, origin="lower", extent=extent, cmap=cmap,
                               vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"t = {times[s]:.3g} ns")
        fig.colorbar(im, ax=axes, shrink=0.8, label="T (K)" if field == "T" else "Y3")
        fig.savefig(f"{args.output}_{field}.png", dpi=200)
        print(f"saved {args.output}_{field}.png")

    np.savez_compressed(f"{args.output}.npz", times=times, **{name.split(" ")[0] + ("_avg" if "averaged" in name else ""): U.numpy()
                                                              for name, U in runs.items()})
    print(f"saved {args.output}.npz")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
