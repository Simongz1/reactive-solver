"""Call a trained increment surrogate (train_increment.py) like a function.

    from surrogate import KineticsSurrogate
    model = KineticsSurrogate("increment_surrogate.pt")
    T_new, Y_new = model.predict(T, Y, dt, h=(hx, hy))      # cell size in nm
    T_new, Y_new = model.predict(T, Y, dt, extent=(Lx, Ly)) # or the domain size: h = L / cells
    S_T, S_Y = model.rates(T, Y, dt, h=h)                   # (new - old) / dt, a residual source

or, with the model cached per file,

    from surrogate import predict
    T_new, Y_new = predict(T, Y, dt, h=16.0)

Inputs are cell averages on a uniform grid:
  T  (nx, ny) or (nx,) for 1D, K
  Y  (3, nx, ny) or (3, nx): Y1, Y2, Y3
  dt step, ns
numpy in -> numpy out, torch in -> torch out (same dtype and device as T). The grid shape
comes from the arrays; the physical cell size cannot (an array has no length), so pass h or
extent. Pass the current values every call: after a shock compresses the domain, the new h
is just a different input.

Steps outside what was trained are handled here:
  - dt larger than the trained dt range or than --courant front cells per step: split into
    equal sub-steps.
  - dt smaller than the smallest trained step dt_min: one dt_min step, scaled by dt / dt_min
    (the rate over dt_min), which is what a residual source needs.
  - h outside the trained cell sizes: a warning (the network extrapolates).
Boundaries are zero flux (insulated), as in training.
"""
import math
import warnings

import numpy as np
import torch

from train_increment import MODEL_KIND, S_REF, Stepper


class KineticsSurrogate:
    def __init__(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["args"]
        if cfg.get("model_kind") != MODEL_KIND:
            raise ValueError(f"{path} is not an increment surrogate (train_increment.py)")
        self.cfg = cfg
        self.device = torch.device(device)
        self.version = cfg.get("model_version", 1)
        self.model = Stepper(cfg["width"], cfg["depth"], cfg["dt_ref"], cfg["h_ref"],
                             version=self.version).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.dt_min = cfg["dt_range"][0]
        self.dt_max = cfg.get("dt_max", cfg["dt_range"][1])  # v2: synthetic large-cell steps go further
        self.courant = cfg["courant"]
        self.h_range = cfg["h_range"]
        self._warned = set()

    def _cell_size(self, shape, h, extent):
        if (h is None) == (extent is None):
            raise ValueError("pass exactly one of h=(hx, hy) or extent=(Lx, Ly)")
        if h is not None:
            hx, hy = (h, h) if np.isscalar(h) else h
        else:
            Lx, Ly = (extent, extent) if np.isscalar(extent) else extent
            hx, hy = Lx / shape[0], Ly / shape[1]
        return float(hx), float(hy)

    def substeps(self, dt, hx, hy, shape):
        """Number of equal sub-steps for dt: trained dt range and front Courant cap."""
        h_min = min(h for h, n in ((hx, shape[0]), (hy, shape[1])) if n > 1) if max(shape) > 1 else max(hx, hy)
        return max(1, math.ceil(dt / self.dt_max - 1e-9), math.ceil(S_REF * dt / (self.courant * h_min) - 1e-9))

    def _warn(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, stacklevel=3)

    @torch.no_grad()
    def predict(self, T, Y, dt, h=None, extent=None):
        """(T, Y) cell averages at t -> (T, Y) at t + dt. See the module docstring."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        is_numpy = not torch.is_tensor(T)
        T_t = torch.as_tensor(np.asarray(T) if is_numpy else T)
        Y_t = torch.as_tensor(np.asarray(Y) if is_numpy else Y)
        one_d = T_t.dim() == 1
        if T_t.dim() not in (1, 2) or Y_t.shape != (3, *T_t.shape):
            raise ValueError(f"expected T (nx, ny) or (nx,) and Y (3, *T.shape); got {tuple(T_t.shape)}, {tuple(Y_t.shape)}")
        U = torch.cat([Y_t, T_t[None]], 0)
        if one_d:
            U = U[..., None]
        shape = U.shape[-2:]
        hx, hy = self._cell_size(shape, h, extent)
        if one_d:
            hy = hx  # no y faces; the network sees a square-celled 1D strip
        lo, hi = self.h_range
        for value in ((hx,) if one_d else (hx, hy)):
            if not lo <= value <= hi:
                self._warn(("h", value), f"cell size {value:g} nm is outside the trained {lo:g}-{hi:g} nm")

        U = U.to(self.device, torch.float32)[None]
        if dt < self.dt_min:
            U_new = U + (dt / self.dt_min) * (self.model(U, self.dt_min, hx, hy) - U)
        else:
            steps = self.substeps(dt, hx, hy, shape)
            sub = dt / steps
            if sub < self.dt_min:
                self._warn(("sub", hx, hy), f"cell size {min(hx, hy):g} nm needs sub-steps of {sub:g} ns, "
                           f"below the trained {self.dt_min:g} ns")
            U_new = U
            for _ in range(steps):
                U_new = self.model(U_new, sub, hx, hy)

        U_new = U_new[0]
        if one_d:
            U_new = U_new[..., 0]
        T_new, Y_new = U_new[3], U_new[:3]
        if is_numpy:
            dtype = np.asarray(T).dtype if np.issubdtype(np.asarray(T).dtype, np.floating) else np.float64
            return T_new.cpu().numpy().astype(dtype), Y_new.cpu().numpy().astype(dtype)
        return T_new.to(T_t.dtype).to(T_t.device), Y_new.to(Y_t.dtype).to(Y_t.device)

    def rates(self, T, Y, dt, h=None, extent=None):
        """Average rates over the step, (T_new - T) / dt and (Y_new - Y) / dt (K/ns, 1/ns)."""
        T_new, Y_new = self.predict(T, Y, dt, h=h, extent=extent)
        return (T_new - T) / dt, (Y_new - Y) / dt


_models = {}


def predict(T, Y, dt, h=None, extent=None, model="increment_surrogate.pt", device="cpu"):
    """Module-level shortcut; loads `model` once and reuses it."""
    key = (str(model), str(device))
    if key not in _models:
        _models[key] = KineticsSurrogate(model, device)
    return _models[key].predict(T, Y, dt, h=h, extent=extent)


def seed_hotspots(nx, ny, hx, hy, spots, t_cold, sub=8):
    """Cell averages of circular hotspots on a cold, unreacted grid (numpy).

    spots: list of (cx, cy, r, T_hot, burnt) in nm, origin at the lower-left corner. Each cell
    gets the area fraction phi each hotspot covers (sub x sub samples per cell), so hotspots
    smaller than a cell become a partial-cell average. Where hotspots overlap, the hotter /
    more reacted value wins (not energy conserving, but only for the initial condition).
    """
    T = np.full((nx, ny), float(t_cold))
    Y = np.zeros((3, nx, ny))
    Y[0] = 1.0
    offsets = (np.arange(sub) + 0.5) / sub
    for cx, cy, r, t_hot, burnt in spots:
        i0, i1 = max(0, int((cx - r) // hx)), min(nx, int((cx + r) // hx) + 1)
        j0, j1 = max(0, int((cy - r) // hy)), min(ny, int((cy + r) // hy) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = ((np.arange(i0, i1)[:, None] + offsets[None, :]) * hx).reshape(-1)
        ys = ((np.arange(j0, j1)[:, None] + offsets[None, :]) * hy).reshape(-1)
        inside = (xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2 <= r * r
        phi = inside.reshape(i1 - i0, sub, j1 - j0, sub).mean(axis=(1, 3))
        cells = (slice(i0, i1), slice(j0, j1))
        T[cells] = np.maximum(T[cells], t_cold + phi * (t_hot - t_cold))
        if burnt:
            Y3 = np.maximum(Y[2][cells], phi)
            Y[2][cells], Y[1][cells], Y[0][cells] = Y3, 0.0, 1.0 - Y3
    return T, Y


def main():
    """Command line runs: one centered hotspot (default) or a random field of hotspots.

    One hotspot:
      python ml/surrogate.py --model increment_surrogate_v2.pt --nx 128 --ny 128 --h 500 --dt 0.5 \\
          --steps 40 --radius 3000 --burnt --output single
    Random field, 1000 x 500 um of 500 nm cells, 300 hotspots of 1-5 um, 30 % reacted:
      python ml/surrogate.py --model increment_surrogate_v2.pt --lx 1e6 --ly 5e5 --h 500 --dt 2 \\
          --steps 200 --hotspots 300 --radius 1000 5000 --t-hot 1500 2000 --burnt-fraction 0.3 \\
          --device cuda --output field
    """
    import argparse
    import time

    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="increment_surrogate.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--nx", type=int, default=64, help="cells in x (or give --lx)")
    p.add_argument("--ny", type=int, default=64, help="cells in y (or give --ly)")
    p.add_argument("--lx", type=float, help="domain width, nm (sets nx = lx / hx)")
    p.add_argument("--ly", type=float, help="domain height, nm (sets ny = ly / hy)")
    p.add_argument("--h", type=float, nargs="+", default=[16.0], help="cell size, nm (one value, or hx hy)")
    p.add_argument("--dt", type=float, default=0.1, help="step, ns")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--hotspots", type=int, default=1,
                   help="1: one hotspot at the center; more: random positions, sizes and temperatures")
    p.add_argument("--radius", type=float, nargs="+", default=[40.0], help="hotspot radius, nm (value or min max)")
    p.add_argument("--t-hot", type=float, nargs="+", default=[1800.0], help="hotspot temperature, K (value or min max)")
    p.add_argument("--t-cold", type=float, default=300.0)
    p.add_argument("--burnt", action="store_true", help="start every hotspot reacted (Y3 = 1)")
    p.add_argument("--burnt-fraction", type=float, default=0.0, help="random field: fraction of hotspots reacted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=4, help="snapshots in the figure")
    p.add_argument("--max-gb", type=float, default=16.0, help="stop if the memory estimate is larger")
    p.add_argument("--output", default="surrogate_run")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = KineticsSurrogate(args.model, args.device)
    hx, hy = (args.h[0], args.h[0]) if len(args.h) == 1 else args.h[:2]
    nx = round(args.lx / hx) if args.lx else args.nx
    ny = round(args.ly / hy) if args.ly else args.ny
    memory_gb = nx * ny * model.cfg["width"] * 4 * 6 / 1e9
    print(f"model {args.model} (version {model.version}): trained h {model.h_range[0]:g}-{model.h_range[1]:g} nm, "
          f"dt {model.dt_min:g}-{model.dt_max:g} ns")
    print(f"domain {nx * hx / 1000:g} x {ny * hy / 1000:g} um = {nx} x {ny} cells of {hx:g} x {hy:g} nm; "
          f"dt {args.dt:g} ns ({model.substeps(args.dt, hx, hy, (nx, ny))} sub-steps each), "
          f"{args.steps} steps = {args.steps * args.dt:g} ns; estimated memory {memory_gb:.2f} GB")
    if memory_gb > args.max_gb:
        raise SystemExit(f"estimated {memory_gb:.1f} GB > --max-gb {args.max_gb:g}; use fewer or larger cells")

    rng = np.random.default_rng(args.seed)
    Lx, Ly = nx * hx, ny * hy
    if args.hotspots == 1:
        spots = [(Lx / 2, Ly / 2, args.radius[0], args.t_hot[0], args.burnt)]
    else:
        spots = []
        for _ in range(args.hotspots):
            r = rng.uniform(*(args.radius * 2)[:2])
            spots.append((rng.uniform(0, Lx), rng.uniform(0, Ly), r, rng.uniform(*(args.t_hot * 2)[:2]),
                          args.burnt or rng.uniform() < args.burnt_fraction))
    subgrid = sum(s[2] < 0.5 * min(hx, hy) for s in spots)
    if subgrid:
        print(f"note: {subgrid} of {len(spots)} hotspots are smaller than half a cell; they become a small "
              f"partial-cell temperature rise and will not ignite like the real hotspot would")
    T, Y = seed_hotspots(nx, ny, hx, hy, spots, args.t_cold)
    print(f"seeded {len(spots)} hotspots ({sum(s[4] for s in spots)} reacted)")

    # keep the state on the model's device between steps (no copies each call)
    T = torch.tensor(T, dtype=torch.float32, device=model.device)
    Y = torch.tensor(Y, dtype=torch.float32, device=model.device)
    times, burnt_area, snaps = [], [], []
    keep = set(np.linspace(0, args.steps, args.frames).round().astype(int))
    start = time.time()
    for s in range(args.steps + 1):
        if s > 0:
            T, Y = model.predict(T, Y, args.dt, h=(hx, hy))
        times.append(s * args.dt)
        burnt_area.append(Y[2].sum().item() * hx * hy)
        if s in keep:
            snaps.append((s * args.dt, T.cpu().numpy(), Y[2].cpu().numpy()))
            print(f"t = {s * args.dt:8.3f} ns: burnt area {burnt_area[-1] / 1e6:10.4g} um^2 "
                  f"({burnt_area[-1] / (Lx * Ly):6.1%}), T [{T.min().item():.0f}, {T.max().item():.0f}] K, "
                  f"{time.time() - start:.1f} s", flush=True)

    extent = [0, Lx / 1000, 0, Ly / 1000]
    rows = len(snaps)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * (4.2 * Ly / Lx + 0.6) + 0.5),
                             squeeze=False, constrained_layout=True)
    Tmax = max(s[1].max() for s in snaps)
    for k, (t, Tk, Y3k) in enumerate(snaps):
        imT = axes[k, 0].imshow(Tk.T, origin="lower", extent=extent, cmap="inferno", vmin=args.t_cold,
                                vmax=Tmax, interpolation="nearest")
        imY = axes[k, 1].imshow(Y3k.T, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1,
                                interpolation="nearest")
        axes[k, 0].set_ylabel(f"t = {t:.3g} ns\ny (um)")
    axes[0, 0].set_title("T (K)")
    axes[0, 1].set_title("Y3")
    for ax in axes[-1]:
        ax.set_xlabel("x (um)")
    fig.colorbar(imT, ax=axes[:, 0], shrink=0.8)
    fig.colorbar(imY, ax=axes[:, 1], shrink=0.8)
    fig.savefig(f"{args.output}_maps.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    if len(spots) == 1:
        ax.plot(times, np.sqrt(np.array(burnt_area) / math.pi))
        ax.set_ylabel("burnt radius (nm)")
    else:
        ax.plot(times, np.array(burnt_area) / (Lx * Ly))
        ax.set_ylabel("burnt area fraction")
    ax.set_xlabel("time (ns)")
    ax.grid(alpha=0.3)
    fig.savefig(f"{args.output}_burnt.png", dpi=150)
    np.savez_compressed(f"{args.output}.npz", times=np.array(times), burnt_area=np.array(burnt_area),
                        hotspots=np.array(spots, dtype=float), h=np.array([hx, hy]),
                        snapshot_times=np.array([s[0] for s in snaps]),
                        T=np.array([s[1] for s in snaps], dtype=np.float32),
                        Y3=np.array([s[2] for s in snaps], dtype=np.float32))
    print(f"saved {args.output}_maps.png, {args.output}_burnt.png, {args.output}.npz")


if __name__ == "__main__":
    main()
