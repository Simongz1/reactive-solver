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
        self.model = Stepper(cfg["width"], cfg["depth"], cfg["dt_ref"], cfg["h_ref"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.dt_min, self.dt_max = cfg["dt_range"]
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


def main():
    """Command line demo: march one hotspot and save figures.

    python ml/surrogate.py --model increment_surrogate.pt --nx 64 --ny 64 --h 16 --dt 0.1 --steps 50
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="increment_surrogate.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--h", type=float, nargs="+", default=[16.0], help="cell size, nm (one value, or hx hy)")
    p.add_argument("--dt", type=float, default=0.1, help="step, ns")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--radius", type=float, default=40.0, help="hotspot radius, nm")
    p.add_argument("--t-hot", type=float, default=1800.0)
    p.add_argument("--t-cold", type=float, default=300.0)
    p.add_argument("--burnt", action="store_true", help="start the hotspot reacted (Y3 = 1)")
    p.add_argument("--frames", type=int, default=4, help="snapshots in the figure")
    p.add_argument("--output", default="surrogate_run")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = KineticsSurrogate(args.model, args.device)
    hx, hy = (args.h[0], args.h[0]) if len(args.h) == 1 else args.h[:2]
    print(f"model {args.model}: trained h {model.h_range[0]:g}-{model.h_range[1]:g} nm, "
          f"dt {model.dt_min:g}-{model.dt_max:g} ns; running {args.nx} x {args.ny} cells of {hx:g} x {hy:g} nm, "
          f"dt {args.dt:g} ns ({model.substeps(args.dt, hx, hy, (args.nx, args.ny))} sub-steps each)")

    x = (np.arange(args.nx) + 0.5 - args.nx / 2) * hx
    y = (np.arange(args.ny) + 0.5 - args.ny / 2) * hy
    hot = x[:, None] ** 2 + y[None, :] ** 2 <= args.radius ** 2
    T = np.where(hot, args.t_hot, args.t_cold)
    Y = np.zeros((3, args.nx, args.ny))
    Y[0] = 1.0
    if args.burnt:
        Y[0][hot], Y[2][hot] = 0.0, 1.0

    times, radius, snaps = [], [], []
    keep = set(np.linspace(0, args.steps, args.frames).round().astype(int))
    for s in range(args.steps + 1):
        if s > 0:
            T, Y = model.predict(T, Y, args.dt, h=(hx, hy))
        times.append(s * args.dt)
        radius.append(math.sqrt(Y[2].sum() * hx * hy / math.pi))
        if s in keep:
            snaps.append((s * args.dt, T.copy(), Y[2].copy()))
            print(f"t = {s * args.dt:7.3f} ns: burnt radius {radius[-1]:6.1f} nm, T [{T.min():.0f}, {T.max():.0f}] K")

    extent = [x[0] - hx / 2, x[-1] + hx / 2, y[0] - hy / 2, y[-1] + hy / 2]
    fig, axes = plt.subplots(2, len(snaps), figsize=(3 * len(snaps), 6), squeeze=False, constrained_layout=True)
    for k, (t, Tk, Y3k) in enumerate(snaps):
        imT = axes[0, k].imshow(Tk.T, origin="lower", extent=extent, cmap="inferno",
                                vmin=args.t_cold, vmax=max(s[1].max() for s in snaps))
        imY = axes[1, k].imshow(Y3k.T, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
        axes[0, k].set_title(f"t = {t:.3g} ns")
    axes[0, 0].set_ylabel("T (K)\ny (nm)")
    axes[1, 0].set_ylabel("Y3\ny (nm)")
    fig.colorbar(imT, ax=axes[0])
    fig.colorbar(imY, ax=axes[1])
    fig.savefig(f"{args.output}_maps.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.plot(times, radius)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("burnt radius (nm)")
    ax.grid(alpha=0.3)
    fig.savefig(f"{args.output}_radius.png", dpi=150)
    np.savez_compressed(f"{args.output}.npz", times=np.array(times), radius=np.array(radius), T=T, Y=Y)
    print(f"saved {args.output}_maps.png, {args.output}_radius.png, {args.output}.npz")


if __name__ == "__main__":
    main()
