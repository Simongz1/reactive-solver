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
