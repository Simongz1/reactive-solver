"""Temperature-area curves from a 2D Exodus output.

For each selected time, A(T*) is the material area with temperature above T*:
    A(T*) = sum over cells with T > T* of material_fraction * dx * dy
Plotted with T* on the vertical axis and A on a log horizontal axis.
Void cells (material_fraction = 0, written with T = 0) do not contribute, and neither do
nearly empty cells at the pore surface (material_fraction < --min-fraction), whose
temperature is not meaningful.

Example:
  python plot_temperature_area.py pore_sharper.e
  python plot_temperature_area.py pore_sharper.e --times 0.005 0.01 0.015 --output TA.png
  python plot_temperature_area.py pore500hydro.e --tmax 0.65 --count 8
"""
import argparse

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("exodus", help="Exodus file with nodal T (and material_fraction or alpha_* if present)")
parser.add_argument("--times", type=float, nargs="+", help="times to plot, ns (nearest saved step); default: 6 evenly spaced")
parser.add_argument("--count", type=int, default=6, help="number of evenly spaced times when --times is not given")
parser.add_argument("--tmax", type=float, help="latest time for the evenly spaced times, ns")
parser.add_argument("--tmin", type=float, default=300.0, help="lowest temperature shown, K")
parser.add_argument("--min-fraction", type=float, default=0.01, help="ignore cells with less material than this")
parser.add_argument("--output", help="save the figure here instead of showing it")
args = parser.parse_args()

data = netCDF4.Dataset(args.exodus)
names = [b"".join(n).decode("latin1").strip("\x00 ") for n in data.variables["name_nod_var"][:]]


def field(name, step):
    return np.asarray(data.variables[f"vals_nod_var{names.index(name) + 1}"][step], dtype=float)


x = np.asarray(data.variables["coordx"][:])
y = np.asarray(data.variables["coordy"][:])
dx = np.diff(np.unique(x)).min()
dy = np.diff(np.unique(y)).min()
times = np.asarray(data.variables["time_whole"][:])

# fraction of each cell occupied by condensed material
if "material_fraction" in names:
    fraction = lambda step: field("material_fraction", step)
elif "alpha_pore" in names:
    fraction = lambda step: 1.0 - field("alpha_pore", step)
else:
    fraction = lambda step: np.ones_like(x, dtype=float)

if args.times:
    steps = sorted({int(np.abs(times - t).argmin()) for t in args.times})
else:
    last = len(times) - 1 if args.tmax is None else int(np.searchsorted(times, args.tmax, side="right")) - 1
    if last < 1:
        raise SystemExit(f"no saved step after t = 0 up to --tmax {args.tmax} ns")
    steps = sorted(set(np.linspace(1, last, args.count).astype(int)))

fig, ax = plt.subplots(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(steps)))
for color, step in zip(colors, steps):
    T = field("T", step)
    phi = fraction(step)
    area = phi * dx * dy
    keep = (phi >= args.min_fraction) & (T >= args.tmin)
    order = np.argsort(T[keep])[::-1]  # hottest first
    T_sorted = T[keep][order]
    A_above = np.cumsum(area[keep][order])  # area with T >= T_sorted[i]
    ax.semilogx(A_above, T_sorted, color=color, lw=1.6, label=f"t = {times[step]:.4g} ns")

ax.set_xlabel("Area above temperature (nm$^2$)")
ax.set_ylabel("Temperature (K)")
ax.set_title(args.exodus)
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
if args.output:
    fig.savefig(args.output, dpi=200)
    print(f"saved {args.output}")
else:
    plt.show()
