"""Phase map for a pore collapse: RDX with one circular air pore.

The map points are the FV cell centers of an Nx x Ny grid on Lx x Ly, so use the
same Nx Ny Lx Ly when running the solver.

Example:
  python3 density_maps/generate_pore.py --nx 250 --ny 250 --lx 5000 --ly 5000 --radius 500
"""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nx", type=int, required=True)
parser.add_argument("--ny", type=int, required=True)
parser.add_argument("--lx", type=float, required=True, help="domain width in nm")
parser.add_argument("--ly", type=float, required=True, help="domain height in nm")
parser.add_argument("--radius", type=float, required=True, help="pore radius in nm")
parser.add_argument("--cx", type=float, help="pore center x in nm (default lx/4)")
parser.add_argument("--cy", type=float, help="pore center y in nm (default ly/2)")
parser.add_argument("--rdx-density", type=float, default=1820e-9, help="solver units (kg/m^3 * 1e-9)")
parser.add_argument("--pore-density", type=float, default=1.2e-9, help="solver units (kg/m^3 * 1e-9)")
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pore.csv"))
args = parser.parse_args()

cx = args.lx / 4 if args.cx is None else args.cx
cy = args.ly / 2 if args.cy is None else args.cy
dx, dy = args.lx / args.nx, args.ly / args.ny

pore_cells = 0
with args.output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["x", "y", "rho", "phase"])
    for j in range(args.ny):
        y = (j + 0.5) * dy
        for i in range(args.nx):
            x = (i + 0.5) * dx
            if (x - cx) ** 2 + (y - cy) ** 2 <= args.radius ** 2:
                writer.writerow([f"{x:.12g}", f"{y:.12g}", f"{args.pore_density:.12g}", "pore"])
                pore_cells += 1
            else:
                writer.writerow([f"{x:.12g}", f"{y:.12g}", f"{args.rdx_density:.12g}", "rdx"])

print(f"{args.output}: {args.nx} x {args.ny} cells of {dx:g} x {dy:g} nm, "
      f"pore at ({cx:g}, {cy:g}) with radius {args.radius:g} nm ({pore_cells} cells)")
