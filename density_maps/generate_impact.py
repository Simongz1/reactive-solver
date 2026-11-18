"""Cell-center phase map for an RDX cylinder impacting an end-supported PS film.
Coordinates: nm. Density: solver units (kg/m^3 times 1e-9).
"""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("impact.csv"))
parser.add_argument("--spacing", type=float, default=25, help="Square cell width in nm")
parser.add_argument("--height", type=float, default=12000)
parser.add_argument("--film-bottom", type=float, default=2000)
parser.add_argument("--gap", type=float, default=250)
args = parser.parse_args()
width, thickness, radius = 20000, 250, 3750
if args.spacing <= 0 or args.height <= 0 or args.gap < 0 or args.film_bottom < 0:
    parser.error("Invalid geometry or spacing")
nx, ny = round(width/args.spacing), round(args.height/args.spacing)
if nx < 1 or ny < 1 or abs(nx*args.spacing-width) > 1e-8 or abs(ny*args.spacing-args.height) > 1e-8:
    parser.error("Spacing must divide width and height")
cy = args.film_bottom + thickness + args.gap + radius
if cy + radius > args.height:
    parser.error("Impactor does not fit inside the domain")
counts = dict(rdx=0, ps=0, pore=0)
with args.output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["x", "y", "rho", "phase"])
    for j in range(ny):
        y = (j+.5)*args.spacing
        for i in range(nx):
            x = (i+.5)*args.spacing
            phase, rho = "pore", 1.2e-9
            if args.film_bottom <= y < args.film_bottom+thickness:
                phase, rho = "ps", 1020e-9
            elif (x-width/2)**2+(y-cy)**2 <= radius**2:
                phase, rho = "rdx", 1820e-9
            counts[phase] += 1
            writer.writerow([f"{x:g}", f"{y:g}", f"{rho:.12g}", phase])
print(f"{args.output}: {nx} x {ny}, spacing {args.spacing:g} nm; {counts}")
print("Use --rdx-velocity 0 -750 --open-boundaries --fixed-ps-ends 250 with piston velocity 0.")
