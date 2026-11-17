"""Map scattered Points:0,Points:1,c samples to FV cell centers.

Example: python3 density_maps/generate_map.py --nx 250 --ny 250 --lx 5000 --ly 5000
Coordinates and densities must already use solver units. No plotting dependencies.
"""
import argparse
import csv
import math
from pathlib import Path

DENSITIES = {'ps': 1020e-9, 'pore': 1e-9, 'rdx': 1820e-9}


def phase_from_c(c):
    # Include exact threshold values: the original masks omitted both.
    if c < 0.1:
        return 'ps'
    if c > 0.95:
        return 'pore'
    return 'rdx'


def read_points(filename):
    points = []
    with filename.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Points:0', 'Points:1', 'c'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{filename} must contain {sorted(required)}')
        for line, row in enumerate(reader, 2):
            x, y, c = (float(row[key]) for key in ('Points:0', 'Points:1', 'c'))
            if not all(math.isfinite(v) for v in (x, y, c)):
                raise ValueError(f'Nonfinite sample on line {line}')
            points.append((x, y, phase_from_c(c)))
    if not points:
        raise ValueError('Input contains no samples')
    # A projected 3D cloud needs an explicit slice rather than silently mixing planes.
    seen = {}
    for x, y, phase in points:
        if (x, y) in seen and seen[x, y] != phase:
            raise ValueError(f'Conflicting phases at ({x}, {y}); select a single slice first')
        seen[x, y] = phase
    return [(x, y, phase) for (x, y), phase in seen.items()]


class NearestPoint:
    """Balanced 2D k-d tree; exact nearest neighbors without SciPy dependency."""
    def __init__(self, points):
        self.points = points

        def build(indices, depth=0):
            if not indices:
                return None
            axis = depth % 2
            indices.sort(key=lambda i: (points[i][axis], i))
            middle = len(indices) // 2
            return (indices[middle], axis,
                    build(indices[:middle], depth + 1),
                    build(indices[middle + 1:], depth + 1))

        self.root = build(list(range(len(points))))

    def query(self, x, y):
        best = [math.inf, len(self.points)]

        def visit(node):
            if node is None:
                return
            index, axis, left, right = node
            px, py, _ = self.points[index]
            distance2 = (x - px)**2 + (y - py)**2
            if (distance2, index) < tuple(best):
                best[:] = distance2, index
            delta = (x - px) if axis == 0 else (y - py)
            near, far = (left, right) if delta <= 0 else (right, left)
            visit(near)
            if delta * delta <= best[0]:
                visit(far)

        visit(self.root)
        return self.points[best[1]][2], math.sqrt(best[0])


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=folder / 'damage_field.csv')
    parser.add_argument('--output', type=Path, default=folder / 'rho_field.csv')
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--xmin', type=float, help='Source rectangle origin; default: min sample x')
    parser.add_argument('--ymin', type=float, help='Source rectangle origin; default: min sample y')
    parser.add_argument('--lx', type=float, help='Domain width; default: full sample x span')
    parser.add_argument('--ly', type=float, help='Domain height; default: full sample y span')
    parser.add_argument('--max-distance', type=float, help='Reject cells farther than this from any sample')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must differ')
    if args.nx < 2 or args.ny < 2:
        parser.error('Require at least two cells per direction')
    points = read_points(args.input)
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    origin_x = xmin if args.xmin is None else args.xmin
    origin_y = ymin if args.ymin is None else args.ymin
    lx = xmax - origin_x if args.lx is None else args.lx
    ly = ymax - origin_y if args.ly is None else args.ly
    if not all(math.isfinite(v) for v in (origin_x, origin_y, lx, ly)) or min(lx, ly) <= 0:
        parser.error('Domain origin and dimensions must be finite; lengths must be positive')
    if origin_x < xmin or origin_y < ymin or origin_x + lx > xmax or origin_y + ly > ymax:
        parser.error('Requested rectangle extends outside source bounding box')
    if args.max_distance is not None and (not math.isfinite(args.max_distance) or args.max_distance <= 0):
        parser.error('--max-distance must be finite and positive')
    print(f'Mapping {len(points)} unique source points to {args.nx} x {args.ny} cells')
    tree = NearestPoint(points)
    rows = []
    counts = dict.fromkeys(DENSITIES, 0)
    max_distance = 0
    for j in range(args.ny):
        y = (j + 0.5) * ly / args.ny
        for i in range(args.nx):
            x = (i + 0.5) * lx / args.nx
            phase, distance = tree.query(origin_x + x, origin_y + y)
            if args.max_distance is not None and distance > args.max_distance:
                raise ValueError(f'Cell ({i}, {j}) is {distance:g} from nearest source sample')
            max_distance = max(max_distance, distance)
            counts[phase] += 1
            rows.append((f'{x:.17g}', f'{y:.17g}', f'{DENSITIES[phase]:.17g}', phase))
    # Do not replace an existing output until the entire map has been validated.
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('x', 'y', 'rho', 'phase'))
        writer.writerows(rows)
    print(f'Saved {args.output}')
    print(f'Solver grid arguments: {args.nx} {args.ny} {lx:g} {ly:g}')
    print(f'Subtracted source origin: ({origin_x:g}, {origin_y:g})')
    print(f'Phase cell counts: {counts}; largest nearest-sample distance: {max_distance:g}')
    print('Nearest-neighbor mapping fills gaps; use --max-distance to reject unsupported regions.')


if __name__ == '__main__':
    main()
