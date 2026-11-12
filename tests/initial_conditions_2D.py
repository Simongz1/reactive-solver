"""Run after building mechanics2D, mechanics2DFV, kinetics2D, and kinetics2DFV."""
import csv
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'solution.csv'
    for solver in ('mechanics2D', 'mechanics2DFV', 'mechanics2DFV_limiter', 'kinetics2D', 'kinetics2DFV'):
        base = [str(ROOT / solver), '21', '13', '100', '60', '1000', '300',
                '1e-6', str(output), '2', '1']
        if solver.startswith('mechanics'):
            base += ['0']
        for options in ([], ['20'], ['--circular'], ['--circular', '15']):
            subprocess.run(base + options, check=True, capture_output=True)
            with output.open() as file:
                reader = csv.DictReader(file)
                columns = reader.fieldnames
                rows = list(reader)
            assert len(rows) == 4 * 21 * 13
            assert {r['step'] for r in rows} == {'1', '2', '3', '4'}
            if solver.startswith('kinetics'):
                assert columns == ['step', 'X', 'Y', 'Y1', 'Y2', 'Y3', 'T']
            for n, row in enumerate(rows[:21 * 13]):
                i, j = n % 21, n // 21
                x, y = float(row['X']), float(row['Y'])
                if not options:
                    hot = 8 <= i < 12 and 5 <= j < 7
                elif options[0] == '--circular':
                    radius = float(options[1]) if len(options) == 2 else 10
                    hot = (x - 50)**2 + (y - 30)**2 <= radius**2
                else:
                    hot = x < 20
                assert math.isclose(float(row['T']), 1000 if hot else 300, abs_tol=1e-9)
                assert float(row['Y1']) == 1 and float(row['Y2']) == float(row['Y3']) == 0
            for row in rows:
                assert all(math.isfinite(float(value)) for value in row.values())
                # mechanics2D retains its original six-digit CSV precision.
                assert math.isclose(sum(float(row[k]) for k in ('Y1', 'Y2', 'Y3')), 1, abs_tol=2e-6)
        for options in (['0'], ['100'], ['nan'], ['20junk'], ['--circular', '0'],
                        ['--circular', '-1'], ['--circular', 'nan'], ['--circular', '15junk']):
            assert subprocess.run(base + options, capture_output=True).returncode != 0
        exodus = Path(directory) / f'{solver}.e'
        base[8] = str(exodus)
        subprocess.run(base + ['--circular', '15'], check=True, capture_output=True)
        assert exodus.stat().st_size > 0
print('Passed: all listed solvers, all hotspot modes, finite short runs, species sums, CSV schemas, invalid arguments, Exodus creation.')
