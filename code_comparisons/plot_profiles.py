#!/usr/bin/env python3
"""Plot MOOSE line-profile CSVs at selected physical times.

Example (constant dt, time in ns):
  python3 code_comparisons/plot_profiles.py /path/to/csvs \
      --variable T --times 0.1 0.5 1.0 --dt 1e-3 --save temperature.png

For adaptive time stepping replace --dt with:
      --time-file /path/to/1D_out_profiles_time.csv

Filename suffixes are simulation timestep numbers, NOT saved-frame indices.
"""
import argparse
import csv
import math
import re
from pathlib import Path


def load_times(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        names = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        step_name = next((names[k] for k in ('timestep', 'time_step', 'step') if k in names), None)
        if step_name is None or 'time' not in names:
            raise ValueError(f'{path}: expected timestep and time columns')
        result = {}
        for row in reader:
            number, time = float(row[step_name]), float(row[names['time']])
            if not math.isfinite(number) or not number.is_integer() or not math.isfinite(time):
                raise ValueError(f'{path}: invalid timestep/time row')
            step = int(number)
            if step in result and result[step] != time:
                raise ValueError(f'{path}: conflicting times for timestep {step}')
            result[step] = time
        return result


def find_profiles(directory, pattern, dt=None, time_file=None, start_time=0):
    times = load_times(time_file) if time_file else None
    result = []
    seen = set()
    for path in directory.glob(pattern):
        match = re.search(r'_(\d+)\.csv$', path.name)
        if not path.is_file() or not match:
            continue
        step = int(match.group(1))
        if step in seen:
            raise ValueError(f'Multiple profiles for timestep {step}; narrow --pattern')
        seen.add(step)
        if times is not None:
            if step not in times:
                raise ValueError(f'{time_file}: no time for profile timestep {step}')
            time = times[step]
        else:
            time = start_time + step * dt
        result.append((time, step, path))
    if not result:
        raise ValueError(f'No numbered profile CSVs matching {pattern!r} in {directory}')
    return sorted(result)


def select_profile(profiles, target, tolerance=None):
    if target < profiles[0][0]-1e-12 or target > profiles[-1][0]+1e-12:
        raise ValueError(f'Requested time {target:g} outside saved range '
                         f'[{profiles[0][0]:g}, {profiles[-1][0]:g}]')
    chosen = min(profiles, key=lambda item: (abs(item[0]-target), item[0]))
    if tolerance is not None and abs(chosen[0]-target) > tolerance+1e-12:
        raise ValueError(f'Nearest saved time to {target:g} is {chosen[0]:g}; exceeds --tolerance')
    return chosen


def read_profile(path, variable, coordinate='x'):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = {name.strip(): name for name in (reader.fieldnames or [])}
        for name in (coordinate, variable):
            if name not in columns:
                raise ValueError(f'{path}: missing {name!r}; available columns: {", ".join(columns)}')
        points = [(float(row[columns[coordinate]]), float(row[columns[variable]])) for row in reader]
    if not points or not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError(f'{path}: empty profile or nonfinite values')
    points.sort()
    return tuple(zip(*points))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('directory', nargs='?', type=Path, default=Path('.'))
    p.add_argument('--pattern', default='1D_out_profiles_*.csv')
    p.add_argument('--variable', '-v', default='T')
    p.add_argument('--times', '-t', nargs='+', type=float, required=True)
    timing = p.add_mutually_exclusive_group(required=True)
    timing.add_argument('--dt', type=float, help='Constant simulation timestep, NOT output interval')
    timing.add_argument('--time-file', type=Path, help='CSV containing timestep,time (supports adaptive dt)')
    p.add_argument('--start-time', type=float, default=0, help='Time at step 0; used only with --dt')
    p.add_argument('--tolerance', type=float, help='Maximum permitted nearest-time mismatch')
    p.add_argument('--coordinate', default='x', help='Distance column (x or id, for example)')
    p.add_argument('--time-unit', default='ns')
    p.add_argument('--length-unit', default='nm')
    p.add_argument('--font-size', type=float, default=18)
    p.add_argument('--save', type=Path, help='Save figure, e.g. profiles.png or profiles.pdf')
    p.add_argument('--no-show', action='store_true', help='Save without opening a window')
    a = p.parse_args()
    if (a.dt is not None and (not math.isfinite(a.dt) or a.dt <= 0)
        or not all(math.isfinite(t) for t in a.times)
        or not math.isfinite(a.start_time)
        or not math.isfinite(a.font_size) or a.font_size <= 0
        or a.tolerance is not None and (not math.isfinite(a.tolerance) or a.tolerance < 0)):
        p.error('dt/font-size must be positive; times finite; tolerance nonnegative')
    if a.no_show and not a.save:
        p.error('--no-show requires --save')
    try:
        profiles = find_profiles(a.directory, a.pattern, a.dt, a.time_file, a.start_time)
        selected, seen = [], set()
        for requested in a.times:
            time, step, path = select_profile(profiles, requested, a.tolerance)
            if step in seen:
                print(f'Requested {requested:g}: timestep {step} already selected; skipping duplicate')
                continue
            seen.add(step)
            x, y = read_profile(path, a.variable, a.coordinate)
            print(f'Requested {requested:g} {a.time_unit} -> saved {time:g} {a.time_unit}, '
                  f'step {step}: {path.name}')
            selected.append((time, x, y))
        import matplotlib
        if a.no_show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': a.font_size, 'axes.labelsize': a.font_size,
                             'legend.fontsize': a.font_size-2, 'lines.linewidth': 2.5})
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        for time, x, y in selected:
            ax.plot(x, y, label=f't = {time:g} {a.time_unit}')
        ax.set_xlabel(f'Distance ({a.length_unit})')
        ax.set_ylabel('Temperature (K)' if a.variable == 'T' else a.variable)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        if a.save:
            fig.savefig(a.save, dpi=300, bbox_inches='tight')
            print(f'Saved {a.save}')
        if not a.no_show:
            plt.show()
        plt.close(fig)
    except (OSError, ValueError, ImportError) as error:
        p.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
