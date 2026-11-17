from pathlib import Path
import subprocess as su

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input numbers are MOOSE timestep numbers, not physical times or FD steps.
dt_moose = 1e-4
dt_fd = 1e-4
elements = 10
fd_nodes = 10  # FD node count; separate from MOOSE element count.
centered_hot_width_nm = 2000
length = 10000
E1 = 24400
E2 = 23500
k = 0.361
# None centers the full FD domain by its mean x-coordinate, like MOOSE.
# Set to 0 only for a left-ignited FD half-domain whose boundary is the center.
fd_center_nm = None
plot_moose = True  #set True to download MOOSE profiles as solid lines
folder = Path(__file__).resolve().parent
name_FS = folder.parent / f'csv/kinetics1D_solN{fd_nodes}T1920T0300l{length}dt{dt_fd:g}k{k}E1{E1}E2{E2}_centered{centered_hot_width_nm:g}nm.csv'
remote_folder = 'gonz1075@negishi.rcac.purdue.edu:/scratch/negishi/gonz1075/mist2/sakano_proof/comparison'

def main():
    steps_moose = np.array(input('Provide MOOSE timesteps (comma separated): ').split(','), dtype=int)
    print(f'Loading FD file: {name_FS}')
    if not name_FS.exists():
        raise FileNotFoundError(f'Run the matching case first: {name_FS}')
    data_FS = pd.read_csv(name_FS)
    saved_steps = np.sort(data_FS['step'].unique())
    if len(saved_steps) == 0:
        raise ValueError(f'No FD profiles in {name_FS}')
    initial_fd = data_FS.loc[data_FS['step'] == saved_steps[0]].sort_values('X')
    if not (np.isclose(initial_fd['X'].min(), 0)
            and np.isclose(initial_fd['X'].max(), length)
            and len(initial_fd) == fd_nodes):
        raise ValueError('FD CSV coordinates/node count do not match the configured case')
    initial_hot = initial_fd.loc[initial_fd['T'] > 0.5 * (1920 + 300), 'X']
    if len(initial_hot):
        print(f'Initial FD hot nodes: {initial_hot.min():g} to {initial_hot.max():g} nm')
    fd_origin = initial_fd['X'].mean() if fd_center_nm is None else fd_center_nm
    print(f'FD coordinate origin: {fd_origin:g} nm')
    # kinetics1D.C writes the initial state as step 1, and the first update as 2.
    saved_times = (saved_steps - 1) * dt_fd

    plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2})
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    plotted_fd = set()
    for step_moose in steps_moose:
        target_time = step_moose * dt_moose
        if target_time < saved_times[0] - 1e-12 or target_time > saved_times[-1] + 1e-12:
            print(f'Skipping t={target_time:g} ns: FD output spans '
                  f'{saved_times[0]:g} to {saved_times[-1]:g} ns')
            continue

        nearest = int(np.argmin(np.abs(saved_times - target_time)))
        fd_step = saved_steps[nearest]
        actual_time = saved_times[nearest]
        current_FS = data_FS.loc[data_FS['step'] == fd_step].sort_values('X')
        print(f'MOOSE step {step_moose}, t={target_time:g} ns -> '
              f'FD step {fd_step:g}, saved t={actual_time:g} ns ({len(current_FS)} points)')
        color = None
        if fd_step not in plotted_fd:
            line, = ax.plot(current_FS['X'] - fd_origin, current_FS['T'],
                            label=f'FD t={actual_time:g} ns', linestyle='--')
            color = line.get_color()
            plotted_fd.add(fd_step)

        if plot_moose:
            # Four digits for ALL steps: 5 -> 0005, 50 -> 0050, 570 -> 0570.
            name = f'1D_{elements}elements_out_profiles_{step_moose:04d}.csv'
            local_file = folder / name
            if not local_file.exists():
                su.run(['scp', f'{remote_folder}/{name}', str(local_file)], check=True)
            data = pd.read_csv(local_file).sort_values('x')
            print(f'{name}: {len(data)} samples, x = '
                  f'{data["x"].min():g} to {data["x"].max():g} nm; '
                  f'T = {data["T"].min():g} to {data["T"].max():g} K')
            if np.allclose(data['T'], 300):
                print('MOOSE profile is uniformly 300 K: shifting cannot recover a hotspot. '
                      'Check the run initial condition and whether this cached file is from an older run.')
            if not (np.isclose(data['x'].min(), 0) and np.isclose(data['x'].max(), length)):
                print(f'Warning: expected sampling from 0 to {length:g} nm. '
                      'Check the input used for this run and whether the cached CSV is outdated.')
            ax.plot(data['x'] - data['x'].mean(), data['T'], color=color,
                    label=f'MOOSE t={target_time:g} ns')

    ax.set_xlabel('Distance from hotspot center (nm)')
    ax.set_ylabel('Temperature (K)')
    ax.grid(alpha=0.25)
    # Keep both halves visible; shifting does not change physical lengths.
    ax.margins(x=0.02)
    if ax.lines:
        ax.legend()
        plt.show()
    else:
        print('No requested times were within the saved FD range.')
        plt.close(fig)


if __name__ == '__main__':
    main()
