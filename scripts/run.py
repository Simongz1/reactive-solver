#run cases from here

from os import path
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rerun = input('Run again all cases? (y/n): ')

dt = [1e-6]
N = [2000, 1000, 500, 250]
l = [100]
T_init = [1920]
T_0 = [300]
numSteps = [200000]
stride = [100]

if (rerun == 'y'):
    for time in dt:
        for n in N:
            for T in T_init:
                for T0 in T_0:
                    name = f'solN{n}T{T}T0{T0}dt{time}.csv'
                    subprocess.run(['./solver', 
                                    str(n), 
                                    str(l[0]), 
                                    str(T), 
                                    str(T0), 
                                    str(time),
                                    name,
                                    str(numSteps[0]),
                                    str(stride[0])])


# Compare the right-moving Y3 threshold front for every case.
threshold = 0.99
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.get_cmap("tab10")
line_styles = ["-", "--", "-.", ":"]
plotted = 0

for dt_index, case_dt in enumerate(dt):
    for n_index, n in enumerate(N):
        for T in T_init:
            for T0 in T_0:
                name = f"solN{n}T{T}T0{T0}dt{case_dt}.csv"
                try:
                    data = pd.read_csv(name, usecols=["step", "X", "Y3"])
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    print(f"Skipping {name}: {error}")
                    continue

                positions = []
                steps = []
                invalid_frames = 0
                for step, frame in data.groupby("step", sort=True):
                    frame = frame.sort_values("X")
                    x = frame["X"].to_numpy()
                    y = frame["Y3"].to_numpy()
                    steps.append(step)
                    positions.append(np.nan)
                    if (len(x) < 2 or not np.isfinite(x).all()
                            or not np.isfinite(y).all()
                            or np.any(np.diff(x) <= 0)
                            or np.any(y < -1e-6) or np.any(y > 1 + 1e-6)):
                        invalid_frames += 1
                        continue

                    midpoint = 0.5 * (x[0] + x[-1])
                    crossings = np.flatnonzero(
                        (x[:-1] >= midpoint)
                        & (y[:-1] >= threshold)
                        & (y[1:] < threshold)
                    )
                    if crossings.size:
                        i = crossings[-1]
                        fraction = (threshold - y[i]) / (y[i + 1] - y[i])
                        positions[-1] = x[i] + fraction * (x[i + 1] - x[i])

                if invalid_frames:
                    print(f"{name}: excluded {invalid_frames} invalid Y3/grid frames")

                # Step 1 is the initial condition; use this case's timestep.
                time_ns = (np.asarray(steps, dtype=float) - 1) * case_dt
                velocity = np.diff(np.asarray(positions)) / np.diff(time_ns)
                time_mid = 0.5 * (time_ns[:-1] + time_ns[1:])
                if not np.isfinite(velocity).any():
                    print(f"Skipping {name}: no consecutive valid front positions")
                    continue

                # NaNs preserve gaps before ignition/after boundary arrival.
                ax.plot(time_mid, velocity,
                        color=colors(dt_index % 10),
                        linestyle=line_styles[n_index % len(line_styles)],
                        label=f"N={n}, dt={case_dt:g} ns, Tinit={T}, T0={T0}")
                plotted += 1

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Front velocity (m/s)")  # 1 nm/ns = 1 m/s
ax.set_title(f"Right-moving front at Y3 = {threshold}")
ax.grid(alpha=0.3)
if plotted:
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
else:
    print("No cases have a measurable front velocity.")

ax.set_xlim(0.075, 0.1)
ax.set_ylim(0, 200)
fig.tight_layout()
fig.savefig("deflagration_velocity_all_cases.png", dpi=300, bbox_inches="tight")
plt.show()
