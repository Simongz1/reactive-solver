from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "solution.csv")
steps = sorted(data["step"].unique())

frame_stride = 100
variable = "T"
dt = 1e-5

selected_steps = steps[::frame_stride]
if selected_steps[-1] != steps[-1]:
    selected_steps.append(steps[-1])

frames = [
    data.loc[data["step"] == step].sort_values("X")
    for step in selected_steps
]

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
title = ax.set_title("")

ax.set_xlabel("Position")
ax.set_ylabel(variable)
ax.set_xlim(data["X"].min(), data["X"].max())

tmin, tmax = data[variable].min(), data[variable].max()
padding = max(0.05 * (tmax - tmin), 1.0)
ax.set_ylim(tmin - padding, tmax + padding)
ax.grid(alpha=0.3)

def update(frame_index):
    frame = frames[frame_index]
    line.set_data(frame["X"], frame[variable])
    title.set_text(f"Step {selected_steps[frame_index]}")
    return line, title

animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=100,
    blit=True,
)

output = folder / f"{variable.lower()}.gif"
animation.save(output, writer=PillowWriter(fps=10))
plt.close(fig)

print(f"Saved {output}")


##calculate the deflagration velocity as a function of domain
##refinement and timestep size

threshold = 0.95
positions = []

for step in steps:
    frame = data.loc[data["step"] == step].sort_values("X")
    x = frame["X"].to_numpy()
    y = frame["Y3"].to_numpy()

    midpoint = 0.5 * (x[0] + x[-1])

    crossings = np.flatnonzero(
        (x[:-1] >= midpoint)
        & (y[:-1] >= threshold)
        & (y[1:] < threshold)
    )

    if crossings.size == 0:
        # Front not formed yet, or has already reached the boundary.
        positions.append(np.nan)
        continue

    i = crossings[-1]
    fraction = (threshold - y[i]) / (y[i + 1] - y[i])
    positions.append(x[i] + fraction * (x[i + 1] - x[i]))

positions = np.asarray(positions)

# Your CSV labels the initial condition as step 1.
time_ns = (np.asarray(steps, dtype=float) - 1) * dt

velocity = np.diff(positions) / np.diff(time_ns)
time_mid = 0.5 * (time_ns[:-1] + time_ns[1:])

fig, ax = plt.subplots()
ax.plot(time_mid, velocity)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Front velocity (m/s)")
ax.set_xlim(0.02, 0.03)
ax.set_ylim(0, 1500)
ax.grid(alpha=0.3)
plt.savefig(f'def_dt{dt}.png', dpi = 750)
plt.show()