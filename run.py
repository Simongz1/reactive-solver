#run cases from here

from os import path
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rerun = input('Run again all cases? (y/n): ')

dt = [1e-6]
# Enable a 5,000 nm domain with a 20 nm hot section at the left end.
long_left_ignition = False
left_hot_width_nm = 50
N = [500]
l = [1000]
T_init = [1920]
T_0 = [300]
numSteps = [1000000]
stride = [500]

def case_filename(n, length, hot, cold, timestep):
    suffix = f"_left{left_hot_width_nm:g}nm" if long_left_ignition else ""
    return f"./csv/solN{n}T{hot}T0{cold}l{length}dt{timestep}{suffix}.csv"


# Profile comparisons: physical times in ns, one panel per requested time.
profile_variable = "T"  # Y1, Y2, Y3, or T
profile_times_ns = [0, 0.03, 0.05, 0.07]

if not profile_times_ns or not all(
    np.isfinite(t) and t >= 0 for t in profile_times_ns
):
    raise ValueError("profile_times_ns must contain finite, nonnegative times")

profile_columns = min(2, len(profile_times_ns))
profile_rows = (len(profile_times_ns) + profile_columns - 1) // profile_columns
profile_fig, profile_axes = plt.subplots(
    profile_rows, profile_columns,
    figsize=(7 * profile_columns, 4.5 * profile_rows), squeeze=False,
)
profile_axes = profile_axes.ravel()
profile_counts = [0] * len(profile_times_ns)
for index, target_time in enumerate(profile_times_ns):
    profile_axes[index].set_title(f"Time: {target_time:g} ns")
    profile_axes[index].set_xlabel("Distance (nm)")
    profile_axes[index].set_ylabel(
        "Temperature (K)" if profile_variable == "T" else f"{profile_variable} mass fraction"
    )
    profile_axes[index].grid(alpha=0.3)
for unused_ax in profile_axes[len(profile_times_ns):]:
    unused_ax.set_visible(False)

if (rerun == 'y'):
    for time in dt:
        for n in N:
            for length in l:
                for T in T_init:
                    for T0 in T_0:
                        name = case_filename(n, length, T, T0, time)
                        subprocess.run(['./mechanics1D', 
                                        str(n), 
                                        str(length),
                                        str(T), 
                                        str(T0), 
                                        str(time),
                                        name,
                                        str(numSteps[0]),
                                        str(stride[0]),
                                        str(1)] + ([str(left_hot_width_nm)] if long_left_ignition else []), check=True)

# Compare the right-moving Y3 threshold front for every case.
threshold = 0.99
# Set these to a steady propagation interval in ns; None uses available times.
fit_start_ns = None
fit_end_ns = None
if (fit_start_ns is not None and fit_end_ns is not None
        and fit_start_ns >= fit_end_ns):
    raise ValueError("fit_start_ns must be smaller than fit_end_ns")
fit_results = []
position_fig, position_ax = plt.subplots(figsize=(12, 7))
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.get_cmap("tab10")
line_styles = ["-", "--", "-.", ":"]
plotted = 0
plt.rcParams.update({'font.size': 12})

for dt_index, case_dt in enumerate(dt):
    for n_index, n in enumerate(N):
        for T in T_init:
            for length in l:
                for T0 in T_0:
                    name = case_filename(n, length, T, T0, case_dt)
                    try:
                        data = pd.read_csv(name, usecols=list(dict.fromkeys(["step", "X", "Y3", profile_variable])))
                    except (OSError, ValueError, pd.errors.ParserError) as error:
                        print(f"Skipping {name}: {error}")
                        continue

                    # Plot each case at the nearest saved time, labeling the actual time.
                    saved_steps = np.sort(data["step"].unique())
                    saved_times = (saved_steps.astype(float) - 1) * case_dt
                    case_index = (((dt_index * len(N) + n_index) * len(T_init)
                                + T_init.index(T)) * len(T_0) + T_0.index(T0))
                    case_color = colors(case_index % 10)
                    case_style = line_styles[(case_index // 10) % len(line_styles)]
                    for panel, target_time in enumerate(profile_times_ns):
                        tolerance = 1e-12 * max(1.0, abs(target_time))
                        if (len(saved_times) == 0
                                or target_time < saved_times[0] - tolerance
                                or target_time > saved_times[-1] + tolerance):
                            print(f"{name}: requested {target_time:g} ns is outside saved times")
                            continue
                        nearest = int(np.argmin(np.abs(saved_times - target_time)))
                        profile = data.loc[data["step"] == saved_steps[nearest]].sort_values("X")
                        values = profile[["X", profile_variable]].to_numpy()
                        if len(values) < 2 or not np.isfinite(values).all():
                            print(f"{name}: invalid profile at {saved_times[nearest]:g} ns")
                            continue
                        profile_axes[panel].plot(
                            profile["X"], profile[profile_variable],
                            color=case_color, linestyle=case_style,
                            label=(f"N={n}, dt={case_dt:g}, Tinit={T}, T0={T0}; "
                                f"t={saved_times[nearest]:g} ns"),
                        )
                        ##load the reference data
                        # current_reference = pd.read_csv(f'line_T_{target_time:g}.csv')
                        # profile_axes[panel].plot(
                        #     current_reference["arc_length"] - np.mean(current_reference["arc_length"]),
                        #     current_reference[profile_variable],
                        #     color='black', linestyle='--',
                        #     label=f"Reference t={target_time:g} ns"
                        # )
                        profile_counts[panel] += 1

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

                        front_start = x[0] if long_left_ignition else 0.5 * (x[0] + x[-1])
                        crossings = np.flatnonzero(
                            (x[:-1] >= front_start)
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
                    positions = np.asarray(positions, dtype=float)
                    label = f"N={n}, L={length}, dt={case_dt:g} ns, Tinit={T}, T0={T0}"
                    color = colors(len(fit_results) % 10)
                    position_ax.plot(time_ns, positions, color=color, alpha=0.45,
                                     label=label)
                    valid = np.isfinite(time_ns) & np.isfinite(positions)
                    if fit_start_ns is not None:
                        valid &= time_ns >= fit_start_ns
                    if fit_end_ns is not None:
                        valid &= time_ns <= fit_end_ns

                    # Do not fit across missing fronts: use the longest valid segment.
                    indices = np.flatnonzero(valid)
                    segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                    segment = max(segments, key=len)
                    if len(segment) < 3:
                        print(f"Skipping fit for {name}: fewer than 3 consecutive valid positions")
                        continue
                    t_fit = time_ns[segment]
                    x_fit = positions[segment]
                    if np.ptp(t_fit) <= 0:
                        print(f"Skipping fit for {name}: zero time span")
                        continue
                    # Center time for numerical conditioning. nm/ns equals m/s.
                    t_origin = t_fit.mean()
                    speed, offset = np.polyfit(t_fit - t_origin, x_fit, 1)
                    fitted = speed * (t_fit - t_origin) + offset
                    residual_sum = np.sum((x_fit - fitted)**2)
                    total_sum = np.sum((x_fit - x_fit.mean())**2)
                    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
                    position_ax.plot(t_fit, fitted, "--", color=color,
                                     label=f"Fit: {speed:.4g} m/s, R²={r_squared:.4f}")
                    ax.plot([t_fit[0], t_fit[-1]], [speed, speed], color=color,
                            label=f"{label}: {speed:.4g} m/s")
                    fit_results.append({
                        "file": name, "N": n, "length_nm": length, "dt_ns": case_dt,
                        "T_init": T, "T_0": T0, "threshold": threshold,
                        "fit_start_ns": t_fit[0], "fit_end_ns": t_fit[-1],
                        "samples": len(segment), "speed_m_per_s": speed,
                        "intercept_nm": offset - speed * t_origin,
                        "r_squared": r_squared,
                    })
                    print(f"{name}: fitted speed = {speed:.6g} m/s; "
                          f"interval [{t_fit[0]:g}, {t_fit[-1]:g}] ns; R²={r_squared:.6g}")
                    plotted += 1

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Fitted front velocity (m/s)")  # 1 nm/ns = 1 m/s
ax.set_title(f"Right-moving front at Y3 = {threshold}")
ax.grid(alpha=0.3)
if plotted:
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
else:
    print("No cases have a measurable front velocity.")

#ax.set_xlim(0.075, 0.1)
# Velocity limits are automatic so fitted values are not clipped.
fig.tight_layout()
fig.savefig("deflagration_velocity_all_cases.png", dpi=300, bbox_inches="tight")
position_ax.set_xlabel("Time (ns)")
position_ax.set_ylabel("Front position (nm)")
position_ax.set_title(f"Front positions and linear fits at Y3 = {threshold}")
position_ax.grid(alpha=0.3)
if position_ax.lines:
    position_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
position_fig.tight_layout()
position_fig.savefig("front_position_fits_all_cases.png", dpi=300, bbox_inches="tight")
pd.DataFrame(fit_results, columns=[
    "file", "N", "length_nm", "dt_ns", "T_init", "T_0", "threshold",
    "fit_start_ns", "fit_end_ns", "samples", "speed_m_per_s", "intercept_nm", "r_squared"
]).to_csv("deflagration_velocity_fits.csv", index=False)

for panel, count in enumerate(profile_counts):
    if count:
        profile_axes[panel].legend(fontsize=7)
        #profile_axes[panel].set_xlim(0, 20)
    else:
        profile_axes[panel].text(
            0.5, 0.5, "No valid profiles at this time",
            ha="center", va="center", transform=profile_axes[panel].transAxes,
        )
        
profile_fig.suptitle(f"{profile_variable} profiles across all cases")
profile_fig.tight_layout()
profile_output = f"profiles_{profile_variable}_all_cases.png"
profile_fig.savefig(profile_output, dpi=300, bbox_inches="tight")
print(f"Saved {profile_output}")
plt.show()
