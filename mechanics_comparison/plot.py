import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

times = [0.01]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots()

for t in times:
    data_fe = pd.read_csv(f'line{t}.csv')
    data_fv = pd.read_csv(f'line{t}_fv.csv')

    ax.plot(data_fe['arc_length'], data_fe['p_mix'], label=f'FE t={t:g} ns')
    ax.plot(data_fv['arc_length'], data_fv['p'], '--', label=f'FV t={t:g} ns')

ax.set_xlabel('distance (nm)')
ax.set_ylabel('pressure (GPa)')
ax.legend()
plt.tight_layout()
plt.show()