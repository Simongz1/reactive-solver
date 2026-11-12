import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

meshes = [200, 100, 50]
dt = 1e-4

fig, ax = plt.subplots()
for mesh in meshes:
    name = f'area_fvmechanics.csv'
    data = pd.read_csv(name)
    a = np.array(data['area'])
    t = np.array(data['time'])
    r = np.sqrt(a / np.pi)
    ax.plot(t, r, label=f'mesh {mesh}')

ax.set_xlabel('time (s)')
ax.set_ylabel('radius (nm)')
ax.legend()
plt.tight_layout()
plt.show()

#plot derivative
fig, ax = plt.subplots()
for mesh in meshes:
    name = f'a{mesh}.csv'
    data = pd.read_csv(name)
    a = np.array(data['area'])
    r = np.sqrt(a / np.pi)
    diff = np.zeros_like(a)
    for i in range(1, len(diff)-1):
        diff[i] = (r[i+1]-r[i-1]) / (2*dt)
    ax.plot(t[1:-1], diff[1:-1], label=f'mesh {mesh}')

ax.set_xlabel('time (s)')
ax.set_ylabel('v (nm/ns)')
ax.legend()
plt.tight_layout()
plt.show()