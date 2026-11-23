import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

N = 10
kernel = np.ones((N,N))/ N**2

print(kernel.shape)

#read image
data = pd.read_csv('hs_snapshot.csv')

X = data['mesh_model_coordinates:0']
Y = data['mesh_model_coordinates:1']
Z = data['mesh_model_coordinates:2']

T = data['T']

#project data to 2D
T2D = data.pivot(
    index='mesh_model_coordinates:0',
    columns='mesh_model_coordinates:1',
    values='T',
).to_numpy()


# #define number of convolution passes

n = 2
fig, ax = plt.subplots(1, 2)
images = [T2D]

for i in range(n):
    print(f'applying convolution step {i}')
    conv_image = convolve2d(images[-1], kernel, mode='valid', boundary='symm')
    images.append(conv_image)

ax[0].imshow(images[0])
ax[0].set_aspect('equal')

pooled = images[-1][::N, ::N]
xs = np.sort(data['mesh_model_coordinates:0'].unique())
ys = np.sort(data['mesh_model_coordinates:1'].unique())
nx, ny = pooled.shape
x_pooled = xs[:nx * N].reshape(nx, N).mean(axis=1)
y_pooled = ys[:ny * N].reshape(ny, N).mean(axis=1)

ax[1].pcolormesh(x_pooled, y_pooled, pooled.T, shading='nearest', vmin=T2D.min(), vmax=T2D.max())
ax[1].set_aspect('equal')
plt.show()