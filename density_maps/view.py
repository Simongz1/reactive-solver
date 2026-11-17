import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('rho_field.csv')

fig, ax = plt.subplots()
ax.scatter(data['x'], data['y'], c = data['rho'], s = 1)
plt.show()