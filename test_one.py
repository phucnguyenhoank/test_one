
import numpy as np

X = np.array([1, 2, 3])  # X values
Y = np.array([4, 5, 6])  # Y values

# Create a grid
X_grid, Y_grid = np.meshgrid(X, Y)

print("X_grid:")
print(X_grid)

print("Y_grid:")
print(Y_grid)

Z_grid = X_grid**2 + Y_grid**2
print(Z_grid)

import matplotlib.pyplot as plt
plt.contourf(X_grid, Y_grid, Z_grid, levels=9, cmap='viridis')
# plt.contour(X_grid, Y_grid, Z_grid, levels=9, cmap='viridis')
plt.colorbar()
plt.show()

