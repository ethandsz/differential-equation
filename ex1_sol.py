
from fipy import *
import matplotlib.pyplot as plt
import numpy as np

# Setting up the mesh
nx = ny = 100
dx = dy = 0.01
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

phi = CellVariable(mesh=mesh, name=r"$\phi$", value=0.0)

# Dirichlet BCs
phi.constrain(1.0, mesh.facesTop)
phi.constrain(0.0, mesh.facesBottom | mesh.facesLeft | mesh.facesRight)

# Solve
eq = DiffusionTerm(coeff=1.0) == 0
eq.solve(var=phi)
# Reshape for plotting
phi_2D = np.reshape(np.array(phi), (nx, ny))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Numerical solution
im1 = axs[0].imshow(phi_2D.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=1)
axs[0].set_title("Numerical Solution")
fig.colorbar(im1, ax=axs[0])

plt.tight_layout()
plt.show()
