from fipy import CellVariable, Grid2D, PowerLawConvectionTerm, Viewer
from fipy.tools import numerix

# Mesh setup
nx, ny = 100, 100
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Initial scalar field (e.g. concentration or temperature)
phi0 = 1.0
x, y = mesh.cellCenters
alpha = 1.0
phi = CellVariable(name=r"$\phi$", mesh=mesh, value=phi0 * numerix.exp(-alpha * ((x - Lx/2)**2 + (y - Ly/2)**2)))

# Define constant 2D velocity field (right and upward)
velocity = numerix.array([1.0, 1.0])  # x and y components

# Set boundary conditions
phi.constrain(phi0, mesh.facesLeft)  # inflow at left
phi.faceGrad.constrain([0.0], mesh.facesRight)  # fake outflow on right
phi.faceGrad.constrain([0.0], mesh.facesTop)
phi.faceGrad.constrain([0.0], mesh.facesBottom)

# Define convection equation only (no diffusion here)
eq = PowerLawConvectionTerm(coeff=velocity)

# Solve
eq.solve(var=phi)

# Viewer
viewer = Viewer(vars=phi)
viewer.plot()
input("Press Enter to exit.")
