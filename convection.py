
import numpy as np
import pyvista as pv
from fipy import CellVariable, Grid2D, PowerLawConvectionTerm, TransientTerm
from fipy.tools import numerix

# --- 1) FiPy Simulation Setup (2D) ---
nx, ny = 100, 100
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

x, y = mesh.cellCenters
phi0 = 2.0
alpha = 1.0
phi = CellVariable(
    name="phi",
    mesh=mesh,
    value=phi0 * numerix.exp(-alpha * ((x - Lx/4)**2 + (y - Ly/2)**2))
)

velocity = numerix.array([0.1, 0.0])   # flow along +x
phi.constrain(0.0, mesh.exteriorFaces)  # absorbing walls

eq = TransientTerm() + PowerLawConvectionTerm(coeff=velocity)

dt = 0.1
steps = 500
save_interval = 20

snapshots = []
for step in range(steps):
    eq.solve(var=phi, dt=dt)
    if step % save_interval == 0:
        # reshape into (nx, ny)
        vals = phi.value.reshape((nx, ny), order="F").copy()
        snapshots.append(vals)
        print(f"Captured step {step}")

# --- 2) Build an ImageData “slab” for 2D cell data ---
grid = pv.ImageData(dimensions=(nx, ny, 2), spacing=(dx, dy, 1.0), origin=(0, 0, 0))

# --- 3) Animate with PyVista ---
plotter = pv.Plotter()
plotter.open_movie("phi_advection.mp4", framerate=15)

for i, vals in enumerate(snapshots):
    # ImageData.cell_data must have one entry per cell: nx*ny*1 cells
    grid.cell_data["phi"] = vals.flatten(order="F")
    plotter.clear()
    plotter.add_mesh(grid, scalars="phi", cmap="viridis", show_edges=False)
    plotter.add_text(f"Step {i*save_interval}", font_size=12, position="upper_left")
    plotter.write_frame()

plotter.close()
print("Saved animation to phi_advection.mp4")
