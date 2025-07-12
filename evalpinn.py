import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer, ImplicitSourceTerm, residual
from fipy.terms.transientTerm import TransientTerm

from ipinn import PINN
from PollutionPDE import PollutionPDE

# ----------------------------------------
# 1) Re‑define your PINN and load weights
# ----------------------------------------

TIMESTEP = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ----------------------------------------
# 2) Build the Grid3D and get cell centers
# ----------------------------------------

NUM_CELLS = 25
L = 1.0
dx = L / NUM_CELLS

# Ground Truth Variables
diffusionCoef = 0.1
convectionCoef = (1.0, 0.0, 0.0)
# ------------------------------

pde_problem = PollutionPDE(
        num_cells=NUM_CELLS,
        diffusion_coef=diffusionCoef,
        convection_coef=convectionCoef
    )

# 2. Access the components
var = pde_problem.get_variable()
eq = pde_problem.get_equation()
mesh = pde_problem.get_mesh()

x_centers, y_centers, z_centers = mesh.cellCenters  # each is a 1D array of length NUM_CELLS**3


# Convert to numpy arrays and reshape to (N, )
x_np = np.array(x_centers).flatten() * NUM_CELLS  # scale to [0,25]
y_np = np.array(y_centers).flatten() * NUM_CELLS
z_np = np.array(z_centers).flatten() * NUM_CELLS

# ----------------------------------------
# 3) Run inference at t = 73
# ----------------------------------------

# create torch tensors
x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

times = np.arange(0,300,1)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')

cb = None
plt.ion()
for t in times:
    t_t = torch.full_like(x_t, t)

    with torch.no_grad():
        c_t = model(x_t, y_t, z_t, t_t).cpu().numpy().flatten()


    predicted_var = CellVariable(name="pred_poll", mesh=mesh, value=c_t)
    # eq.solve(var=var, dt=1)
    residual = eq.justResidualVector(var=predicted_var, dt=t)
    residual = np.array(residual)
    print(f"Sum of residual error: {np.sum(residual)}")
    residual_norm = np.linalg.norm(residual)
    print(f"Sum of residual error (normalized): {residual_norm}")

# ----------------------------------------
# 4) Plot in 3D (Matplotlib scatter), filtering below threshold
# ----------------------------------------

    POLLUTANT_DETECTION_THRESHOLD = 5e-3

# filter
    mask = c_t >= POLLUTANT_DETECTION_THRESHOLD
    x_plot = x_np[mask]
    y_plot = y_np[mask]
    z_plot = z_np[mask]
    c_plot = c_t[mask]

# 4) Clear old axes and colorbar
    ax.clear()
    if cb is not None:
        cb.remove()

    # 5) Plot updated scatter
    sc = ax.scatter(x_plot, y_plot, z_plot,
                    c=c_plot,
                    cmap='plasma',
                    s=20)
    ax.set_xlim(0, NUM_CELLS)
    ax.set_ylim(0, NUM_CELLS)
    ax.set_zlim(0, NUM_CELLS)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Predicted plume at t={t:.2f}")

    # 6) Update colorbar
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label('Concentration')

    # 7) Draw and pause
    plt.draw()
    plt.pause(0.001)

    # 8) Wait for user to press Enter before next step
    input("Next timestep:")

# turn off interactive
plt.ioff()
