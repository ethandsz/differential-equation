from scipy.sparse import coo
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer, ImplicitSourceTerm, residual
from fipy.terms.transientTerm import TransientTerm

from ipinn import PINN, denormalize_c
from PollutionPDE import PollutionPDE

# ----------------------------------------
# 1) Re‑define your PINN and load weights
# ----------------------------------------

TIMESTEP = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
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

times = np.arange(0,300,1/60, dtype=np.float32)

x_pred = np.arange(0,25,1)
y_pred = np.arange(0,25,1)
z_pred = np.arange(0,25,1)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')



all_values = np.load("tall-data.npy")
training_data = all_values[4,:]
print(training_data)
c_min = np.min(training_data)
c_max = np.max(training_data)

cb = None
plt.ion()

for t in times:
    coords = []
    for z in z_pred:
        for y in y_pred:
            for x in x_pred:
                coords.append([x,y,z,t])

    coords = torch.tensor(coords, device=device, requires_grad=True)


    # with torch.no_grad():
    c_t = model(coords).cpu()
    c_t = denormalize_c(c_t, c_min, c_max )
    print(c_t.shape)

    grads = torch.autograd.grad(
        outputs=c_t, 
        inputs=coords,
        grad_outputs=torch.ones_like(c_t),  # usually ones
        create_graph=True  # so we can take further derivatives later if needed
    )[0]


    c_x = grads[:, 0]
    c_y = grads[:, 1]
    c_z = grads[:, 2]

    c_wrt_t = grads[:, 3]

    grads_x = torch.autograd.grad(
        c_x, coords,
        grad_outputs=torch.ones_like(c_x),
        create_graph=True
    )[0]
    c_xx = grads_x[:, 0] 

    grads_y = torch.autograd.grad(
        c_y, coords,
        grad_outputs=torch.ones_like(c_y),
        create_graph=True
    )[0]
    c_yy = grads_y[:, 1]

    grads_z = torch.autograd.grad(
        c_z, coords,
        grad_outputs=torch.ones_like(c_z),
        create_graph=True
    )[0]
    c_zz = grads_z[:, 2]



    sum_cx = torch.sum(c_x)
    sum_cy = torch.sum(c_y)
    sum_cz = torch.sum(c_z)


    sum_cxx = torch.sum(c_xx)
    sum_cyy = torch.sum(c_yy)
    sum_czz = torch.sum(c_zz)

    print(f"Sum of c_x {sum_cx} shape of c_x {c_x.shape}")
    print(f"Sum of c_y {sum_cy} shape of c_y {c_y.shape}")
    print(f"Sum of c_z {sum_cz} shape of c_z {c_z.shape}")
    print(f"Sum of c_t {torch.sum(c_wrt_t)} shape of c_z {c_wrt_t.shape}")

    print(f"Sum of c_xx {sum_cxx} shape of c_xx {c_xx.shape}")
    print(f"Sum of c_yy {sum_cyy} shape of c_yy {c_yy.shape}")
    print(f"Sum of c_zz {sum_czz} shape of c_zz {c_zz.shape}")

    print(f"Sum of laplacian (total) {sum_cxx + sum_cyy + sum_czz}")

    pde_problem = PollutionPDE(
            num_cells=NUM_CELLS,
            diffusion_coef=0.6,
            convection_coef=(1.0,0.0,0.0)
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = torch.tensor(pde_problem.get_source().value)

    print(f"Sum of source {torch.sum(source)} shape of source {source.shape}")
    
    idx = 0

    laplacian = c_xx[idx] + c_yy[idx] + c_zz[idx]


    print(f"Sum of laplacian {laplacian}")
    print(f"source {source[idx]}")
    print(f"C-wrt[{idx}] {c_wrt_t[idx]}")
    print(f"C-x[{idx}] {c_x[idx]}")
    print(f"C-y[{idx}] {c_y[idx]}")
    print(f"C-z[{idx}] {c_z[idx]}")
    

    residual = c_wrt_t[idx] - 0.6*(laplacian) + c_x[idx] 
    print(f"Residual = {c_wrt_t[idx]} - {0.6*laplacian} + {c_x[idx]}")
    print(f"Sum of residual {residual}")

    c_t = c_t.detach().numpy()





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
