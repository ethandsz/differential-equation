
import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from ipinn3d import PINN_3D, denormalize_c  # assume similar to your PINN_2D

# ----------------------
# 1) Load model
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN_3D().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ----------------------
# 2) Setup grid and time
# ----------------------
NUM_X = 100
NUM_Y = 100
NUM_Z = 100
steps = 25
dt = 1/60

times = np.arange(0, steps*dt, dt, dtype=np.float32)
t_min = times[0]
t_max = times[-1]

# ----------------------
# 3) Load GT data and normalize ranges
# ----------------------
all_values = np.load("3d-data.npy")  # shape: (steps, NUM_X*NUM_Y*NUM_Z)
training_data = all_values[:steps, :]
c_min = np.min(training_data)
c_max = np.max(training_data)

# ----------------------
# 4) Prepare coordinate grid (x, y, z)
# ----------------------
x_idx = np.arange(NUM_X)
y_idx = np.arange(NUM_Y)
z_idx = np.arange(NUM_Z)

x_norm = x_idx / (NUM_X - 1)
y_norm = y_idx / (NUM_Y - 1)
z_norm = z_idx / (NUM_Z - 1)

X_norm, Y_norm, Z_norm = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')
coords_xyz = np.stack([X_norm.flatten(), Y_norm.flatten(), Z_norm.flatten()], axis=1)  # (N, 3)

# ----------------------
# 5) Predict over time
# ----------------------
pred_data = np.zeros((steps, NUM_X, NUM_Y, NUM_Z))
gt_data = []

# Setup PyVista volume viewer
grid = pv.ImageData()
grid.dimensions = np.array([NUM_X, NUM_Y, NUM_Z]) + 1  # +1 for points
grid.spacing = (1,1,1)

plotter = pv.Plotter()

plotter.show_bounds(grid='back', location='outer', all_edges=True, font_size=16, color='black')
actor = None
plotter.add_axes()


for idx, t in enumerate(times):
    print(f"Timestep {idx+1}/{steps}")
    t_norm = (t - t_min) / (t_max - t_min)
    t_column = np.full((coords_xyz.shape[0], 1), t_norm, dtype=np.float32)
    
    coords = np.hstack([coords_xyz, t_column])  # (N,4)

    coords_tensor = torch.tensor(coords, device=device, dtype=torch.float32)
    with torch.no_grad():
        c_t = model(coords_tensor).cpu().numpy().flatten()
    c_t = denormalize_c(c_t, c_min, c_max)

    c_t_3d = c_t.reshape(NUM_X, NUM_Y, NUM_Z, order='F')  # shape matches UniformGrid
    pred_data[idx, :, :, :] = c_t_3d
    c_t_3d[c_t_3d <= 1e-6] = 0

    # GT
    gt_t_flat = all_values[idx]
    gt_t_3d = gt_t_flat.reshape(NUM_X, NUM_Y, NUM_Z)
    gt_data.append(gt_t_3d)

    # Update visualization
    grid.cell_data["concentration"] = c_t_3d.flatten()
    # grid.cell_data["concentration"] = gt_t_3d.flatten(order='F')

    if actor is None:
        actor = plotter.add_volume(grid, scalars="concentration", cmap='viridis')
    else:
        plotter.remove_actor(actor)
        actor = plotter.add_volume(grid, scalars="concentration", cmap='viridis')

    plotter.add_text(f"Time step {idx}", font_size=12, name="timestep")
    plotter.show(auto_close=False, interactive_update=True)
    plotter.render()

    input("Next timestep")

    
gt_data = np.array(gt_data)

for t in range(steps):
    error = gt_data[t] - pred_data[t]
    mean_error_z = np.mean(error, axis=2)
    
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(mean_error_z, origin='lower', cmap='inferno', 
                   aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean error over z', fontsize=12)
    ax.set_title(f'Mean error projection on x-y plane (t={t})', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    
    # Histogram of absolute errors
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(np.abs(error).ravel(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error distribution at t={t}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Boxplot of absolute errors
    fig, ax = plt.subplots(figsize=(6,6))
    ax.boxplot(np.abs(error).ravel(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='black'),
               medianprops=dict(color='black'))
    ax.set_ylabel('Absolute error', fontsize=12)
    ax.set_title(f'Boxplot of absolute errors at t={t}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
