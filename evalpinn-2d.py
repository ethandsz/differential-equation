
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ipinn2d import PINN_2D, denormalize_c

# ----------------------
# 1) Load model
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN_2D().to(device)
model.load_state_dict(torch.load("best_model_2d.pth", map_location=device))
model.eval()

# ----------------------
# 2) Setup grid and time
# ----------------------
NUM_X = 100     # after slicing: 0:50
NUM_Y = 100    # after slicing: 0:100
steps = 100
dt = 1/60

times = np.arange(0, steps*dt, dt, dtype=np.float32)

start_idx = 0
end_idx = steps

t_min = start_idx * dt
t_max = end_idx * dt

# ----------------------
# 3) Load GT data and normalize ranges
# ----------------------
all_values = np.load("2d-data.npy")
training_data = all_values[start_idx:end_idx, :]
c_min = np.min(training_data)
c_max = np.max(training_data)

# ----------------------
# 4) Prepare coordinate grid (x, y)
# ----------------------
# Physical indices
x_idx = np.arange(NUM_X)
y_idx = np.arange(NUM_Y)

# Normalize to [0,1]
x_norm = x_idx / (NUM_X - 1)
y_norm = y_idx / (NUM_Y - 1)

# Create meshgrid (X,Y) for whole domain
X_norm, Y_norm = np.meshgrid(x_norm, y_norm, indexing='ij')   # shape: (NUM_X, NUM_Y)

# Flatten to (N,2)
coords_xy = np.stack([X_norm.flatten(), Y_norm.flatten()], axis=1)  # (NUM_X*NUM_Y, 2)

# ----------------------
# 5) Loop over time steps, predict
# ----------------------
pred_data = np.zeros((steps, NUM_Y, NUM_X))  # (steps, ny, nx)
gt_data = []

plt.ion()
plt.rcParams["figure.figsize"] = (20,20)
for idx, t in enumerate(times):
    t_norm = (t - t_min) / (t_max - t_min)
    t_column = np.full((coords_xy.shape[0], 1), t_norm, dtype=np.float32)
    
    coords = np.hstack([coords_xy, t_column])  # shape: (N, 3)

    coords_tensor = torch.tensor(coords, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        c_t = model(coords_tensor).cpu().numpy().flatten()
    
    c_t = denormalize_c(c_t, c_min, c_max)

    # Reshape back to (ny, nx)
    c_t_2d = c_t.reshape(NUM_X, NUM_Y).T  # transpose to (ny, nx)
    pred_data[idx,:,:] = c_t_2d

    # Get GT for timestep, reshape
    gt_t_flat = all_values[idx].reshape(100,100)  # original domain
    gt_t_2d = gt_t_flat[0:100, 0:100]              # slice to match prediction
    gt_data.append(gt_t_2d)

    # --- Plot ---
    plt.clf()
    plt.suptitle(f"timestep {idx} (t={t:.2f})")

    plt.subplot(1,2,1)
    plt.title("Prediction")
    plt.imshow(c_t_2d, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Ground truth")
    plt.imshow(gt_t_2d, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.pause(0.01)
    input("Press Enter to continue")

plt.ioff()
plt.show()


# Compute absolute error once
error = pred_data - gt_data
error_abs_max = np.max(np.abs(error))

# ------------------------------------
# A) Existing: mean over y, visualize over (time, x)
# ------------------------------------
fig_hm_y, ax_hm_y = plt.subplots()
im_y = ax_hm_y.imshow(np.mean(error, axis=1).T,   # mean over y -> shape (steps, nx).T -> (nx, steps)
                      aspect='auto', origin='lower',
                      cmap='coolwarm', vmin=-error_abs_max, vmax=error_abs_max)
ax_hm_y.set_xlabel('Time step')
ax_hm_y.set_ylabel('x index')
ax_hm_y.set_title('Mean error over y (heatmap)')
fig_hm_y.colorbar(im_y, label='Error')
plt.show()

# ------------------------------------
# B) New: mean over x, visualize over (time, y)
# ------------------------------------
# mean over axis=2 (x): pred_data shape (steps, ny, nx) -> mean over nx
error_mean_over_x = np.mean(error, axis=2)   # shape: (steps, ny)

fig_hm_x, ax_hm_x = plt.subplots()
im_x = ax_hm_x.imshow(error_mean_over_x.T,    # transpose to (ny, steps)
                      aspect='auto', origin='lower',
                      cmap='coolwarm', vmin=-error_abs_max, vmax=error_abs_max)
ax_hm_x.set_xlabel('Time step')
ax_hm_x.set_ylabel('y index')
ax_hm_x.set_title('Mean error over x (heatmap)')
fig_hm_x.colorbar(im_x, label='Error')
plt.show()

# ------------------------------------
# C) Optional: 3D surface plots to match earlier visualization
# ------------------------------------
fig3d_y = plt.figure()
ax3d_y = fig3d_y.add_subplot(111, projection='3d')

T, X = np.meshgrid(np.arange(steps), np.arange(NUM_X), indexing='ij')
pred_mean_y = np.mean(pred_data, axis=1)  # mean over y
gt_mean_y = np.mean(gt_data, axis=1)

ax3d_y.plot_surface(T, X, pred_mean_y, cmap=cm.spring, alpha=0.8)
ax3d_y.plot_surface(T, X, gt_mean_y, cmap=cm.winter, alpha=0.5)
ax3d_y.set_xlabel('Time step')
ax3d_y.set_ylabel('x index')
ax3d_y.set_zlabel('Mean concentration')
ax3d_y.set_title('Pred vs GT over time (mean along y)')
plt.show()

# Mean over x
fig3d_x = plt.figure()
ax3d_x = fig3d_x.add_subplot(111, projection='3d')

T, Y = np.meshgrid(np.arange(steps), np.arange(NUM_Y), indexing='ij')
pred_mean_x = np.mean(pred_data, axis=2)  # mean over x
gt_mean_x = np.mean(gt_data, axis=2)

ax3d_x.plot_surface(T, Y, pred_mean_x, cmap=cm.hot, alpha=0.8)
ax3d_x.plot_surface(T, Y, gt_mean_x, cmap=cm.cool, alpha=0.5)
ax3d_x.set_xlabel('Time step')
ax3d_x.set_ylabel('y index')
ax3d_x.set_zlabel('Mean concentration')
ax3d_x.set_title('Pred vs GT over time (mean along x)')
plt.show()
