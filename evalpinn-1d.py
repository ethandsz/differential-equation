
from scipy.sparse import coo
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer, ImplicitSourceTerm, residual
from fipy.terms.transientTerm import TransientTerm

from ipinn1d import PINN_1D, denormalize_c
from PollutionPDE1D import PollutionPDE1D
from matplotlib import cm

# ----------------------------------------
# 1) Re‑define your PINN and load weights
# ----------------------------------------

TIMESTEP = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN_1D().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ----------------------------------------
# 2) Build the Grid3D and get cell centers
# ----------------------------------------

NUM_CELLS = 100

# Convert to numpy arrays and reshape to (N, )
x_np = np.arange(0,100,0.5).flatten() 


# ----------------------------------------
# 3) Run inference at t = 73
# ----------------------------------------

# create torch tensors
x_t = torch.tensor(x_np, dtype=torch.float32, device=device)

steps = 100
times = np.arange(0,steps*1/60,1/60, dtype=np.float32)

x_pred = np.arange(0,NUM_CELLS,1)


start_idx = 0
end_idx = steps

dt = 1/60

t_min = start_idx * dt
t_max = end_idx * dt



all_values = np.load("1d-data.npy")
training_data = all_values[start_idx:end_idx,:]
print(training_data)
c_min = np.min(training_data)
c_max = np.max(training_data)


cb = None
plt.ion()

fig, ax = plt.subplots()
line, = ax.plot(np.zeros(NUM_CELLS), label="Pollutant concentration")
line_gt, = ax.plot(np.zeros(NUM_CELLS), label="Ground truth")
ax.set_ylim(0, 5)
ax.set_xlim(0, NUM_CELLS)
ax.set_xlabel('Cell index')
ax.set_ylabel('Concentration')
ax.set_title('Pollutant concentration over time')
ax.legend()


data = np.zeros((steps, NUM_CELLS))
ground_truth = np.load("1d-data.npy")
for idx, t in enumerate(times):
    coords = []
    for x in x_pred:

        x = float(x / (NUM_CELLS-1))

        t_normalized = float((t - t_min) / (t_max - t_min))
        coords.append([x,t_normalized])
    print(coords)

    coords = torch.tensor(coords, device=device, requires_grad=False)


    with torch.no_grad():
        c_t = model(coords).cpu()
    c_t = denormalize_c(c_t, c_min, c_max )
    print(c_t.shape)

    c_t = c_t.detach().numpy()

    data[idx, :] = c_t

 # --- Update plot ---
    line.set_ydata(c_t)

    gt_t = ground_truth[idx, :]
    line_gt.set_ydata(gt_t)

    ax.set_ylim(0, max(1.5, np.max(c_t)*1.1))
    plt.draw()
    plt.pause(0.001)
    input("Next timestep")

    # 8) Wait for user to press Enter before next step

# turn off interactive
plt.ioff()
plt.show()



fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

# Prepare grid: time steps and cell indices
time_axis = np.arange(steps)
cell_axis = np.arange(NUM_CELLS)
T, X = np.meshgrid(time_axis, cell_axis, indexing='ij')

# Plot surface
surf = ax3d.plot_surface(
    T, X, data,
    cmap=cm.viridis,
    linewidth=0,
    antialiased=False
)


surf_gt = ax3d.plot_surface(
    T, X, ground_truth[start_idx:end_idx,:],
    cmap=cm.plasma,
    alpha=0.6,
    label='Ground truth'
)

ax3d.set_xlabel('Time step')
ax3d.set_ylabel('Cell index')
ax3d.set_zlabel('Concentration')
ax3d.set_title('Pollutant concentration over time (3D view)')

fig3d.colorbar(surf, shrink=0.5, aspect=10)
fig3d.colorbar(surf_gt, shrink=0.5, aspect=10, label='Ground truth')

plt.show()

# --- Calculate the error (if not already done) ---
ground_truth_slice = ground_truth[start_idx:end_idx, :]
error = ground_truth - data 

# --- Create the 2D heatmap for the error ---
fig_hm, ax_hm = plt.subplots()

# Find the maximum absolute error to center the colormap
max_abs_error = np.max(np.abs(error))

# Use imshow to create the heatmap. Note the transpose on `error.T`.
# origin='lower' puts (0,0) at the bottom-left.
im = ax_hm.imshow(
    error.T, 
    aspect='auto', 
    origin='lower',
    cmap='coolwarm',
    vmin=-max_abs_error,
    vmax=max_abs_error
)

ax_hm.set_xlabel('Time step')
ax_hm.set_ylabel('Cell index')
ax_hm.set_title('Prediction Error Heatmap')
fig_hm.colorbar(im, label='Error (Prediction - GT)')

plt.show()


