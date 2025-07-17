
#!/usr/bin/env python3
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PollutionPDE1D import PollutionPDE1D
from matplotlib import cm
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Parameters ---
NUM_CELLS = 100
diffusionCoef = 0.7

parser = argparse.ArgumentParser(prog='1D Pollution PDE Simulation')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=True,
                    help='Enable matplotlib viewer')
parser.add_argument('--simSteps', type=int, default=300, help='Number of simulation steps')
parser.add_argument('--xFlow', type=float, default=1.0, help='Flow in X direction')
args = parser.parse_args()

convectionCoef = (args.xFlow,)  # single value tuple for 1D
dt = 1/60
steps = args.simSteps

# --- Create PDE problem ---
pde_problem = PollutionPDE1D(
    num_cells=NUM_CELLS,
    diffusion_coef=diffusionCoef,
    convection_coef=convectionCoef
)

var = pde_problem.get_variable()

# --- Setup viewer ---
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(var.value, label="Pollutant concentration")
ax.set_ylim(0, 5)
ax.set_xlim(0, NUM_CELLS)
ax.set_xlabel('Cell index')
ax.set_ylabel('Concentration')
ax.set_title('Pollutant concentration over time')
ax.legend()

input("Press Enter to start simulation...")

data = np.zeros((steps, NUM_CELLS))

# --- Simulation loop ---
for step in range(steps):
    start_time = time.time()

    var.updateOld()
    data[step,:] = var.value
    residual = pde_problem.sweep_eq(dt)
    matrix = pde_problem.get_eq_matrix()
    rhs_vector = pde_problem.get_eq_rhsVector()
    print(rhs_vector.shape)
    print(rhs_vector)
    print(matrix.shape)
    residual_vector = matrix @ var.value - rhs_vector
    print(np.sum(residual_vector))
    l2_residual = np.linalg.norm(residual_vector, ord=2)
    print(l2_residual)
    print(f"Step {step}: sum concentration = {np.sum(var.value):.4f}, residual = {residual:.4e}")

    # --- Update plot ---
    line.set_ydata(var.value)
    ax.set_ylim(0, max(1.5, np.max(var.value)*1.1))
    plt.draw()
    plt.pause(0.001)

    print(f"Time taken: {time.time() - start_time:.4f} seconds")

plt.ioff()
plt.show()

print("Simulation complete.")

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

ax3d.set_xlabel('Time step')
ax3d.set_ylabel('Cell index')
ax3d.set_zlabel('Concentration')
ax3d.set_title('Pollutant concentration over time (3D view)')

fig3d.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
np.save("1d-data.npy", data)
