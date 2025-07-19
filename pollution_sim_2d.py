

#!/usr/bin/env python3
import argparse
from pickle import NONE
import time
import numpy as np
import matplotlib.pyplot as plt
from PollutionPDE2D import PollutionPDE2D
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

parser = argparse.ArgumentParser(prog='2D Pollution PDE Simulation')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=True,
                    help='Enable matplotlib viewer')
parser.add_argument('--simSteps', type=int, default=100, help='Number of simulation steps')
parser.add_argument('--xFlow', type=float, default=10.0, help='Flow in X direction')
parser.add_argument('--yFlow', type=float, default=0.0, help='Flow in Y direction')
args = parser.parse_args()

convectionCoef = (args.xFlow, args.yFlow)  # single value tuple for 2D
dt = 1/60
steps = args.simSteps

# --- Create PDE problem ---
pde_problem = PollutionPDE2D(
    num_cells=NUM_CELLS,
    diffusion_coef=diffusionCoef,
    convection_coef=convectionCoef
)

var = pde_problem.get_variable()

# --- Setup viewer ---
plt.ion()
fig, ax = plt.subplots()
ax.set_ylim(0, NUM_CELLS)
ax.set_xlim(0, NUM_CELLS)
ax.set_xlabel('Cell index')
ax.set_ylabel('Concentration')
ax.set_title('Pollutant concentration over time')
ax.legend()
cb = None

input("Press Enter to start simulation...")

data = np.zeros((steps, NUM_CELLS*NUM_CELLS))


x = np.arange(NUM_CELLS)
y = np.arange(NUM_CELLS)
X, Y = np.meshgrid(x, y)

# --- Simulation loop ---
for step in range(steps):
    start_time = time.time()
    var.updateOld()
    data[step,:] = var.value
    residual = pde_problem.sweep_eq(dt)

    matrix = pde_problem.get_eq_matrix()
    rhs_vector = pde_problem.get_eq_rhsVector()

    residual_vector = matrix @ var.value - rhs_vector
    l2_residual = np.linalg.norm(residual_vector, ord=2)
    print(f"Step {step}: sum concentration = {np.sum(var.value):.4f}, residual = {residual:.4e}")

    concentration_values = []
    for idx, value in enumerate(var.value):
                y = (idx % (NUM_CELLS ** 2)) // NUM_CELLS
                x = idx % NUM_CELLS
                concentration_values.append([x,y,value])


    Z = var.value.reshape(NUM_CELLS, NUM_CELLS)

    concentration_values = np.array(concentration_values)
    # --- Update plot ---
    if(cb != None):
        cb.remove()
    ax.clear()
    cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    cb = plt.colorbar(cf, label='Concentration')
    plt.draw()
    plt.pause(0.001)

    print(f"Time taken: {time.time() - start_time:.4f} seconds")

plt.ioff()
plt.show()

np.save("2d-data.npy", data)

