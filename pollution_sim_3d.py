#!/usr/bin/env python3
import argparse
from pickle import NONE
import time
import numpy as np
import matplotlib.pyplot as plt
from PollutionPDE3D import PollutionPDE3D
from matplotlib import cm
import pyvista as pv

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

parser = argparse.ArgumentParser(prog='3D Pollution PDE Simulation')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=True,
                    help='Enable matplotlib viewer')
parser.add_argument('--simSteps', type=int, default=25, help='Number of simulation steps')
parser.add_argument('--xFlow', type=float, default=10.0, help='Flow in X direction')
parser.add_argument('--yFlow', type=float, default=0.0, help='Flow in Y direction')
parser.add_argument('--zFlow', type=float, default=0.0, help='Flow in Y direction')
args = parser.parse_args()

convectionCoef = (args.xFlow, args.yFlow, args.zFlow)  # single value tuple for 3D
dt = 1/60
steps = args.simSteps

# --- Create PDE problem ---
pde_problem = PollutionPDE3D(
    num_cells=NUM_CELLS,
    diffusion_coef=diffusionCoef,
    convection_coef=convectionCoef
)

var = pde_problem.get_variable()

# # --- Setup viewer ---
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_ylim(0, NUM_CELLS)
# ax.set_xlim(0, NUM_CELLS)
# ax.set_xlabel('Cell index')
# ax.set_ylabel('Concentration')
# ax.set_title('Pollutant concentration over time')
# ax.legend()
# cb = None
#

data = np.zeros((steps, NUM_CELLS**3))

#
# x = np.arange(NUM_CELLS)
# y = np.arange(NUM_CELLS)
# X, Y = np.meshgrid(x, y)

nx = ny = nz = NUM_CELLS
grid = pv.ImageData()
grid.dimensions = np.array((nx, ny, nz)) + 1  # number of points!
grid.spacing = (1,1,1)

# Initial data
Z = var.value.reshape((nx, ny, nz), order='F')
grid.cell_data["concentration"] = Z.flatten(order='F')

plotter = pv.Plotter()
actor = plotter.add_volume(grid, scalars="concentration", cmap='viridis')
plotter.show_bounds(grid='back', location='outer', all_edges=True, font_size=16, color='black')
plotter.show(auto_close=False, interactive_update=True)

# --- Simulation loop ---
for step in range(steps):
    start_time = time.time()
    var.updateOld()
    data[step, :] = var.value

    residual = pde_problem.sweep_eq(dt)
    print(f"Step {step}: Time taken: {time.time() - start_time:.4f} seconds | Residual error: {residual}")

    # Update the grid data
    Z = var.value.reshape((NUM_CELLS, NUM_CELLS, NUM_CELLS), order='F')
    grid.cell_data["my_scalars"] = Z.flatten(order='F')

    # Remove the old volume and add new
    plotter.remove_actor(actor)
    actor = plotter.add_volume(grid, scalars="my_scalars", cmap='viridis')

    # Render
    plotter.render()

    input("Next timestep")


np.save("3d-data.npy", data)
