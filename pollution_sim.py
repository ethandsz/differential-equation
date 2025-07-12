from utils import get_cartesian_concentration
from operator import index
import time
from PollutionPDE import PollutionPDE
import matplotlib.pyplot as plt
import argparse
import numpy as np
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer, ImplicitSourceTerm
from fipy.terms.transientTerm import TransientTerm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


NUM_CELLS = 25
POLLUTANT_DETECTION_THRESHOLD = 5e-3

parser = argparse.ArgumentParser(prog='PDE Simulation', usage='[options]')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=True, help='Enable matplot viewer for a lighter weight simulation')
parser.add_argument('--useMayavi', type=str2bool, nargs='?', const=True, default=False, help='Disable/Enable viewer')
parser.add_argument('--simSteps', type=int, default=300, help='Number of simulation steps')
parser.add_argument('--xFlow', type=float, default=1.0, help='Flow in X direction')
parser.add_argument('--yFlow', type=float, default=0.0, help='Flow in Y direction')
parser.add_argument('--zFlow', type=float, default=0.0, help='Flow in Z direction')

args = parser.parse_args()
useMayavi = args.useMayavi
useMatplot = args.useMatplot
if(useMatplot):
    useMayavi = False

diffusionCoef = 0.1
convectionCoef = (args.xFlow, args.yFlow, args.zFlow)
dt = 1/60
steps = args.simSteps

pde_problem = PollutionPDE(
        num_cells=NUM_CELLS,
        diffusion_coef=diffusionCoef,
        convection_coef=convectionCoef
    )

# 2. Access the components
var = pde_problem.get_variable()
eq = pde_problem.get_equation()

viewer = None
if useMayavi:
    viewer = Viewer(vars=var)
    # viewer = Viewer(vars=source)
elif useMatplot:
    plt.ion()

fig = None
ax = None
cb = None        # will hold the current colorbar

input("Start sim?")

pollutant_position_overtime_x = []
pollutant_position_overtime_y = []
pollutant_position_overtime_z = []
all_values = np.zeros((steps,NUM_CELLS**3))
for step in range(steps):
    start_time = time.time()
    var.updateOld()
    eq.solve(var=var, dt=dt)
    # adv_residual = convection_term.justResidualVector(var=var)
    # diff_residual = diffusion_term.justResidualVector(var=var)
    residual = eq.justResidualVector(var=var, dt=dt)
    print(len(var))
    residual = np.array(residual)
    print(np.sum(residual))
    input("Step")
    if useMatplot:
        values = var.value
        values = np.array(values)
        # np.save("t73-data.npy", values)
        all_values[step,:] = values
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax.clear()

        positions = []
        for idx, value in enumerate(values):
            z = idx // (NUM_CELLS ** 2)
            y = (idx % (NUM_CELLS ** 2)) // NUM_CELLS
            x = idx % NUM_CELLS
            positions.append(((x, y, z), value))

        x_filtered = []
        y_filtered = []
        z_filtered = []
        values_filtered = []

        for (x_val, y_val, z_val), v in positions:
            if v >= POLLUTANT_DETECTION_THRESHOLD:
                x_filtered.append(x_val)
                y_filtered.append(y_val)
                z_filtered.append(z_val)
                values_filtered.append(v)

        if cb is not None:
            cb.remove()

        sc = ax.scatter(x_filtered, y_filtered, z_filtered, c=values_filtered, cmap='plasma', s=20, label='plume')
        # print(f"X: {np.array(x_filtered).shape} Y: {y_filtered}, Z: {z_filtered}, C-Value: {np.array(values_filtered).shape}")
        # ax.scatter(positions[:,0], positions[:,1],positions[:,2]  label='pollutant')
        ax.set_title("Contaminated zone")

        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label('Concentration')

        ax.legend()
        ax.set_xlim(0, NUM_CELLS)
        ax.set_ylim(0, NUM_CELLS)
        ax.set_zlim(0, NUM_CELLS)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update


    elif viewer is not None:
        viewer.plot()

    print(f"Step: {step}\nTime taken: {time.time() - start_time}")
if viewer is not None:
    viewer.plot()
input("Exit")

plt.ioff()
if fig is not None:
    plt.close(fig)
#

simulation_steps, _ = all_values.shape
for step in range(simulation_steps):
    concentration_values = get_cartesian_concentration(all_values[step, :], NUM_CELLS)
    #x.y,z,c
    print(concentration_values.shape)
    print(concentration_values[1])


# residuals = []
# vx, vy, vz = convectionCoef
# for t in range(1, steps-1):  # skip first/last step for central diff
#     C_t = (all_values[t + 1] - all_values[t - 1]) / (2 * dt)
#
#     C = all_values[t].reshape(NUM_CELLS, NUM_CELLS, NUM_CELLS)
#
#     C_x = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / (2 * dx)
#     C_y = (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / (2 * dy)
#     C_z = (np.roll(C, -1, axis=2) - np.roll(C, 1, axis=2)) / (2 * dz)
#
#     C_xx = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / (dx**2)
#     C_yy = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / (dy**2)
#     C_zz = (np.roll(C, -1, axis=2) - 2 * C + np.roll(C, 1, axis=2)) / (dz**2)
#
#     S = np.zeros_like(C)
#     S_flat = S.reshape(-1)
#     S_flat[sourceRegion] = sourceStrength
#     S = S_flat.reshape(NUM_CELLS, NUM_CELLS, NUM_CELLS)
#
#     # Now compute residual at this time step
#     f = C_t.reshape(NUM_CELLS, NUM_CELLS, NUM_CELLS) \
#         - diffusionCoef * (C_xx + C_yy + C_zz) \
#         + (vx * C_x + vy * C_y + vz * C_z) 
#
#
#
#     print(np.sum(f))
#     residuals.append(np.linalg.norm(f.ravel(), 2))  # L2 norm of residual
