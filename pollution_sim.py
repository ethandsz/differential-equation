
from operator import index
import time
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


NUM_CELLS = 50
POLLUTANT_DETECTION_THRESHOLD = 5e-3

parser = argparse.ArgumentParser(prog='PDE Simulation', usage='[options]')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=False, help='Enable matplot viewer for a lighter weight simulation')
parser.add_argument('--useMayavi', type=str2bool, nargs='?', const=True, default=True, help='Disable/Enable viewer')
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

nx = ny = nz = NUM_CELLS # number of cells 

L = 1.0
dx = dy = dz = L / nx #grid spacing
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

x, y, z = mesh.cellCenters

center_x = NUM_CELLS/2
center_y = NUM_CELLS/2
radius = NUM_CELLS


sourceStrength = 2.0
sourceRegion = []
for idx in range(nx * ny * nz):
    z_pos = idx // (nx * ny)
    y_pos = (idx % (nx * ny)) // nx
    x_pos = idx % nx
# (x-center_x)^2 + (y - center_y)^2 < radius^2

    if (x_pos < 10) and ((z_pos-center_x)**2 + (y_pos - center_y)**2 < radius):
        sourceRegion.append(True)
    else:
        sourceRegion.append(False)

var = CellVariable(mesh=mesh, name="pollutant", hasOld=True)
source = CellVariable(name="source", mesh=mesh, value=0.0)
# print(source[:10])
source.setValue(sourceStrength, where=sourceRegion)
# source[:10] = 10.0
# Boundary conditions
var.constrain(0, mesh.facesTop)
var.constrain(0, mesh.facesBottom)

var.constrain(0, mesh.facesFront)#Z-bottom
var.constrain(0, mesh.facesBack)#Z-top 

var.constrain(0, mesh.facesLeft)
var.constrain(0, mesh.facesRight)

# Transient convection-diffusion equation
eq = TransientTerm() == DiffusionTerm(coeff=diffusionCoef) - PowerLawConvectionTerm(coeff=convectionCoef,) + source

viewer = None
if useMayavi:
    viewer = Viewer(vars=var)
    # viewer = Viewer(vars=source)
elif useMatplot:
    plt.ion()

fig = None
ax = None

input("Start sim?")

pollutant_position_overtime_x = []
pollutant_position_overtime_y = []
pollutant_position_overtime_z = []

for step in range(steps):
    input("Hit Enter to step forward")
    start_time = time.time()
    var.updateOld()
    eq.solve(var=var, dt=dt)
    if useMatplot:

        values = var.value
        values = np.array(values)

        positions = []
        for idx, value in enumerate(values):
            z = idx // (nx * ny)
            y = (idx % (nx * ny)) // nx
            x = idx % nx
            positions.append(((x, y, z), value))


        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax.clear()

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


        ax.scatter(x_filtered, y_filtered, z_filtered, c=values_filtered, cmap='plasma', s=20, label='plume')
        # ax.scatter(positions[:,0], positions[:,1],positions[:,2]  label='pollutant')
        ax.set_title("Contaminated zone")

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
