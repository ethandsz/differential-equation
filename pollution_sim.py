from operator import index
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer
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
POLLUTANT_DETECTION_THRESHOLD = 0.05

parser = argparse.ArgumentParser(prog='PDE Simulation', usage='[options]')
parser.add_argument('--useMatplot', type=str2bool, nargs='?', const=True, default=False, help='Enable matplot viewer for a lighter weight simulation')
parser.add_argument('--useMayavi', type=str2bool, nargs='?', const=True, default=True, help='Disable/Enable viewer')
parser.add_argument('--simSteps', type=int, default=300, help='Number of simulation steps')
parser.add_argument('--xFlow', type=float, default=0.0, help='Flow in X direction')
parser.add_argument('--yFlow', type=float, default=0.0, help='Flow in Y direction')
parser.add_argument('--zFlow', type=float, default=1.0, help='Flow in Z direction')

args = parser.parse_args()
useMayavi = args.useMayavi
useMatplot = args.useMatplot

diffusionCoef = 0.1
convectionCoef = (args.xFlow, args.yFlow, args.zFlow)
dt = 0.01
steps = args.simSteps

nx = ny = nz = NUM_CELLS # number of cells 
L = 1.0
dx = dy = dz = L / nx #grid spacing
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

x, y, z = mesh.cellCenters
sourceStrength = 1.0
sourceRegion = ((x - L/2)**2 + (y - L/2)**2 + (z - 0.0)**2) < (0.5**2) 


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
    start_time = time.time()
    var.updateOld()
    eq.solve(var=var, dt=dt)
    if useMatplot:
        num_cells = nx * ny * nz
        slice_z = num_cells/nz
        body = np.array(var)

        values = np.array(var.value)
        x_vals = np.array(x)
        y_vals = np.array(y)
        z_vals = np.array(z)

        x_unique = np.unique(x_vals)
        avg_values_x_slice = []

        #Extracting average pollutant values across all x slices
        for x_val in x_unique:
            mask = np.isclose(x_vals, x_val)
            avg_x = values[mask].mean()
            avg_values_x_slice.append(avg_x)

        y_unique = np.unique(y_vals)
        avg_values_y_slice = []

        for y_val in y_unique:
            mask = np.isclose(y_vals, y_val)
            avg_y = values[mask].mean()
            avg_values_y_slice.append(avg_y)

        z_unique = np.unique(z_vals)
        avg_values_z_slice = []

        for z_val in z_unique:
            mask = np.isclose(z_vals, z_val)
            avg_z = values[mask].mean()
            avg_values_z_slice.append(avg_z)

        #List to store the 'grid' of the body of water which is used later to show pollutant positions
        positions = np.linspace(0,1,NUM_CELLS)

        #Get the index of the pollutant if greater than a threshold
        for p in avg_values_x_slice:
            if p > POLLUTANT_DETECTION_THRESHOLD:
                #Index of the pollutant accross the x slice
                index_of_pollutant = avg_values_x_slice.index(p)
                pollutant_position_overtime_x.append(positions[index_of_pollutant])

                #Get the indices of the max values of the other slices since consider these to be above the x axis threshold
                current_y_position_index = avg_values_y_slice.index(max(avg_values_y_slice))
                current_z_position_index = avg_values_z_slice.index(max(avg_values_z_slice))
                pollutant_position_overtime_y.append(positions[current_y_position_index])
                pollutant_position_overtime_z.append(positions[current_z_position_index])


        for p in avg_values_y_slice:
            if p > POLLUTANT_DETECTION_THRESHOLD:
                index_of_pollutant = avg_values_y_slice.index(p)
                pollutant_position_overtime_y.append(positions[index_of_pollutant])

                current_x_position_index = avg_values_x_slice.index(max(avg_values_x_slice))
                current_z_position_index = avg_values_z_slice.index(max(avg_values_z_slice))
                pollutant_position_overtime_x.append(positions[current_x_position_index])
                pollutant_position_overtime_z.append(positions[current_z_position_index])


        for p in avg_values_z_slice:
            if p > POLLUTANT_DETECTION_THRESHOLD:
                index_of_pollutant = avg_values_z_slice.index(p)
                pollutant_position_overtime_z.append(positions[index_of_pollutant])


                current_x_position_index = avg_values_x_slice.index(max(avg_values_x_slice))
                current_y_position_index = avg_values_y_slice.index(max(avg_values_y_slice))
                pollutant_position_overtime_x.append(positions[current_x_position_index])
                pollutant_position_overtime_y.append(positions[current_y_position_index])

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax.clear()


        ax.scatter(pollutant_position_overtime_x, pollutant_position_overtime_y, pollutant_position_overtime_z, label='pollutant')
        ax.set_title("Contaminated zone")

        ax.legend()
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
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
