
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer
from fipy.terms.transientTerm import TransientTerm
from fipy.tools import numerix

diffusionCoef = 0.1
convectionCoef = (0., 0., 1.0)
dt = 0.01
steps = 300

nx = ny = nz = 25 # number of cells 
L = 1.0
dx = dy = dz = L / nx #grid spacing
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

x, y, z = mesh.cellCenters
sourceStrength = 10.0
# sourceRegion = ((x - L/2)**2 + (y - L/2)**2 + (z - 0.0)**2) < (0.5**2) 

sourceRegion = (
    (x - L/2)**2 + (y - L/2)**2 < (0.1**2)
) & (z < 0.5)  #bottom
print(sourceRegion)

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

viewer = Viewer(vars=var)
# viewer = Viewer(vars=source)

input("Start sim?")
for step in range(steps):
    var.updateOld()
    eq.solve(var=var, dt=dt)
    viewer.plot()
    print(f"Step {step}")

input("Exit")
