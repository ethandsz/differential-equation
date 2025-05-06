from fipy import CellVariable, Grid3D, DefaultAsymmetricSolver, DiffusionTerm, ExponentialConvectionTerm, PowerLawConvectionTerm, Viewer 
from fipy.terms.transientTerm import TransientTerm
from fipy.tools import numerix

diffusionCoef = 1.0
sourceCoef = 1.0
convectionCoef = (1., 0., 0.)
dt = 0.01

L = 10.0
nx = ny = nz = 10.0 # number of cells 
dx = dy = dz = L/nx #Grid Spacing
mesh = Grid3D(dx = dx, dy = dy, dz=dz, nx = nx, ny = ny, nz=nz)

var = CellVariable(mesh=mesh, name="variable", hasOld=1)
var.constrain(0, mesh.facesTop)
var.constrain(1, mesh.facesBottom)

var.constrain(0, mesh.facesFront)
var.constrain(0, mesh.facesBack)

var.constrain(0, mesh.facesLeft)
var.constrain(0, mesh.facesRight)
# var.constrain(None, mesh.facesRight)
# var.faceGrad.constrain(0, mesh.facesRight)
# var.faceGrad.constrain(1, mesh.facesLeft)

eq = (TransientTerm() - DiffusionTerm(coeff=0.0) + PowerLawConvectionTerm(coeff=convectionCoef) + sourceCoef)
# eq = (TransientTerm() - ExponentialConvectionTerm(coeff=convectionCoef))
# eq = (TransientTerm() == DiffusionTerm(coeff=diffusionCoef)) 
steps = 300
sweeps = 2
viewer = Viewer(vars=var)

viewer.plot()
input("start sim?")
for step in range(steps):
    var.updateOld()
    viewer.plot()
    print(step)
    for sweep in range(sweeps):
        eq.solve(var=var, solver=DefaultAsymmetricSolver(tolerance=1e-15, iterations=10000), dt=dt)


input("Exit")
