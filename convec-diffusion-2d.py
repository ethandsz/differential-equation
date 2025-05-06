from fipy import CellVariable, Grid1D, DefaultAsymmetricSolver, Grid2D, DiffusionTerm, ExponentialConvectionTerm, Viewer 
from fipy.terms.transientTerm import TransientTerm
from fipy.tools import numerix

diffusionCoef = 10.
sourceCoef = 1.0
convectionCoef = (1., 1.)
dt = 0.01

L = 10.0
nx = ny = 10.0 # number of cells 
dx = dy = L/nx #Grid Spacing
mesh = Grid2D(dx = dx, dy = dy, nx = nx, ny = ny)


valueLeft = 1.0
valueRight = 0.0

var = CellVariable(mesh=mesh, name="variable", hasOld=1)
# var.constrain(None, mesh.facesLeft)
# var.constrain(None, mesh.facesRight)
var.faceGrad.constrain(0, mesh.facesRight)
var.faceGrad.constrain(1, mesh.facesLeft)
var.setValue(1.0, where= (mesh.cellCenters[0] < 2.0) & (mesh.cellCenters[1] > 1.0))


eq = (TransientTerm() == DiffusionTerm(coeff=diffusionCoef) + ExponentialConvectionTerm(coeff=convectionCoef))
# eq = (TransientTerm() - ExponentialConvectionTerm(coeff=convectionCoef))
# eq = (TransientTerm() == DiffusionTerm(coeff=diffusionCoef)) 
steps = 300
sweeps = 2
viewer = Viewer(vars=var)
for step in range(steps):
    var.updateOld()

    viewer.plot()
    for sweep in range(sweeps):
        eq.solve(var=var, solver=DefaultAsymmetricSolver(tolerance=1e-15, iterations=10000), dt=dt)

input("Exit")
