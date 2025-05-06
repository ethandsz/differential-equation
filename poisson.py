import importlib.util
from petsc4py.PETSc import ScalarType  # type: ignore
import pyvista
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

msh = mesh.create_rectangle(comm = MPI.COMM_WORLD, points=((0.0, 0.0), (2.0, 1.0)), n = (32, 16), cell_type=mesh.CellType.triangle)
V = fem.functionspace(msh, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0)) #Locate edges of the mesh whos xcords are 0 or 2 
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets) # Get the edges of the function space
# print(facets.shape)
# print(facets)
# print(dofs.shape)
# print(dofs)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V) #Apply dirichlet boundary condition on the function space. 

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

f= 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Plotting stuff
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
plotter = pyvista.Plotter()

dof_points = x[dofs]
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter.add_points(dof_points, color="blue", point_size=10)
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()
