import importlib.util
from petsc4py.PETSc import ScalarType  # type: ignore
import pyvista
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
)
from ufl import ds, dx, grad, inner
from dolfinx.mesh import create_unit_square
k0 = 4 * np.pi

# Approximation space polynomial degree
deg = 1

# Number of elements in each direction of the mesh
n_elem = 64

msh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)

is_complex_mode = np.issubdtype(PETSc.ScalarType, np.complexfloating)

# Source amplitude
A = 1

# Test and trial function space
V = functionspace(msh, ("Lagrange", deg))
V_exact = functionspace(msh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)

# Define variational problem:
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx

# solve for plane wave with mixed Dirichlet and Neumann BCs
theta = np.pi / 4
u_exact.interpolate(lambda x: A * np.exp(1j * k0 * (np.cos(theta) * x[0] + np.sin(theta) * x[1])))
n = ufl.FacetNormal(msh)
g = -ufl.dot(n, ufl.grad(u_exact))
L = -ufl.inner(g, v) * ufl.ds


dofs_D = locate_dofs_geometrical(
    V, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
)
u_bc = Function(V)
u_bc.interpolate(u_exact)
bcs = [dirichletbc(u_bc, dofs_D)]

# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    u=uh,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
problem.solve()
# Plotting stuff
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
plotter = pyvista.Plotter()

grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter.add_mesh(grid, show_edges=True)
plotter.show()
