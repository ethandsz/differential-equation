#!/usr/bin/env python3
from typing import Tuple
import numpy as np
from fipy import CellVariable, Grid1D, DiffusionTerm, PowerLawConvectionTerm, FixedValue, FaceVariable
from fipy.terms.transientTerm import TransientTerm

class PollutionPDE1D:
    """
    A class to set up and manage a 1D pollution diffusion-convection PDE using FiPy.
    """
    def __init__(self, num_cells: int, diffusion_coef: float, convection_coef: float):
        """
        Initializes the PollutionPDE simulation.

        Args:
            num_cells (int): The number of cells along each axis of the 1D grid.
            diffusion_coef (float): The diffusion coefficient.
            convection_coef (Tuple[float, float, float]): The convection coefficient vector.
        """
        self.num_cells = num_cells
        self.diffusion_coef = diffusion_coef
        self.convection_coef = convection_coef
        
        # Instance variables to be populated by setup_pde
        self.mesh = None
        self.pollution_var = None
        self.eq = None
        
        self.source = None
        # Run the setup
        self.setup_pde()

    def setup_pde(self):
        """
        Configures the 1D grid, variables, boundary conditions, and the PDE itself.
        """
        # Grid setup
        nx = self.num_cells
        L = 1.0
        dx  = L / nx
        self.mesh = Grid1D(dx=dx,  nx=nx,)

        
        source_strength = 20.0
        source_region = np.zeros(self.mesh.numberOfCells, dtype=bool)

        cell_centers = self.mesh.cellCenters[0]
        source_region[cell_centers < 0.2] = True
        
        # Vectorized way to define the cylindrical source region for efficiency
        # x_coords = self.mesh.cellCenters
        #
        # mask = (x_indices < 10)         
        # source_region[mask] = True
        #

        # Define the cell variables
        self.pollution_var = CellVariable(mesh=self.mesh, name="pollutant", hasOld=True)
        self.source = CellVariable(name="source", mesh=self.mesh, value=0.0)
        self.source.setValue(source_strength, where=source_region)
        # print(f"Source Sum: {np.sum(np.array(self.source.value))}")

        velocity = FaceVariable(mesh=self.mesh, rank=1, value=self.convection_coef)
        # Boundary conditions (concentration is zero at all boundaries)
        self.pollution_var.constrain(0, self.mesh.facesLeft) #X-min
        self.pollution_var.constrain(0, self.mesh.facesRight) #X-Max

        # Define the transient convection-diffusion equation
        diffusion_term = DiffusionTerm(coeff=self.diffusion_coef)
        convection_term = PowerLawConvectionTerm(coeff=velocity)
        
        self.eq = TransientTerm() == diffusion_term - convection_term + self.source




    def sweep_eq(self, dt) -> float:
        """
        Solves and returns the residual error for the equation.

        Returns:
            Float: residual error.
        """

        self.eq.cacheMatrix()
        self.eq.cacheRHSvector()
        self.pollution_var.updateOld()
        res = self.eq.sweep(var=self.get_variable(), dt=dt)
        return res

    def get_eq_matrix(self):
        return self.eq.matrix.numpyArray

    def get_eq_rhsVector(self):
        return self.eq.RHSvector

    def get_mesh(self) -> Grid1D:
        """
        Returns the simulation mesh.

        Returns:
            Grid1D: The FiPy grid object.
        """
        return self.mesh

    def get_variable(self) -> CellVariable:
        """
        Returns the cell variable representing the pollutant concentration.

        Returns:
            CellVariable: The FiPy variable for the pollutant.
        """
        return self.pollution_var

    def get_source(self) -> CellVariable:
        return self.source

    def get_equation(self):
        """
        Returns the configured PDE.

        Returns:
            The FiPy equation object.
        """
        return self.eq
