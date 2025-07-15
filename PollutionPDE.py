
#!/usr/bin/env python3
from typing import Tuple
import numpy as np
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, FixedValue
from fipy.terms.transientTerm import TransientTerm

class PollutionPDE:
    """
    A class to set up and manage a 3D pollution diffusion-convection PDE using FiPy.
    """
    def __init__(self, num_cells: int, diffusion_coef: float, convection_coef: Tuple[float, float, float]):
        """
        Initializes the PollutionPDE simulation.

        Args:
            num_cells (int): The number of cells along each axis of the 3D grid.
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
        
        # Run the setup
        self.setup_pde()

    def setup_pde(self):
        """
        Configures the 3D grid, variables, boundary conditions, and the PDE itself.
        """
        # Grid setup
        nx = ny = nz = self.num_cells
        L = 1.0
        dx = dy = dz = L / nx
        self.mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

        # Define the pollution source region
        center_x = self.num_cells / 2
        center_y = self.num_cells / 2
        radius = self.num_cells / 3 # Made radius slightly smaller for a more defined source
        
        source_strength = 2.0
        source_region = np.zeros(self.mesh.numberOfCells, dtype=bool)
        
        # Vectorized way to define the cylindrical source region for efficiency
        x_coords, y_coords, z_coords = self.mesh.cellCenters
        x_indices = (x_coords / dx).astype(int)
        y_indices = (y_coords / dy).astype(int)
        z_indices = (z_coords / dz).astype(int)

        mask = (x_indices < 10) & ((z_indices - center_x)**2 + (y_indices - center_y)**2 < radius**2)
        source_region[mask] = True

        # Define the cell variables
        self.pollution_var = CellVariable(mesh=self.mesh, name="pollutant", hasOld=True)
        self.source = CellVariable(name="source", mesh=self.mesh, value=0.0)
        self.source.setValue(source_strength, where=source_region)
        print(f"Source Sum: {np.sum(np.array(self.source.value))}")

        print(self.mesh.facesTop)
        # Boundary conditions (concentration is zero at all boundaries)
        self.pollution_var.constrain(0, self.mesh.facesTop) #Y-Max
        self.pollution_var.constrain(0, self.mesh.facesBottom) #Y-Min
        self.pollution_var.constrain(0, self.mesh.facesFront) # Z-bottom
        self.pollution_var.constrain(0, self.mesh.facesBack)  # Z-top
        self.pollution_var.constrain(0, self.mesh.facesLeft) #X-min
        self.pollution_var.constrain(0, self.mesh.facesRight) #X-Max

        # Define the transient convection-diffusion equation
        diffusion_term = DiffusionTerm(coeff=self.diffusion_coef)
        convection_term = PowerLawConvectionTerm(coeff=self.convection_coef)
        
        self.eq = TransientTerm() == diffusion_term - convection_term + self.source



    def sweep_eq(self, dt) -> float:
        """
        Solves and returns the residual error for the equation.

        Returns:
            Float: residual error.
        """

        res = self.eq.sweep(var=self.get_variable(), dt=dt)
        return res

    def get_mesh(self) -> Grid3D:
        """
        Returns the simulation mesh.

        Returns:
            Grid3D: The FiPy grid object.
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
