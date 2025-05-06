Help on function Grid3D in module fipy.meshes.factoryMeshes:

GGrriidd33DD(dx=1.0, dy=1.0, dz=1.0, nx=None, ny=None, nz=None, Lx=None, Ly=None, Lz=None, overlap=2, communicator=SerialPETScCommWrapper())
    Create a 3D Cartesian mesh
    
    Factory function to select between
    :class:`~fipy.meshes.uniformGrid3D.UniformGrid3D` and
    :class:`~fipy.meshes.nonUniformGrid3D.NonUniformGrid3D`.  If `L{x,y,z}`
    is specified, the length of the domain is always `L{x,y,z}` regardless
    of `d{x,y,z}`, unless `d{x,y,z}` is a list of spacings, in which case
    `L{x,y,z}` will be the sum of `d{x,y,z}` and `n{x,y,z}` will be the
    count of `d{x,y,z}`.
    
    Parameters
    ----------
    dx : float
        Grid spacing in the horizontal direction
    dy : float
        Grid spacing in the vertical direction
    dz : float
        Grid spacing in the depth direction
    nx : int
        Number of cells in the horizontal direction
    ny : int
        Number of cells in the vertical direction
    nz : int
        Number of cells in the depth direction
    Lx : float
        Domain length in the horizontal direction
    Ly : float
        Domain length in the vertical direction
    Lz : float
        Domain length in the depth direction
    overlap : int
        Number of overlapping cells for parallel simulations.  Generally 2
        is adequate.  Higher order equations or discretizations require
        more.
    communicator : ~fipy.tools.comms.commWrapper.CommWrapper
        MPI communicator to use.  Select :attr:`~fipy.tools.serialComm` to
        create a serial mesh when running in parallel; mostly used for test
        purposes.  (default: :attr:`~fipy.tools.parallelComm`).
