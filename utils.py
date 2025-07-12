import numpy as np

#utils
def get_cartesian_concentration(values, num_cells):
    positions = []
    for idx, value in enumerate(values):
        z = int(idx // (num_cells ** 2))
        y = int((idx % (num_cells ** 2)) // num_cells)
        x = int(idx % num_cells)
        positions.append((x,y,z,value))

    return np.array(positions)

def get_cartesian_value(data, x, y, z):
    mask = (data[:,0]==x) & (data[:,1]==y) & (data[:,2]==z)
    return data[mask, 3]

