from operator import index

from torch.cuda import device_count
from utils import get_cartesian_concentration, get_cartesian_value
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
from fipy import CellVariable, Grid3D, DiffusionTerm, PowerLawConvectionTerm, Viewer, ImplicitSourceTerm
from fipy.terms.transientTerm import TransientTerm
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

NUM_TIMESTEPS = 50

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(4,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,1),
        )
        # self.delta = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor([[0.1]]))

    def forward(self, x, y, z, t):
        # inputs = torch.cat([x,y,z,t], dim=1)

        inputs = torch.cat([x.view(-1,1), y.view(-1,1), z.view(-1,1), t.view(-1,1)], dim=1)
        c = self.hidden(inputs)
        return c


def pde_residual(x, y, z, t, model, sourceRegion):
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True

    c = model(x,y,z,t)

    c_t = torch.autograd.grad(c, t, torch.ones_like(c), create_graph=True)[0] #dc/dt

    c_x = torch.autograd.grad(c, x, torch.ones_like(c), create_graph=True)[0] #dc/dx
    c_y = torch.autograd.grad(c, y, torch.ones_like(c), create_graph=True)[0] #dc/dy
    c_z = torch.autograd.grad(c, z, torch.ones_like(c), create_graph=True)[0] #dc/dz


    c_xx = torch.autograd.grad(c_x, x, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2x
    c_yy = torch.autograd.grad(c_y, y, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2y
    c_zz = torch.autograd.grad(c_z, z, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2z

    # print(f"x: {x.shape}")
    # print(f"source: {source.shape}")
    # print(f"c: {c.shape}")

    # sourceStrength = 2.0
    # source_value = sourceStrength if get_cartesian_value(source, x, y, z) == 1 else 0 


    # sourceRegion_tensor = torch.tensor(sourceRegion, dtype=torch.float32, device=device).view(-1,1)

    # source_values = sourceRegion_tensor.repeat(NUM_TIMESTEPS, 1)
# shape: [15625 * 50, 1] = [781250,1]
    
    residual = c_t - model.delta.abs() * (c_xx + c_yy + c_zz) + (1*c_x) - sourceRegion

    return residual

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 


#----------------------------------------------------------------------------------------------------


NUM_CELLS = 25
nx = ny = nz = 25

center_x = NUM_CELLS/2
center_y = NUM_CELLS/2
radius = NUM_CELLS


sourceStrength = 2.0
sourceRegion = []
for idx in range(nx * ny * nz):
    z_pos = idx // (nx * ny)
    y_pos = (idx % (nx * ny)) // nx
    x_pos = idx % nx

    if (x_pos < 10) and ((z_pos-center_x)**2 + (y_pos - center_y)**2 < radius):
        sourceRegion.append(True)
    else:
        sourceRegion.append(False)



#----------------------------------------------------------------------------------------------------

# x_train, y_train, z_train, t_train = torch.meshgrid(x.squeeze(), y.squeeze(), z.squeeze(), t.squeeze(), indexing="xy")
#
# x_train = x_train.reshape(-1,1)
# y_train = y_train.reshape(-1,1)
# z_train = z_train.reshape(-1,1)
# t_train = t_train.reshape(-1,1)


all_values = np.load("tall-data.npy")

pollutant_values = []

steps, _ = all_values.shape
for step in range(steps):
    value_t = all_values[step, :]
    pollutant_value = get_cartesian_concentration(value_t, 25)
    pollutant_values.append(pollutant_value)


pollutant_values = np.array(pollutant_values)
print(f"pollutant_values shape: {pollutant_values.shape}") # (300, 15625, 4) timestep, gridsize, xyz-concentration
print(get_cartesian_value(pollutant_values[0], 5, 3, 6))

#
x = torch.arange(0, NUM_CELLS, dtype=torch.float32)
y = torch.arange(0, NUM_CELLS, dtype=torch.float32)
z = torch.arange(0, NUM_CELLS, dtype=torch.float32)
t = torch.arange(0, NUM_TIMESTEPS, dtype=torch.float32)


# Create meshgrid for all spatio-temporal points
# Note: Using 'ij' indexing to match NumPy's default (z, y, x) order if you load data
gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')

# Flatten and repeat to create the full training data coordinates
num_spatial_points = NUM_CELLS * NUM_CELLS * NUM_CELLS
x_train = gx.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)
y_train = gy.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)
z_train = gz.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)
t_train = t.repeat_interleave(num_spatial_points).view(-1, 1)


# Create the source term tensor for all spatio-temporal points
sourceRegion = [s * sourceStrength for s in sourceRegion]
source_map_spatial = torch.tensor(sourceRegion, dtype=torch.float32).view(-1, 1)
source_values = source_map_spatial.repeat(NUM_TIMESTEPS, 1)


model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


source_values = torch.tensor(sourceRegion, dtype=torch.float32).view(-1,1)
# Repeat over timesteps if needed (check your source logic)
source_values = source_values.repeat(NUM_TIMESTEPS, 1).to(device)

c_data = torch.tensor(all_values[:NUM_TIMESTEPS, :].flatten(), dtype=torch.float32).view(-1, 1)

x_train = x_train.to(device)
y_train = y_train.to(device)
z_train = z_train.to(device)
t_train = t_train.to(device)
c_data = c_data.to(device)
source_values = source_values.to(device)


# Hyperparameters
num_epochs = 12000
batch_size = 4096 * 32 # Adjust based on your GPU memory
dataset_size = x_train.shape[0]

# --- 5. BATCHED TRAINING LOOP ---

print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    
    # Shuffle indices for random batching
    indices = torch.randperm(dataset_size)
    
    epoch_loss = 0.0
    pde_loss = 0.0
    data_loss = 0.0

    for i in range(0, dataset_size, batch_size):
        batch_idx = indices[i:i+batch_size]
        
        # Get data for the current batch
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
        z_batch = z_train[batch_idx]
        t_batch = t_train[batch_idx]
        c_batch = c_data[batch_idx]
        source_batch = source_values[batch_idx]
        
        # --- Calculate Losses ---
        
        # 1. PDE Loss (Physics Loss)
        residual = pde_residual(x_batch, y_batch, z_batch, t_batch, model, source_batch)
        loss_pde = torch.mean(residual**2)
        
        # 2. Data Loss (MSE Loss)
        c_pred = model(x_batch, y_batch, z_batch, t_batch)
        loss_data = 1000 * torch.mean((c_pred - c_batch)**2)
        
        # Combine losses with a weighting factor
        loss = loss_pde + loss_data
        
        # --- Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() 
        pde_loss += loss_pde
        data_loss += loss_data
        
    avg_epoch_loss = epoch_loss / dataset_size

    epoch_end_time = time.time()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Loss: {epoch_loss:.6f} | "
          f"Loss Pde: {pde_loss:.6f} | "
          f"Loss Data: {data_loss:.6f} | "
          f"Epoch time: {epoch_end_time - epoch_start_time} | "
          f"Time Left: {(((epoch_end_time - epoch_start_time) * (num_epochs - epoch))/3600):2f} hours | "
          f"Delta: {model.delta.item():.6f}")

# --- FIX: REMOVED THE SECOND, FULL-BATCH TRAINING LOOP ---
# The following loop was the source of the out-of-memory error and has been removed.
# It is redundant and computationally infeasible for large datasets.

total_time = time.time() - start_time
print(f"Training finished in {total_time:.2f} seconds.")

