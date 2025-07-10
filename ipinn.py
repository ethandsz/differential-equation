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

NUM_TIMESTEPS = 25

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
        self.delta = nn.Parameter(torch.rand(1, dtype=torch.float32))

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


    sourceRegion_tensor = torch.tensor(sourceRegion, dtype=torch.float32, device=device).view(-1,1)

    source_values = sourceRegion_tensor.repeat(NUM_TIMESTEPS, 1)
# shape: [15625 * 50, 1] = [781250,1]
    
    residual = c_t - model.delta.abs() * (c_xx + c_yy + c_zz) + (1*c_x) - source_values

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

x = torch.linspace(0,25, int(25/(1/25))).view(-1,1)
y = torch.linspace(0,25,int(25/(1/25))).view(-1,1)
z = torch.linspace(0,25,int(25/(1/25))).view(-1,1)
t = torch.linspace(0,NUM_TIMESTEPS,NUM_TIMESTEPS+1).view(-1,1)

print(t)

x_train, y_train, z_train, t_train = torch.meshgrid(x.squeeze(), y.squeeze(), z.squeeze(), t.squeeze(), indexing="xy")

x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
z_train = z_train.reshape(-1,1)
t_train = t_train.reshape(-1,1)


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
# x_data = torch.tensor()
# y_data = torch.tensor()
# z_data = torch.tensor()
# t_data = torch.tensor()
#

x_data = x_train
y_data = y_train
z_data = z_train
t_data = t_train


c_data = torch.tensor(pollutant_values[:NUM_TIMESTEPS,:,3].reshape(-1,1), dtype=torch.float32).to(device)
print(f"C data: {c_data.shape}")

x_data = x_data.to(device)
y_data = y_data.to(device)
z_data = z_data.to(device)
t_data = t_data.to(device)
c_data = c_data.to(device)

x_train = x_train.to(device)
y_train = y_train.to(device)
z_train = z_train.to(device)
t_train = t_train.to(device)


model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

num_epochs = 12000


batch_size = 2048   # you can adjust; smaller if you still hit OOM

num_epochs = 12000
dataset_size = x_train.shape[0]

source_values = torch.tensor(sourceRegion, dtype=torch.float32).view(-1,1)
# Repeat over timesteps if needed (check your source logic)
source_values = source_values.repeat(NUM_TIMESTEPS, 1).to(device)

for epoch in range(num_epochs):
    model.train()

    # Shuffle indices (CPU tensor)
    indices = torch.randperm(dataset_size)

    epoch_loss = 0.0

    for i in range(0, dataset_size, batch_size):
        batch_idx = indices[i:i+batch_size]   # keep on CPU

        # Fetch data for this batch
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
        z_batch = z_train[batch_idx]
        t_batch = t_train[batch_idx]
        c_batch = c_data[batch_idx]
        source_batch = source_values[batch_idx]

        # Compute residual (pde loss)
        residual = pde_residual(x_batch, y_batch, z_batch, t_batch, model, source_batch)
        loss_pde = torch.mean(residual**2)

        # Predict concentration from model
        c_pred = model(x_batch, y_batch, z_batch, t_batch)
        loss_data = torch.mean((c_pred - c_batch) ** 2)

        # Combine losses
        loss = loss_pde + 100 * loss_data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.6f} "
          f"Delta: {model.delta.item():.6f}")


for epoch in range(num_epochs):
    model.train()


    residual = pde_residual(x_train, y_train, z_train, t_train, model, sourceRegion)
    loss_pde = torch.mean(residual**2)
    c_pred_data = model(x_data, y_data, z_data, t_data)
    loss_data = torch.mean((c_pred_data - c_data) ** 2)

    loss = loss_pde + 100*loss_data

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for t_val in t_data:
    #     for x_val in x_data:
    #         for y_val in y_data:
    #             for z_val in z_data:
    #                 residual = pde_residual(x_val, y_val, z_val, t_val, model, sourceRegion)
    #                 loss_pde = torch.mean(residual**2)
    #                 c_pred_data = model(x_val, y_val, z_val, t_data)

    print(f"Epoch: {epoch} Loss Pde: {loss_pde} Loss data: {loss_data} Delta: {model.delta.item():.6f}")




