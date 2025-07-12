#250.649057704589
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

from PollutionPDE import PollutionPDE

NUM_TIMESTEPS = 1
NUM_CELLS = 25

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
        self.delta = nn.Parameter(torch.tensor([[0.9]]))
        # self.source_region_param = torch.nn.Parameter(
        #     torch.randn(NUM_TIMESTEPS*(NUM_CELLS**3), 1, dtype=torch.float32)  # or zeros, or something else
        # )

    def forward(self, x, y, z, t):
        # inputs = torch.cat([x,y,z,t], dim=1)

        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"z shape: {z.shape}")
        print(f"t shape: {t.shape}")
        inputs = torch.cat([x.view(-1,1), y.view(-1,1), z.view(-1,1), t.view(-1,1)], dim=1)
        c = self.hidden(inputs)
        return c

    # def get_source_region(self):
    #     # Keep it positive or in [0,1]:
    #     return torch.sigmoid(self.source_region_param)


def pde_residual(x, y, z, t, model):
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True

    c = model(x,y,z,t)
    #
    c_t = torch.autograd.grad(c, t, torch.ones_like(c), create_graph=True)[0] #dc/dt

    c_x = torch.autograd.grad(c, x, torch.ones_like(c), create_graph=True)[0] #dc/dx
    c_y = torch.autograd.grad(c, y, torch.ones_like(c), create_graph=True)[0] #dc/dy
    c_z = torch.autograd.grad(c, z, torch.ones_like(c), create_graph=True)[0] #dc/dz


    c_xx = torch.autograd.grad(c_x, x, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2x
    c_yy = torch.autograd.grad(c_y, y, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2y
    c_zz = torch.autograd.grad(c_z, z, torch.ones_like(c), create_graph=True)[0] #d^2c/d^2z


    pde_problem = PollutionPDE(
            num_cells=NUM_CELLS,
            diffusion_coef=0.1,
            convection_coef=(1.0,0.0,0.0)
        )

# 2. Access the components
    var = pde_problem.get_variable()
    eq = pde_problem.get_equation()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = torch.tensor(pde_problem.get_source().value).to(device)

#
#     print(model.delta.abs().item())
#
#     pde_problem = PollutionPDE(
#             num_cells=25,
#             diffusion_coef=model.delta.abs().item(),
#             convection_coef=(1.0,0.0,0.0)
#         )
#
# # 2. Access the components
#     # var = pde_problem.get_variable()
#     c_flattened = c.cpu().detach().numpy().flatten()
#
#     residual_total = 0
#     total_values = NUM_CELLS ** 3
#     for t in range(NUM_TIMESTEPS):
#         eq = pde_problem.get_equation()
#         mesh = pde_problem.get_mesh()
#         predicted_var = CellVariable(name="pred_poll", mesh=mesh, value=c_flattened[t*total_values:(t+1)*total_values])
#         residual = eq.sweep(var=predicted_var, dt=1/60 * (t+1))
#         # print(f"Residual Error: {residual}")
#         residual_total += residual
#
#
#
#     # print(f"x: {x.shape}")
#     # print(f"source: {source.shape}")
#     # print(f"c: {c.shape}")
#
#     # sourceStrength = 2.0
#     # source_value = sourceStrength if get_cartesian_value(source, x, y, z) == 1 else 0 
#
#
#     # sourceRegion_tensor = torch.tensor(sourceRegion, dtype=torch.float32, device=device).view(-1,1)
#
#     # source_values = sourceRegion_tensor.repeat(NUM_TIMESTEPS, 1)
# # shape: [15625 * 50, 1] = [781250,1]
#
#     print(f"Diffusion part { torch.sum(model.delta.abs() * (c_xx + c_yy + c_zz)) }")
#     print(f"Convection part { torch.sum(1*c_x) }")
#     print(f"Source part {torch.sum(sourceRegion)}")
#     pred = model.delta.abs() * (c_xx + c_yy + c_zz) - (1*c_x) + sourceRegion


    # residual_total = 0
    # total_values = NUM_CELLS ** 3
    # for t in range(NUM_TIMESTEPS):
    #
    #     idx_start = t*total_values
    #     idx_end = (t+1)*total_values
    #
    #     print(f"pred: {torch.sum(c[idx_start:idx_end])}")
    #     print(f"diffusion part: {torch.sum(model.delta.abs() * (c_xx[idx_start:idx_end] + c_yy[idx_start:idx_end] + c_zz[idx_start:idx_end]))}")
    #     print(f"advection part: {torch.sum(1*c_x[idx_start:idx_end])}")
    #     print(f"source part: {torch.sum(source)}")
    #     residual = c[idx_start:idx_end] - (model.delta.abs() * (c_xx[idx_start:idx_end] + c_yy[idx_start:idx_end] + c_zz[idx_start:idx_end])) + (1*c_x[idx_start:idx_end]) - model.sourceRegion.abs()
    #     residual_total += torch.mean(residual**2)
        
    

    print(f"c_t {torch.sum(c_t)}")
    print(f"c sum: {torch.sum(c)}")
    print(f"source sum: {torch.sum(source)}")
    print(f"Diffusion part { torch.sum(model.delta.abs() * (c_xx + c_yy + c_zz)) }")
    print(f"Convection part { torch.sum(1*c_x) }")

    print(f"Sum c_x: {torch.sum(c_x).item()}")
    print(f"Sum c_y: {torch.sum(c_y).item()}")
    print(f"Sum c_z: {torch.sum(c_z).item()}")

    print(f"Sum c_xx: {torch.sum(c_xx).item()}")
    print(f"Sum c_yy: {torch.sum(c_yy).item()}")
    print(f"Sum c_zz: {torch.sum(c_zz).item()}")

    residual = c_t/1000 - ((model.delta.abs() * (c_xx + c_yy + c_zz)) * 10) + (10*(1*c_x)) - source/1000 #(2 * model.source_region_param.abs())

    return torch.mean(residual**2)
    return residual_total

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name()) 


#----------------------------------------------------------------------------------------------------


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
    print(all_values.shape)

    pollutant_values = []
    #
    # steps, _ = all_values.shape
    # for step in range(steps):
    #     value_t = all_values[step, :]
    #     pollutant_value = get_cartesian_concentration(value_t, 25)
    #     pollutant_values.append(pollutant_value)
    #

    value_t = all_values[2, :]
    pollutant_value = get_cartesian_concentration(value_t, 25)
    pollutant_values.append(pollutant_value)

    pollutant_values = np.array(pollutant_values)
    print(f"pollutant_values shape: {pollutant_values.shape}") # (300, 15625, 4) timestep, gridsize, xyz-concentration

#
    x = torch.arange(0, NUM_CELLS,  dtype=torch.float32)
    y = torch.arange(0, NUM_CELLS,  dtype=torch.float32)
    z = torch.arange(0, NUM_CELLS,  dtype=torch.float32)
    t = torch.arange(0, 1/60, 1/60, dtype=torch.float32)


# Create meshgrid for all spatio-temporal points
# Note: Using 'ij' indexing to match NumPy's default (z, y, x) order if you load data
#     gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
#
# # Flatten and repeat to create the full training data coordinates
    num_spatial_points = NUM_CELLS * NUM_CELLS * NUM_CELLS
#     x_train = gx.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)
#     y_train = gy.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)
#     z_train = gz.flatten().repeat(NUM_TIMESTEPS).view(-1, 1)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # shape: (25,25,25)

# Flatten to (15625, 1)
    x_train = xx.reshape(-1,1)
    y_train = yy.reshape(-1,1)
    z_train = zz.reshape(-1,1)

# Repeat for timesteps if you have multiple timesteps
    t_train = t.repeat_interleave(NUM_CELLS**3).view(-1,1)  # shape: (15625,1) if NUM_TIMESTEPS=1


# Create the source term tensor for all spatio-temporal points
    sourceRegion = [s * sourceStrength for s in sourceRegion]
    source_map_spatial = torch.tensor(sourceRegion, dtype=torch.float32).view(-1, 1)
    source_values = source_map_spatial.repeat(NUM_TIMESTEPS, 1)


    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)


    source_values = torch.tensor(sourceRegion, dtype=torch.float32).view(-1,1)
# Repeat over timesteps if needed (check your source logic)
    source_values = source_values.repeat(NUM_TIMESTEPS, 1).to(device)

    c_data = torch.tensor(all_values[4, :].flatten(), dtype=torch.float32).view(-1, 1)

    print(f"C data shape: {c_data.shape}")

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    z_train = z_train.to(device)
    t_train = t_train.to(device)
    c_data = c_data.to(device)
    source_values = source_values.to(device)


# Hyperparameters
    num_epochs = 12000
    # batch_size = 4096 * 32 # Adjust based on your GPU memory
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

        for i in range(0, 1):
            
            # Get data for the current batch
            x_batch = x_train
            y_batch = y_train
            z_batch = z_train
            t_batch = t_train
            c_batch = c_data
            source_batch = source_values
            
            # --- Calculate Losses ---
            
            # 1. PDE Loss (Physics Loss)
            residual = pde_residual(x_batch, y_batch, z_batch, t_batch, model, )
            loss_pde = 10 * residual #torch.mean(residual**2)
            print(f"loss_pde: {loss_pde}")
            
            # 2. Data Loss (MSE Loss)
            c_pred = model(x_batch, y_batch, z_batch, t_batch)
            print(c_pred.shape)
            print(c_batch.shape)
            loss_data = 100 * torch.mean((c_pred - c_batch)**2)
            
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
            f"Epoch time (s): {(epoch_end_time - epoch_start_time):.1f} | "
              f"Time Left: {(((epoch_end_time - epoch_start_time) * (num_epochs - epoch))/3600):2f} hours | "
              f"Delta: {model.delta.item():.6f} | ")

# --- FIX: REMOVED THE SECOND, FULL-BATCH TRAINING LOOP ---
# The following loop was the source of the out-of-memory error and has been removed.
# It is redundant and computationally infeasible for large datasets.

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

    torch.save(model.state_dict(), "model.pth")
