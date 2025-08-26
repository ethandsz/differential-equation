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
import torch.autograd as autograd         
from torch import Tensor                 
import torch.nn as nn                   
import torch.optim as optim            

from PollutionPDE1D import PollutionPDE1D
from torch.utils.data import Dataset, DataLoader


NUM_TIMESTEPS = 300
NUM_CELLS = 100

class PollutionDataset(Dataset):
    def __init__(self, data):
        # data: list of ((x, y, z, t), c) in this case just x t and c
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (x, t), c = self.data[idx]
        return torch.tensor([x, t], dtype=torch.float32), torch.tensor(c, dtype=torch.float32)


class PINN_1D(nn.Module):
    def __init__(self):
        super(PINN_1D, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1),
        )
        self.delta = nn.Parameter(torch.tensor([[0.1]]))
        self.region = nn.Parameter(torch.tensor([[0.9]]))
        self.strength = nn.Parameter(torch.tensor([[10.0]]))

    def forward(self, coords):
        # print("Forwarding ", coords)

        return self.hidden(coords).squeeze(-1)  # shape [batch_size]

def normalize_c(c_val, c_min, c_max):
    return (c_val - c_min) / (c_max - c_min)

#function to de-normalize (to check physical predictions later) - used in evalpinn.py
def denormalize_c(c_norm, c_min, c_max):
    return c_norm * (c_max - c_min) + c_min


def generate_bc_inputs(axis_idx, value, device):
    bc_face = torch.rand(NUM_CELLS * 100, 2).to(device)
    bc_face[:,axis_idx] = value
    return bc_face

def pde_residual(coords, model, c_min, c_max):
    coords.requires_grad=True
    c_norm = model(coords)
    c_physical = c_norm * (c_max - c_min) + c_min


    data_size = coords.shape[0]
    scaling_factor = data_size/NUM_CELLS

    grads = autograd.grad(c_physical, coords, grad_outputs=torch.ones_like(c_physical), create_graph=True)[0]
    c_x = grads[:,0]
    c_t = grads[:,1]

    grads_x = autograd.grad(c_x, coords, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_xx = grads_x[:,0]

    delta = model.delta.squeeze()  

    v = 1.0

    source = torch.zeros_like(c_t)
    # print(coords.shape)
    x_idxs = (coords[:,0] * (NUM_CELLS-1))
    # print(torch.max(x_idxs))
    
    # source_region = 0.2
    # source[x_idxs*scaling_factor < source_region * data_size] = model.strength.squeeze()

    alpha = 50.0
    x_norm = coords[:,0]  # already in [0,1]

    indicator = torch.sigmoid( alpha * (model.region - x_norm) )
    # continuous source
    source = model.strength * indicator



    residual = c_t + v * c_x - delta * c_xx - source #(source_strength * predicted_booleans)
    # print(f"Residual with the right source: {torch.sum(residual_right_source)}")
    # print(f"Residual with the learned source: {torch.sum(residual)}")
    return torch.mean(residual**2)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
#----------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
    if device == 'cuda': 
        print(torch.cuda.get_device_name()) 

    start_idx = 0
    end_idx = 100
    dt = 1/60
    all_values = np.load("1d-data.npy")
    training_data = all_values[start_idx:end_idx,0:30]
    print(training_data)
    c_min = np.min(training_data)
    c_max = np.max(training_data)

    t_min = start_idx * dt
    t_max = end_idx * dt

    data = []
    # all_values[4,:]

    for step, _ in enumerate(training_data):
        for idx, value in enumerate(training_data[step]):
            if(idx % 2 == 0):
            #
                physical_c = value
                normalized_c = normalize_c(physical_c, c_min, c_max) 
                x_normalized = idx / (NUM_CELLS-1)
                timestep = (step+start_idx) * dt
                t_normalized = (timestep - t_min) / (t_max - t_min)

                data.append(((x_normalized, t_normalized), normalized_c))
        # print(step * dt)

    for d in data:
        print(d)

    dataset = PollutionDataset(data)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)


    model = PINN_1D().to(device)

    #optimizer for the neural network weights with a smaller learning rate
    optimizer_net = optim.Adam(model.hidden.parameters(), lr=1e-3)

    #ptimizer specifically for the delta parameter 
    optimizer_pde_coefs = optim.Adam([model.delta, model.region, model.strength], lr=1e-3)

    # Hyperparameters
    num_epochs = 35000


    print("Starting training...")
    start_time = time.time()

    pde_loss_over_time = []
    data_loss_over_time = []
    delta_values_over_time = []
    region_values_over_time = []
    stength_values_over_time = []

    best_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            
            epoch_loss = 0.0
            pde_loss_total = 0.0
            data_loss_total = 0.0

            loss_bc_total = 0.0
            loss_ic_total = 0.0
            for coords, c in loader:
                coords = coords.to(device)

                initial_conditions = torch.rand(NUM_CELLS * 100, 2).to(device)
                initial_conditions[:,1] = 0
                c_pred_ic = model(initial_conditions)
                loss_ic = torch.mean(c_pred_ic ** 2)


                cell_idxs = 0

                min_bc_position = 0
                max_bc_position = 1
                c_pred_left = model(generate_bc_inputs(cell_idxs, min_bc_position, device))
                c_pred_right = model(generate_bc_inputs(cell_idxs, max_bc_position, device))


                loss_bc = torch.mean((c_pred_left - torch.full_like(c_pred_left, 0))**2) + \
                            torch.mean((c_pred_right - torch.full_like(c_pred_right, 0))**2) 


                c = c.to(device)
                optimizer_net.zero_grad()

                optimizer_pde_coefs.zero_grad()
                c_pred = model(coords)  # coords shape: [batch_size, 3]

                data_loss = torch.mean((c_pred - c) ** 2) 
                
                N_collocation = 10000
                x_colloc = torch.rand(N_collocation, 1).to(device)
                t_colloc = torch.rand(N_collocation, 1).to(device)
                coords_colloc = torch.cat([x_colloc, t_colloc], dim=1)

                pde_loss = pde_residual(coords_colloc, model, c_min, c_max)


                loss = (100 * data_loss) + (100 *loss_bc) + (100 * loss_ic) + pde_loss
                loss.backward()
                optimizer_pde_coefs.step()
                optimizer_net.step()

                with torch.no_grad():
                    pde_loss_total += pde_loss.item()
                    data_loss_total += data_loss.item()
                    loss_bc_total += loss_bc.item()
                    loss_ic_total += loss_ic.item()
                    epoch_loss += loss.item()


            pde_loss_over_time.append(pde_loss_total)
            data_loss_over_time.append(data_loss_total)
            delta_values_over_time.append(model.delta.item())
            region_values_over_time.append(model.region.item())
            stength_values_over_time.append(model.strength.item())
               

            epoch_end_time = time.time()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Loss Pde: {pde_loss_total:.6f} | "
                  f"Loss Data: {data_loss_total:.6f} | "
                  f"Epoch time (s): {(epoch_end_time - epoch_start_time):.1f} | "
                  f"Time Left: {(((epoch_end_time - epoch_start_time) * (num_epochs - epoch))/3600):2f} hours | "
                  f"Delta: {model.delta.item():.6f} | "
                  f"Region: {model.region.item():.6f} | "
                  f"Stength {model.strength.item():.6f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model_1d.pth')
                print(f"Best model saved at epoch {epoch+1} with val loss {epoch_loss:.4f}")

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("Training killed")

    finally:
# Convert tensors to floats if they are still tensors
        pde_loss_over_time = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in pde_loss_over_time]
        data_loss_over_time = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in data_loss_over_time]

# Create a figure with two subplots: losses and delta
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot losses
        axs[0].plot(pde_loss_over_time, label='PDE Loss', color='tab:blue')
        axs[0].plot(data_loss_over_time, label='Data Loss', color='tab:orange')
        axs[0].set_ylabel('Loss Value')
        axs[0].set_title('Training Losses Over Time')
        axs[0].legend()
        axs[0].grid(True)

# Plot delta values separately
        axs[1].plot(delta_values_over_time, label='Delta Parameter', color='tab:green')
        axs[1].plot(region_values_over_time, label='Region Parameter', color='tab:blue')
        axs[1].plot(stength_values_over_time, label='Strength Parameter', color='tab:red')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Model Parameter Values')
        axs[1].set_title('Model Parameters Over Time')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
