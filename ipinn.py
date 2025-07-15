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
from torch.utils.data import Dataset, DataLoader


NUM_TIMESTEPS = 1
NUM_CELLS = 25

class PollutionDataset(Dataset):
    def __init__(self, data):
        # data: list of ((x, y, z), c)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (x, y, z, t), c = self.data[idx]
        return torch.tensor([x, y, z, t], dtype=torch.float32), torch.tensor(c, dtype=torch.float32)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(4,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,1),
        )
        self.delta = nn.Parameter(torch.tensor([[0.1]]))

    def forward(self, coords):
        # print("Forwarding ", coords)

        return self.hidden(coords).squeeze(-1)  # shape [batch_size]

def normalize_c(c_val, c_min, c_max):
    return (c_val - c_min) / (c_max - c_min)

# Function to de-normalize (to check physical predictions later)
def denormalize_c(c_norm, c_min, c_max):
    return c_norm * (c_max - c_min) + c_min


def generate_bc_inputs(input_tensor, axis_idx, value):
    bc_face = torch.clone(input_tensor)
    bc_face[:,axis_idx] = value
    bc_face = torch.unique(bc_face, dim=0)
    return bc_face


def pde_residual(coords, model):
    coords.requires_grad = True

    domain_min = torch.tensor([0.0, 0.0, 0.0], device=coords.device)
    domain_max = torch.tensor([24.0, 24.0, 24.0], device=coords.device)

    coords_spatial = coords[:, :3]
    coords_min = torch.zeros_like(coords_spatial)
    coords_max = torch.full_like(coords_spatial, 24.0)

    coords_spatial_norm = (coords_spatial - coords_min) / (coords_max - coords_min)

    # Time coordinate: fixed at 4/60
    t_fixed = coords[:, -1:]  # keep as is

    # Concatenate normalized spatial coords and time
    coords_norm = torch.cat([coords_spatial_norm, t_fixed], dim=1)


    c = model(coords_norm)

    grads = torch.autograd.grad(
        c, coords_norm,
        grad_outputs=torch.ones_like(c),
        create_graph=True
    )[0]  # shape [batch_size, 4]

    L = domain_max - domain_min

    # Split gradients into components
    c_x = grads[:, 0] 
    c_y = grads[:, 1] 
    c_z = grads[:, 2] 
    c_t = grads[:, 3]

    # Second derivatives
    grads_x = torch.autograd.grad(
        c_x, coords_norm,
        grad_outputs=torch.ones_like(c_x),
        create_graph=True
    )[0]
    c_xx = grads_x[:, 0]

    grads_y = torch.autograd.grad(
        c_y, coords_norm,
        grad_outputs=torch.ones_like(c_y),
        create_graph=True
    )[0]
    c_yy = grads_y[:, 1]

    grads_z = torch.autograd.grad(
        c_z, coords_norm,
        grad_outputs=torch.ones_like(c_z),
        create_graph=True
    )[0]
    c_zz = grads_z[:, 2]
   #
    pde_problem = PollutionPDE(
            num_cells=NUM_CELLS,
            diffusion_coef=0.1,
            convection_coef=(1.0,0.0,0.0)
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = torch.tensor(pde_problem.get_source().value).to(device)

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


    laplacian = c_xx + c_yy + c_zz

    Ux, Uy, Uz = [-1.0, 0.0, 0.0]

    convection = Ux * c_x + Uy * c_y + Uz * c_z


    idx = 0
    print(f"Sum of laplacian {laplacian}")
    print(f"source {source[idx]}")
    print(f"C-wrt[{idx}] {c_t[idx]}")
    print(f"C-x[{idx}] {c_x[idx]}")
    print(f"C-y[{idx}] {c_y[idx]}")
    print(f"C-z[{idx}] {c_z[idx]}")
    

    print(f"Residual with right coef at idx({idx})= {c_t[idx]} - {0.6*laplacian[idx]} + {c_x[idx]} + {source[idx]} = {torch.sum(c_t[idx] - 0.6*laplacian[idx] + c_x[idx] + source[idx])}")
    print(f"Residual with right coef = {c_t} - {0.6*laplacian} + {c_x} + {source} = {torch.sum(c_t - 0.6*laplacian + c_x + source)}")



    # PDE residual:
    residual = c_t - (model.delta.abs() * laplacian - convection + source)
    # residual = (c_t / source) - \
    #            (model.delta.abs() * laplacian - (convection / source) + 1.0)

    # residual = c_t - ((model.delta.abs() * (c_xx + c_yy + c_zz))) + ((1*c_x)) - source #(2 * model.source_region_param.abs())

    print(f"Residual error: {torch.sum(residual)}")

    return torch.mean(residual**2)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name()) 

    all_values = np.load("tall-data.npy")
    training_data = all_values[2:10,:]
    print(training_data)
    c_min = np.min(training_data)
    c_max = np.max(training_data)

    data = []
    # all_values[4,:]

    for step, _ in enumerate(training_data):
        for idx, value in enumerate(training_data[step]):
            z = idx // (NUM_CELLS ** 2)
            y = (idx % (NUM_CELLS ** 2)) // NUM_CELLS
            x = idx % NUM_CELLS

            physical_c = value
            normalized_c = normalize_c(physical_c, c_min, c_max) # Use the normalized value

            data.append(((x, y, z, step * 1/60), normalized_c))
        print(step * 1/60)

    for d in data:
        print(d)

    dataset = PollutionDataset(data)
    loader = DataLoader(dataset, batch_size=15625, shuffle=True)


    model = PINN().to(device)
    # Optimizer for the neural network weights with a smaller learning rate
    optimizer_net = optim.Adam(model.hidden.parameters(), lr=1e-3)
    # Optimizer specifically for the delta parameter with a LARGER learning rate
    optimizer_delta = optim.Adam([model.delta], lr=1e-3)

    # Hyperparameters
    num_epochs = 100000


    print("Starting training...")
    start_time = time.time()

    pde_loss_over_time = []
    data_loss_over_time = []
    delta_values_over_time = []

    best_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            
            epoch_loss = 0.0
            pde_loss_total = 0.0
            data_loss_total = 0.0

            for coords, c in loader:
                coords = coords.to(device)

                c_pred_left = model(generate_bc_inputs(coords, 0, 0))
                c_pred_right = model(generate_bc_inputs(coords, 0, 24))

                c_pred_bottom = model(generate_bc_inputs(coords, 1, 0))
                c_pred_top = model(generate_bc_inputs(coords, 1, 24))

                c_pred_front = model(generate_bc_inputs(coords, 2, 0))
                c_pred_back = model(generate_bc_inputs(coords, 2, 24))


                loss_bc = torch.mean((c_pred_left - torch.full_like(c_pred_left, 0))**2) + \
                            torch.mean((c_pred_right - torch.full_like(c_pred_right, 0))**2) + \
                            torch.mean((c_pred_bottom - torch.full_like(c_pred_bottom, 0))**2) + \
                            torch.mean((c_pred_top - torch.full_like(c_pred_top, 0))**2) + \
                            torch.mean((c_pred_front - torch.full_like(c_pred_front, 0))**2) + \
                            torch.mean((c_pred_back - torch.full_like(c_pred_back, 0))**2)

                c = c.to(device)
                optimizer_net.zero_grad()

                optimizer_delta.zero_grad()
                c_pred = model(coords)  # coords shape: [batch_size, 3]

                data_loss = torch.mean((c_pred - c) ** 2) + loss_bc


                if(epoch % 50 == 0):
                    pde_loss = pde_residual(coords, model, )
                    loss = data_loss + pde_loss
                    loss.backward()
                    optimizer_delta.step()
                    optimizer_net.step()

                    pde_loss_total += pde_loss
                    pde_loss_over_time.append(pde_loss_total)

                else:
                    loss = data_loss
                    loss.backward()
                    optimizer_net.step()

                epoch_loss += loss.item() 

                data_loss_total += data_loss

            data_loss_over_time.append(data_loss_total)
            delta_values_over_time.append(model.delta.item())
               

            epoch_end_time = time.time()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Loss Pde: {pde_loss_total:.6f} | "
                  f"Loss Data: {data_loss_total:.6f} | "
                f"Epoch time (s): {(epoch_end_time - epoch_start_time):.1f} | "
                  f"Time Left: {(((epoch_end_time - epoch_start_time) * (num_epochs - epoch))/3600):2f} hours | "
                  f"Delta: {model.delta.item():.6f} | ")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')
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
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Delta Value')
        axs[1].set_title('Delta Parameter Over Time')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
