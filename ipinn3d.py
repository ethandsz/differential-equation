

from operator import index

from torch.cuda import device_count
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

from PollutionPDE3D import PollutionPDE3D
from torch.utils.data import Dataset, DataLoader


NUM_TIMESTEPS = 300
NUM_CELLS = 100

class PollutionDataset(Dataset):
    def __init__(self, data):
        # data: list of ((x, y, z), c)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (x, y, z, t), c = self.data[idx]
        return torch.tensor([x, y, z, t], dtype=torch.float32), torch.tensor(c, dtype=torch.float32)


class PINN_3D(nn.Module):
    def __init__(self):
        super(PINN_3D, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(4,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,1),
        )
        self.delta = nn.Parameter(torch.tensor([[0.1]]))
        self.source_center = nn.Parameter(torch.tensor([0.7, 0.1, 0.3]))
        self.source_radius = nn.Parameter(torch.tensor(0.05))

    def forward(self, coords):
        # print("Forwarding ", coords)

        return self.hidden(coords).squeeze(-1)  # shape [batch_size]

def normalize_c(c_val, c_min, c_max):
    return (c_val - c_min) / (c_max - c_min)

#function to de-normalize (to check physical predictions later) - used in evalpinn.py
def denormalize_c(c_norm, c_min, c_max):
    return c_norm * (c_max - c_min) + c_min


def generate_bc_inputs(axis_idx, value, device):
    bc_face = torch.rand(NUM_CELLS * 100, 4).to(device)
    bc_face[:,axis_idx] = value
    return bc_face

def pde_residual(coords, model, c_min, c_max):
    coords.requires_grad=True
    c_norm = model(coords)
    c_physical = c_norm * (c_max - c_min) + c_min


    data_size = coords.shape[0]
    scaling_factor = data_size/NUM_CELLS**2

    grads = autograd.grad(c_physical, coords, grad_outputs=torch.ones_like(c_physical), create_graph=True)[0]
    c_x = grads[:,0]
    c_y = grads[:,1]
    c_z = grads[:,2]
    c_t = grads[:,3]

    grads_x = autograd.grad(c_x, coords, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_xx = grads_x[:,0]

    grads_y = autograd.grad(c_y, coords, grad_outputs=torch.ones_like(c_y), create_graph=True)[0]
    c_yy = grads_y[:,1]


    grads_z = autograd.grad(c_z, coords, grad_outputs=torch.ones_like(c_z), create_graph=True)[0]
    c_zz = grads_z[:,2]



    delta = model.delta.abs().squeeze()  

    vx = 10.0
    vy = 0.0
    vz = 0.0

    source = torch.zeros_like(c_t)
    # print(coords.shape)
    z = coords[:, 0] 
    y = coords[:, 1]
    x = coords[:, 2]

    #GT Values
    # center_x = 0.1
    # center_y = 0.5
    # center_z = 0.5
    # radius = 0.05  # adjust as needed



    dist_sq = (x - model.source_center[0])**2 + (y - model.source_center[1])**2 + (z - model.source_center[2]) ** 2
    scale = 1000.0
    source = 20 * torch.sigmoid(scale * (model.source_radius**2 - dist_sq))

    #
    # distance_squared = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2
    # source_region = distance_squared < radius**2
    #
    # source[source_region] = 20
    # print(source.shape)

    residual = c_t + ((vx * c_x) + (vy * c_y) + (vz * c_z)) - delta * (c_xx + c_yy + c_zz) - source
    return torch.mean(residual**2)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name()) 

    start_idx = 0
    end_idx = 15
    dt = 1/60
    all_values = np.load("3d-data.npy")[0:end_idx, :]
    print(all_values.shape)
    training_data = []


    x_min_training_points = 0
    x_max_training_points = 100

    y_min_training_points = 30
    y_max_training_points = 65


    z_min_training_points = 30
    z_max_training_points = 60

    view_data = False
    for timestep, gt_data in enumerate(all_values):

        data_3d = gt_data.reshape((NUM_CELLS, NUM_CELLS, NUM_CELLS), order='F')
        solution_points = data_3d[x_min_training_points:x_max_training_points, y_min_training_points:y_max_training_points, z_min_training_points:z_max_training_points]
        training_data.append([timestep, solution_points])
        # training_data[timestep, :] = data_3d[0:100, 0:50].flatten()
        
    # training_data = all_values[start_idx:end_idx,:]
    training_points = [item[1] for item in training_data]
    training_points = np.array(training_points)
    c_min = np.min(training_points)
    c_max = np.max(training_points)

    t_min = start_idx * dt
    t_max = end_idx * dt

    data = []

    for timestep, data_3d in training_data:
        timestep_real = timestep + start_idx  # because we used enumerate on sliced data
        t = timestep_real * dt
        t_norm = (t - t_min) / (t_max - t_min)



        nz, ny, nx = data_3d.shape
        k = 10000# number of random points per timestep

        total_points = nz * ny * nx
        flat_indices = np.random.choice(total_points, k, replace=False)
        zs, ys, xs = np.unravel_index(flat_indices, (nz, ny, nx))

        for iz, iy, ix in zip(zs, ys, xs):
            x_norm = (ix + x_min_training_points) / (NUM_CELLS - 1)
            y_norm = (iy + y_min_training_points) / (NUM_CELLS - 1)
            z_norm = (iz + z_min_training_points) / (NUM_CELLS - 1)

            c = data_3d[iz, iy, ix]
            c_norm = (c - c_min) / (c_max - c_min)

            data.append(((x_norm, y_norm, z_norm, t_norm), c_norm))



        #
        # for ix in range(nx):
        #     for iy in range(ny):
        #         for iz in range(nz):
        #             # normalize spatial coordinates
        #             x_norm = (ix + x_min_training_points) / (NUM_CELLS - 1)
        #             y_norm = (iy + y_min_training_points) / (NUM_CELLS - 1)
        #             z_norm = (iz + z_min_training_points) / (NUM_CELLS - 1)
        #
        #             c = data_3d[ix, iy, iz]
        #             c_norm = (c - c_min) / (c_max - c_min)
        #
        #             data.append(((x_norm, y_norm, z_norm, t_norm), c_norm))

    for d in data:
        print(d)

    dataset = PollutionDataset(data)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)


    model = PINN_3D().to(device)

    # Optimizer for the neural network weights with a smaller learning rate
    optimizer_net = optim.Adam(model.hidden.parameters(), lr=1e-3)
    # Optimizer specifically for the delta parameter with a LARGER learning rate
    optimizer_pde_coefs = optim.Adam([model.delta, model.source_center, model.source_radius], lr=2e-3)

    # Hyperparameters
    num_epochs = 22000


    print("Starting training...")
    start_time = time.time()

    pde_loss_over_time = []
    data_loss_over_time = []
    delta_values_over_time = []

    source_center_x_values_over_time = []
    source_center_y_values_over_time = []
    source_center_z_values_over_time = []
    source_center_radius_values_over_time = []

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

                initial_conditions = torch.rand(NUM_CELLS * 100, 4).to(device)
                initial_conditions[:,3] = 0
                c_pred_ic = model(initial_conditions)
                loss_ic = torch.mean(c_pred_ic ** 2)


                x_idx = 0
                y_idx = 1
                z_idx = 2

                min_bc_position = 0
                max_bc_position = 1

                bc_left = generate_bc_inputs(x_idx, min_bc_position, device)
                bc_right = generate_bc_inputs(x_idx, max_bc_position, device)


                bc_down = generate_bc_inputs(y_idx, min_bc_position, device)
                bc_up = generate_bc_inputs(y_idx, max_bc_position, device)


                bc_back = generate_bc_inputs(z_idx, min_bc_position, device)
                bc_front = generate_bc_inputs(z_idx, max_bc_position, device)

                c_pred_left = model(bc_left)
                c_pred_right = model(bc_right)
                c_pred_down = model(bc_down)
                c_pred_up = model(bc_up)
                c_pred_back = model(bc_back)
                c_pred_front = model(bc_front)


                loss_bc = torch.mean((c_pred_left - torch.full_like(c_pred_left, 0))**2) + \
                            torch.mean((c_pred_right - torch.full_like(c_pred_right, 0))**2)  + \
                            torch.mean((c_pred_down - torch.full_like(c_pred_down, 0))**2)  + \
                            torch.mean((c_pred_up - torch.full_like(c_pred_up, 0))**2) + \
                            torch.mean((c_pred_back - torch.full_like(c_pred_back, 0))**2) + \
                            torch.mean((c_pred_front - torch.full_like(c_pred_front, 0))**2)


                c = c.to(device)
                optimizer_net.zero_grad()

                optimizer_pde_coefs.zero_grad()
                c_pred = model(coords)  # coords shape: [batch_size, 3]

                data_loss = torch.mean((c_pred - c) ** 2) 
                
                N_collocation = 10000  # e.g. 10k instead of 1M

                x_colloc = torch.rand(N_collocation, 1, device=device)
                y_colloc = torch.rand(N_collocation, 1, device=device)
                z_colloc = torch.rand(N_collocation, 1, device=device)
                t_colloc = torch.rand(N_collocation, 1, device=device)

                coords_colloc = torch.cat([x_colloc, y_colloc, z_colloc, t_colloc], dim=1)

                pde_loss = pde_residual(coords_colloc, model, c_min, c_max)

                lambda_penalty = 1000
                penalty_source_x_lower = torch.relu(0.05 - model.source_center[0]).sum()
                penalty_source_x_upper = torch.relu(model.source_center[0] - 1.0).sum()

                penalty_source_y_lower = torch.relu(0.05 - model.source_center[1]).sum()
                penalty_source_y_upper = torch.relu(model.source_center[1] - 1.0).sum()

                penalty_source_z_lower = torch.relu(0.05 - model.source_center[2]).sum()
                penalty_source_z_upper = torch.relu(model.source_center[2] - 1.0).sum()

                penalty_source_radius_lower = torch.relu(0.01 - model.source_radius).sum()
                penalty_source_radius_upper = torch.relu(model.source_radius - 0.3).sum()

                penalty_diffusion_lower = torch.relu(0.1 - model.delta).sum()
                penalty_diffusion_upper = torch.relu(model.delta - 1.0).sum()

                param_penalities = lambda_penalty * (penalty_source_x_lower + penalty_source_x_upper + penalty_source_y_lower + penalty_source_y_upper + penalty_source_z_lower + penalty_source_z_upper + penalty_source_radius_lower + penalty_source_radius_upper + penalty_diffusion_lower + penalty_diffusion_upper)

                loss = (100 * data_loss) + (100 *loss_bc) + (100 * loss_ic) + pde_loss + param_penalities
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

            source_center_x_values_over_time.append(model.source_center[0].item())
            source_center_y_values_over_time.append(model.source_center[1].item())
            source_center_z_values_over_time.append(model.source_center[2].item())
            source_center_radius_values_over_time.append(model.source_radius.item())
               

            epoch_end_time = time.time()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Loss Pde: {pde_loss_total:.6f} | "
                  f"Loss Data: {data_loss_total:.6f} | "
                f"Epoch time (s): {(epoch_end_time - epoch_start_time):.1f} | "
                  f"Time Left: {(((epoch_end_time - epoch_start_time) * (num_epochs - epoch))/3600):2f} hours | "
                  f"Delta: {model.delta.item():.6f} | "
                  f"Source center X: {model.source_center[0].item():.6f} | "
                  f"Source center Y: {model.source_center[1].item():.6f} | "
                  f"Source center Z: {model.source_center[2].item():.6f} | "
                  f"Source radius: {model.source_radius.item():.6f} | ")

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

        np.save("pde_loss.npy", pde_loss_over_time)
        np.save("data_loss.npy", data_loss_over_time)
        np.save("delta_values.npy", delta_values_over_time)
        np.save("source_center_x.npy", source_center_x_values_over_time)
        np.save("source_center_y.npy", source_center_y_values_over_time)
        np.save("source_center_z.npy", source_center_z_values_over_time)
        np.save("source_center_radius.npy", source_center_radius_values_over_time)

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
        axs[1].plot(source_center_x_values_over_time, label='Source Center X Parameter', color='tab:red')
        axs[1].plot(source_center_y_values_over_time, label='Source Center Y Parameter', color='tab:blue')
        axs[1].plot(source_center_z_values_over_time, label='Source Center Z Parameter', color='tab:cyan')
        axs[1].plot(source_center_radius_values_over_time, label='Source Center Radius Parameter', color='tab:gray')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Delta & Nabla Values')
        axs[1].set_title('Delta & Nabla Parameters Over Time')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
