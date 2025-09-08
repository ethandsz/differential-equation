
from operator import index

import argparse
from fipy import residual
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
        (x, y, t), c = self.data[idx]
        return torch.tensor([x, y, t], dtype=torch.float32), torch.tensor(c, dtype=torch.float32)


class PINN_2D(nn.Module):
    def __init__(self):
        super(PINN_2D, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(3,64),
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
        self.source_center = nn.Parameter(torch.tensor([0.7, 0.1]))
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
    bc_face = torch.rand(NUM_CELLS * 100, 3).to(device)
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
    c_t = grads[:,2]

    grads_x = autograd.grad(c_x, coords, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_xx = grads_x[:,0]

    grads_y = autograd.grad(c_y, coords, grad_outputs=torch.ones_like(c_y), create_graph=True)[0]
    c_yy = grads_y[:,1]


    delta = model.delta.abs().squeeze()  

    vx = 10.0
    vy = 0.0

    source = torch.zeros_like(c_t)
    # print(coords.shape)
    x = coords[:, 0] 
    y = coords[:, 1]
    # print(torch.max(x_idxs))


    # GT Values
    # center_x = 0.1
    # center_y = 0.5
    # radius = 0.05  # adjust as needed

    # distance_squared = (x - model.source_center[0])**2 + (y - model.source_center[1])**2
    # source_region = distance_squared < model.source_radius**2
    #
    # source[source_region] = 20
    # print(source.shape)

    # dist_sq = (x - model.source_center[0])**2 + (y - model.source_center[1])**2
    dist_sq = (x - model.source_center[0])**2 + (y - model.source_center[1])**2
    scale = 1000.0
    source = 20 * torch.sigmoid(scale * (model.source_radius**2 - dist_sq))
    # source = 20 * (dist_sq < model.source_radius**2).float()

    residual = c_t + ((vx * c_x) + (vy * c_y)) - delta * (c_xx + c_yy) - source

    return torch.mean(residual**2), residual


# grid/time scales

def hotspot_cell_indices(hotspots_ij, radius, H, W):
    """Return unique [i,j] cells within a disk of given radius around each hotspot."""
    buckets = []
    for i, j in hotspots_ij.astype(int):
        i0 = max(0, i - radius); i1 = min(H - 1, i + radius)
        j0 = max(0, j - radius); j1 = min(W - 1, j + radius)
        ii, jj = np.mgrid[i0:i1+1, j0:j1+1]
        mask = (ii - i)**2 + (jj - j)**2 <= radius**2
        buckets.append(np.stack([ii[mask], jj[mask]], axis=1))
    if not buckets:
        return np.empty((0,2), dtype=int)
    idx = np.vstack(buckets)
    idx = np.unique(idx, axis=0)  # deduplicate overlaps
    return idx  # shape (M,2): [row=i, col=j]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":


    parser = argparse.ArgumentParser(prog='2D IPINN')
    parser.add_argument('--enableHotspots', type=str2bool, nargs='?', const=True, default=True,help='Closed Loop training with hotspots')
    args = parser.parse_args()
    enable_hotspots = args.enableHotspots
    print(enable_hotspots)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name()) 

    start_idx = 0
    end_idx = 50
    dt = 1/60
    all_values = np.load("2d-data.npy")
    all_values = all_values[start_idx:end_idx,:]
    training_data = []


    x_min_training_points = 0
    x_max_training_points = 70

    y_min_training_points = 0
    y_max_training_points = 65

    view_data = False

    for timestep, gt_data in enumerate(all_values):
        data_2d = gt_data.reshape(NUM_CELLS, NUM_CELLS)
        solution_points = data_2d[y_min_training_points:y_max_training_points, x_min_training_points:x_max_training_points]
        if(view_data):
            x = np.arange(x_min_training_points, x_max_training_points, 1)
            y = np.arange(y_min_training_points, y_max_training_points, 1)
            X, Y = np.meshgrid(x, y)
            plt.figure(figsize=(6, 5))
            cp = plt.contourf(X, Y, solution_points, cmap='viridis')
            plt.colorbar(cp, label='Concentration')
            plt.title(f'Advection–Diffusion solution at t={timestep}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        training_data.append([timestep, solution_points])
        # training_data[timestep, :] = data_2d[0:100, 0:50].flatten()

        
    # training_data = all_values[start_idx:end_idx,:]
    training_points = [item[1] for item in training_data]
    training_points = np.array(training_points)
    print(training_points.shape)

    t_min = start_idx * dt
    t_max = end_idx * dt

    hotspot_idxs = None
    hotspot_data = None
    if enable_hotspots:
        hotspots = np.load("hotspots.npy")
        hotspot_idxs = hotspot_cell_indices(hotspots, radius=0, H=NUM_CELLS, W=NUM_CELLS)

        num_points_in_hotspots = hotspot_idxs.shape[0]
        hotspot_data = []
        for t in range(end_idx):
            t0 = all_values[t,:].reshape(NUM_CELLS,NUM_CELLS)
            hotspot_data_t0 = t0[hotspot_idxs[:,0], hotspot_idxs[:,1]]
            hotspot_data.append(hotspot_data_t0)

        hotspot_data = np.array(hotspot_data)

        c_min_hotspots = np.min(hotspot_data)
        c_max_hotspots = np.max(hotspot_data)

# after building hotspot_data (shape: [T, M])
    c_min_train = np.min(training_points)
    c_max_train = np.max(training_points)


    if(enable_hotspots):
        c_min = min(c_min_train, c_min_hotspots)
        c_max = max(c_max_train, c_max_hotspots)
    else:
        c_min = c_min_train
        c_max = c_max_train

    print(x_min_training_points)
    print(y_min_training_points)

    data = []

    for timestep, data_2d in training_data:
        timestep_real = timestep + start_idx  # because we used enumerate on sliced data
        t = timestep_real * dt
        t_norm = (t - t_min) / (t_max - t_min)

        if(enable_hotspots):
            for idx, h_data in enumerate(hotspot_data[timestep,:]):
                x, y = hotspot_idxs[idx]
                print(f"x: {x}")
                print(f"y: {y}")
                print(f"data: {h_data}")
                x_norm = (x + x_min_training_points) / (NUM_CELLS - 1)
                y_norm = (y + y_min_training_points) / (NUM_CELLS - 1)
                c_norm = (h_data - c_min) / (c_max - c_min)
                print(f"data_norm: {c_norm}")
                data.append(((x_norm, y_norm, t_norm), c_norm))



        ny, nx = data_2d.shape
        k = 1010# number of random points per timestep

        total_points = ny * nx
        flat_indices = np.random.choice(total_points, k, replace=False)
        ys, xs = np.unravel_index(flat_indices, (ny, nx))

        for iy, ix in zip(ys, xs):
            x_norm = (ix + x_min_training_points) / (NUM_CELLS - 1)
            y_norm = (iy + y_min_training_points) / (NUM_CELLS - 1)

            c = data_2d[iy, ix]
            c_norm = (c - c_min) / (c_max - c_min)
            print(f"nonhspot data norm: {c_norm}")

            data.append(((x_norm, y_norm, t_norm), c_norm))


    dataset = PollutionDataset(data)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)


    model = PINN_2D().to(device)

    #--------------------------------------------------NEW--------------------------------------------------
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     if device.type == 'cuda':
#         print(torch.cuda.get_device_name())
#
# # --- build model and move to device ---
#     model = PINN_2D().to(device)
#
# # single optimizer with param groups so it can be checkpointed
#     optimizer = optim.Adam([
#         {'params': model.hidden.parameters(), 'lr': 1e-3},
#         {'params': [model.delta, model.source_center, model.source_radius], 'lr': 2e-3},
#     ])
#
# # --- load checkpoint (handles both formats) ---
#     ckpt_path = "best_model_2d_hotspots_15000.pth"
#     ckpt = torch.load(ckpt_path, map_location=device)
#
#     start_epoch = 0
#     if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
#         model.load_state_dict(ckpt['model_state_dict'])
#         if 'optimizer_state_dict' in ckpt:
#             try:
#                 optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#             except Exception as e:
#                 print(f"Optimizer state not loaded: {e}")
#         start_epoch = ckpt.get('epoch', 0) + 1
#     else:
#         # file is a plain state_dict()
#         model.load_state_dict(ckpt)
#
#     model.train()
    #--------------------------------------------------NEW--------------------------------------------------

    # Optimizer for the neural network weights with a smaller learning rate
    optimizer_net = optim.Adam(model.hidden.parameters(), lr=1e-3)
    # Optimizer specifically for the delta parameter with a LARGER learning rate
    optimizer_pde_coefs = optim.Adam([model.delta, model.source_center, model.source_radius], lr=2e-3)

    # Hyperparameters
    num_epochs = 1501


    print("Starting training...")
    start_time = time.time()

    pde_loss_over_time = []
    data_loss_over_time = []
    delta_values_over_time = []
    source_center_x_values_over_time = []
    source_center_y_values_over_time = []
    source_center_radius_values_over_time = []

    best_loss = float('inf')
    # quit()

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            
            epoch_loss = 0.0
            pde_loss_total = 0.0
            data_loss_total = 0.0

            loss_bc_total = 0.0
            loss_ic_total = 0.0

            residual_tensors = []
            for coords, c in loader:
                coords = coords.to(device)

                initial_conditions = torch.rand(NUM_CELLS * 100, 3).to(device)
                initial_conditions[:,2] = 0
                c_pred_ic = model(initial_conditions)
                loss_ic = torch.mean(c_pred_ic ** 2)

                x_idx = 0
                y_idx = 1

                min_bc_position = 0
                max_bc_position = 1

                bc_left = generate_bc_inputs(x_idx, min_bc_position, device)
                bc_right = generate_bc_inputs(x_idx, max_bc_position, device)

                bc_down = generate_bc_inputs(y_idx, min_bc_position, device)
                bc_up = generate_bc_inputs(y_idx, max_bc_position, device)

                c_pred_left = model(bc_left)
                c_pred_right = model(bc_right)
                c_pred_down = model(bc_down)
                c_pred_up = model(bc_up)

                loss_bc = torch.mean((c_pred_left - torch.full_like(c_pred_left, 0))**2) + \
                            torch.mean((c_pred_right - torch.full_like(c_pred_right, 0))**2)  + \
                            torch.mean((c_pred_down - torch.full_like(c_pred_down, 0))**2)  + \
                            torch.mean((c_pred_up - torch.full_like(c_pred_up, 0))**2)

                c = c.to(device)
                optimizer_net.zero_grad()

                optimizer_pde_coefs.zero_grad()
                c_pred = model(coords)  # coords shape: [batch_size, 3]

                data_loss = torch.mean((c_pred - c) ** 2) 
                
                N_collocation = 10000
                x_colloc = torch.linspace(0,1,NUM_CELLS).unsqueeze(1).repeat(1,NUM_CELLS).flatten().unsqueeze(1).to(device)
                y_colloc = torch.linspace(0,1,NUM_CELLS).repeat(NUM_CELLS).unsqueeze(1).to(device)
                t_colloc = torch.rand(x_colloc.shape[0],1).to(device)
                coords_colloc = torch.cat([x_colloc, y_colloc, t_colloc], dim=1)

                pde_loss, residual_tensor = pde_residual(coords_colloc, model, c_min, c_max)

                residual_tensors.append(residual_tensor)

                lambda_penalty = 1000
                penalty_source_x = torch.relu(0.05 - model.source_center[0]).sum()
                penalty_source_y = torch.relu(0.05 - model.source_center[1]).sum()
                penalty_source_radius = torch.relu(0.01 - model.source_radius).sum()
                penalty_diffusion = torch.relu(0.1 - model.delta).sum()
                param_penalities = lambda_penalty * (penalty_source_x + penalty_source_y + penalty_source_radius + penalty_diffusion)

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


            avg = torch.stack(residual_tensors, dim=0).mean(dim=0)  # shape: (100, 100)
            res = avg.detach().cpu().numpy()  # shape (100, 100)
            if (not enable_hotspots):
                np.save("residual.npy", res)

            pde_loss_over_time.append(pde_loss_total)

            data_loss_over_time.append(data_loss_total)
            delta_values_over_time.append(model.delta.item())
            source_center_x_values_over_time.append(model.source_center[0].item())
            source_center_y_values_over_time.append(model.source_center[1].item())
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
                  f"Source radius: {model.source_radius.item():.6f} | ")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if(enable_hotspots):
                    torch.save(model.state_dict(), f'best_model_2d_hotspots_{num_epochs}.pth')
                else:
                    torch.save(model.state_dict(), f'best_model_2d_{num_epochs}e-10.pth')
                print(f"Best model saved at epoch {epoch+1} with val loss {epoch_loss:.4f}")

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("Training killed")

    finally:
# Convert tensors to floats if they are still tensors
        pde_loss_over_time = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in pde_loss_over_time]
        data_loss_over_time = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in data_loss_over_time]

        if(enable_hotspots):
            np.save(f"pde_loss_hotspots_enabled_{num_epochs}.npy", pde_loss_over_time)
            np.save(f"data_loss_hotspots_enabled_{num_epochs}.npy", data_loss_over_time)
            np.save(f"delta_values_hotspots_enabled_{num_epochs}.npy", delta_values_over_time)
            np.save(f"source_center_x_hotspots_enabled_{num_epochs}.npy", source_center_x_values_over_time)
            np.save(f"source_center_y_hotspots_enabled_{num_epochs}.npy", source_center_y_values_over_time)
            np.save(f"source_center_radius_hotspots_enabled_{num_epochs}.npy", source_center_radius_values_over_time)

        else:
            np.save(f"pde_loss_{num_epochs}e-10.npy", pde_loss_over_time)
            np.save(f"data_loss_{num_epochs}e-10.npy", data_loss_over_time)
            np.save(f"delta_values_{num_epochs}e-10.npy", delta_values_over_time)
            np.save(f"source_center_x_{num_epochs}e-10.npy", source_center_x_values_over_time)
            np.save(f"source_center_y_{num_epochs}e-10.npy", source_center_y_values_over_time)
            np.save(f"source_center_radius_{num_epochs}e-10.npy", source_center_radius_values_over_time)
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
        axs[1].plot(source_center_radius_values_over_time, label='Source Center Radius Parameter', color='tab:gray')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Model Parameter Values')
        axs[1].set_title('Parameter Values Over Time')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
