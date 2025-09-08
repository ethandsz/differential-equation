
import os
import numpy as np
import matplotlib.pyplot as plt

# --- edit these ---
DIR = os.getcwd()
EPOCHS = 1501
#1 Hotspots enabled 
#0 No hotspots 
#4 Radom Sample
MODEL_ENUM = 4  # True -> *_hotspots_enabled_{EPOCHS}.npy, F3lse -> *_{EPOCHS}e.npy
# -------------------

def fp(name):
    return os.path.join(DIR, name)

if MODEL_ENUM == 1:
    pde_fn   = fp(f"metrics/hotspots{EPOCHS}/pde_loss_hotspots_enabled.npy")
    data_fn  = fp(f"metrics/hotspots{EPOCHS}/data_loss_hotspots_enabled.npy")
    delta_fn = fp(f"metrics/hotspots{EPOCHS}/delta_values_hotspots_enabled.npy")
    x_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_x_hotspots_enabled.npy")
    y_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_y_hotspots_enabled.npy")
    r_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_radius_hotspots_enabled.npy")
elif MODEL_ENUM == 0:
    pde_fn   = fp(f"metrics/{EPOCHS}/pde_loss_{EPOCHS}e.npy")
    data_fn  = fp(f"metrics/{EPOCHS}/data_loss_{EPOCHS}e.npy")
    delta_fn = fp(f"metrics/{EPOCHS}/delta_values_{EPOCHS}e.npy")
    x_fn     = fp(f"metrics/{EPOCHS}/source_center_x_{EPOCHS}e.npy")
    y_fn     = fp(f"metrics/{EPOCHS}/source_center_y_{EPOCHS}e.npy")
    r_fn     = fp(f"metrics/{EPOCHS}/source_center_radius_{EPOCHS}e.npy")
elif MODEL_ENUM == 3:
    pde_fn   = fp(f"metrics/hotspots{EPOCHS}/pde_loss.npy")
    data_fn  = fp(f"metrics/hotspots{EPOCHS}/data_loss.npy")
    delta_fn = fp(f"metrics/hotspots{EPOCHS}/delta_values.npy")
    x_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_x.npy")
    y_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_y.npy")
    r_fn     = fp(f"metrics/hotspots{EPOCHS}/source_center_radius.npy")
else:
    print("Random Sample Selected")
    pde_fn   = fp(f"metrics/random{EPOCHS}/pde_loss_{EPOCHS}e-10.npy")
    data_fn  = fp(f"metrics/random{EPOCHS}/data_loss_{EPOCHS}e-10.npy")
    delta_fn = fp(f"metrics/random{EPOCHS}/delta_values_{EPOCHS}e-10.npy")
    x_fn     = fp(f"metrics/random{EPOCHS}/source_center_x_{EPOCHS}e-10.npy")
    y_fn     = fp(f"metrics/random{EPOCHS}/source_center_y_{EPOCHS}e-10.npy")
    r_fn     = fp(f"metrics/random{EPOCHS}/source_center_radius_{EPOCHS}e-10.npy")


# load arrays
pde_loss  = np.load(pde_fn, allow_pickle=True).ravel()
data_loss = np.load(data_fn, allow_pickle=True).ravel()
delta     = np.load(delta_fn, allow_pickle=True).ravel()
src_x     = np.load(x_fn, allow_pickle=True).ravel()
src_y     = np.load(y_fn, allow_pickle=True).ravel()
src_r     = np.load(r_fn, allow_pickle=True).ravel()

def trim_pair(a, b):
    L = min(len(a), len(b))
    L = min(L, 15000)
    return a[:L], b[:L]

pde_loss, data_loss = trim_pair(pde_loss, data_loss)
src_x, src_y = trim_pair(src_x, src_y)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
(ax11, ax12), (ax21, ax22) = axs

# 1) Losses together
eps = 1e-12
ax11.plot(np.clip(pde_loss,  eps, None), label="PDE Loss")
ax11.plot(np.clip(data_loss, eps, None), label="Data Loss")
ax11.set_yscale("log")
ax11.set_title("Training Losses Over Time")
ax11.set_xlabel("Epoch")
ax11.set_ylabel("Loss")
ax11.legend()
ax11.grid(True)

# 2) Delta alone
ax12.plot(delta)
ax12.set_title("Delta Parameter Over Time")
ax12.set_xlabel("Epoch")
ax12.set_ylabel("Delta")
ax12.grid(True)

# 3) Source radius alone
ax21.plot(src_r)
ax21.set_title("Source Center Radius Over Time")
ax21.set_xlabel("Epoch")
ax21.set_ylabel("Radius")
ax21.grid(True)

# 4) Source center X and Y together
ax22.plot(src_x, label="Source Center X")
ax22.plot(src_y, label="Source Center Y")
ax22.set_title("Source Center (X,Y) Over Time")
ax22.set_xlabel("Epoch")
ax22.set_ylabel("Value")
ax22.legend()
ax22.grid(True)

plt.tight_layout()
plt.show()
