import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

def kmeans_hotspots_from_residual(
    R, K=10, sigma=1.0, tau=None, top_pct=10, eps=1e-12, rng=0,
    ignore_mask=None
):
    """
    R: (H,W) residuals >=0
    ignore_mask: (H,W) bool, True = EXCLUDE (trained areas)
    returns peaks_ij (K',2) [row,col], w (K',)
    """
    H, W = R.shape
    Rf = gaussian_filter(R, sigma=sigma)

    if ignore_mask is None:
        ignore_mask = np.zeros_like(R, dtype=bool)

    # threshold on unignored pixels only
    r_all = Rf[~ignore_mask].ravel()
    if r_all.size == 0:
        raise ValueError("All pixels are ignored by ignore_mask.")
    if tau is None:
        thr = np.percentile(r_all, 100 - top_pct)
        base_keep = Rf > thr
    else:
        base_keep = Rf > tau

    # final keep mask: high residual AND not ignored
    keep = base_keep & (~ignore_mask)
    if not np.any(keep):
        raise ValueError("No pixels left after masking/thresholds.")


# coords and weights
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Xs = np.stack([ii[keep], jj[keep]], axis=1).astype(float)  # [i=row, j=col]
    ws = Rf[keep].astype(float)

    K_eff = min(K, max(1, Xs.shape[0]))

    km = KMeans(n_clusters=K_eff, init="k-means++", n_init=30, random_state=rng)
    km.fit(Xs, sample_weight=ws)
    centers_ij = km.cluster_centers_           # already [i, j]
    labels = km.predict(Xs)
    z = np.bincount(labels, weights=ws, minlength=K_eff)
    w = z / (z.sum() + eps)

    peaks_ij = centers_ij                       # DO NOT swap to [j, i]
    return peaks_ij, w

# --- usage ---
res = np.load("residual.npy")
res = res.reshape((100,100), order='F')

ignore_mask = np.zeros_like(res, dtype=bool)
ignore_mask[abs(100-66):100, 0:71] = True  # your trained region [rows, cols]

hotspots_ij, w = kmeans_hotspots_from_residual(
    res, K=15, sigma=1.0, top_pct=10, ignore_mask=ignore_mask
)
hotspots_ij_clipped = []
w_clipped = []

#clip/ignore hotspots on edges
mask = (
    (hotspots_ij[:,0] >= 5) &
    (hotspots_ij[:,0] <= 95) &
    (hotspots_ij[:,1] >= 5) &
    (hotspots_ij[:,1] <= 95)
)

hotspots_ij_clipped = hotspots_ij[mask]
w_clipped = w[mask]


num_hspots = w_clipped.shape[0]

start_pos = np.array([10,30])
diff = hotspots_ij_clipped[:, None, :] - hotspots_ij_clipped[None, :, :]
distance_matrix = np.linalg.norm(diff, axis=2)
print(distance_matrix)
distance_matrix[:, 0] = 0
permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
print(permutation)


hotspots_ij_clipped = hotspots_ij_clipped[permutation]
w_clipped = w_clipped[permutation]


h = 1 #timestep

#inital state and input
x0 = np.array([start_pos[0],start_pos[1],0,0])
u0 = np.array([0,0])

#control/state penalty
R = np.eye(2) * 1e-1
Q = np.eye(2) * 1e-2


C = np.array([[1,0,0,0],
              [0,1,0,0]])
#timehorizon
T = num_hspots * 50

w_vec = np.zeros((num_hspots, T))

for i in range(num_hspots):
    w_vec[i,(int(T/num_hspots))*(i):(int(T/num_hspots))*(i+1)] = 1.0

#Duble integrator matrices
#--------------------------------------------------
A = np.eye(4)
A[0,2] = h
A[1,3] = h


B = np.zeros((4,2))
accel_integrated = (h**2)/2
B[0,0] = accel_integrated
B[1,1] = accel_integrated
B[2,0] = h
B[3,1] = h
#--------------------------------------------------

x = cp.Variable((4,T+1))
u = cp.Variable((2,T))

constraints = []
constraints.append(x[:,0] == x0)
for t in range(T):
    constraints.append(x[:,t+1] == A@x[:,t] + B@u[:,t]) #Double integrator dynamics constraints
    constraints.append(cp.abs(x[2:4,t]) <= 1.5) #Max velocity limits


#QP Formulation
cost = 0
for k in range(T):
    cost += cp.quad_form(u[:,k],R)
    for (ij, weight) in zip(hotspots_ij_clipped, w_vec[:,k]):
        i,j = ij
        e = C @ x[:,k] - np.array([j, i])
        cost += weight * cp.quad_form(e, Q)

prob = cp.Problem(cp.Minimize(cost),constraints)
prob.solve()

H, W = res.shape
traj = np.vstack([x.value[0, :], x.value[1, :]]).T
hot = np.asarray(hotspots_ij_clipped)


print(hotspots_ij_clipped)
cords_y = hotspots_ij_clipped[:,0]
cords_y = np.abs(100 - cords_y)
print(cords_y)
cords_x = hotspots_ij_clipped[:,1]

hotspot_cords = np.stack([cords_x, cords_y], axis=1)
print(hotspot_cords)

np.save("state_vec.npy", x.value)
np.save("control_vec.npy", u.value)

np.save("hotspots.npy", hotspot_cords)

fig, ax = plt.subplots(figsize=(8, 6))

# Show residual map normally
im = ax.imshow(np.abs(res), cmap="hot", origin="upper", interpolation="nearest")

# Overlay ignored region in semi-transparent gray
ignore_overlay = np.zeros((*ignore_mask.shape, 4))  # RGBA
ignore_overlay[..., 0:3] = 0.5  # gray color (R,G,B)
ignore_overlay[..., 3] = ignore_mask * 0.4  # alpha = 0.4 where ignored, 0 elsewhere
ax.imshow(ignore_overlay, origin="upper", interpolation="nearest", zorder=2)

# Add colorbar for residual magnitude
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("PDE Residual Magnitude", rotation=270, labelpad=15)



# Hotspots
ax.scatter(hot[:, 1], hot[:, 0], s=60, facecolors="none", edgecolors="cyan", linewidths=1.5, zorder=3)

# Trajectory already in [x,y]; keep as-is
ax.plot(traj[:, 0], traj[:, 1], linewidth=2, color="blue", zorder=4)
ax.scatter(traj[:, 0], traj[:, 1], s=8, alpha=0.6, color="blue", zorder=4)

# Start and end markers
ax.scatter(traj[0, 0], traj[0, 1], s=90, marker="o", edgecolors="k",
           facecolors="lime", zorder=5, label="Start")
ax.scatter(traj[-1, 0], traj[-1, 1], s=110, marker="X", edgecolors="k",
           facecolors="red", zorder=5, label="End")

# Direction arrows
d = np.diff(traj, axis=0)
ax.quiver(traj[:-1, 0], traj[:-1, 1], d[:, 0], d[:, 1],
          angles="xy", scale_units="xy", scale=1,
          width=0.003, alpha=0.35, color="blue", zorder=4)

# Axes formatting
ax.set_xlim(0, W - 1)
ax.set_ylim(H - 1, 0)  # origin="upper"
ax.set_aspect("equal")
ax.set_title("Residual Map with K-Means Hotspots, Planned Trajectory, and Ignored Region")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

control_inputs_ax = u.value[0, 1:]
control_inputs_ay = u.value[1, 1:]
timesteps = np.arange(T-1)

plt.figure(figsize=(8, 4))
plt.step(timesteps, control_inputs_ax, where="post", label=r"$a_x$", linewidth=2)
plt.step(timesteps, control_inputs_ay, where="post", label=r"$a_y$", linewidth=2)

plt.xlabel("Timestep")
plt.ylabel("Control input")
plt.title("Control Inputs Over Time (Step View)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

