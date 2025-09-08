
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("metrics/summary.csv")

# Order to match the table (M1..M5)
order = [
    "model_15000_no_closeloop",   # M1
    "model_1500_pre_closeloop",   # M2
    "model_1500_post_closeloop",  # M3
    "model_15000_post_closeloop",  # M4
    "model_1500_random_sample"  # M5
]
df = df.set_index("model_tag").loc[order].reset_index()

# Pretty labels to match Overleaf table
label_map = {
    "model_15000_no_closeloop":   "M1 (No-CL, 15k)",
    "model_1500_pre_closeloop":   "M2 (Pre-CL, 1.5k)",
    "model_1500_post_closeloop":  "M3 (Post-CL, 1.5k)",
    "model_15000_post_closeloop": "M4 (Post-CL, 15k)",
    "model_1500_random_sample": "M5 (Post-CL RS, 1.5k)",
}
xtick_labels = [label_map[t] for t in df["model_tag"]]

metrics = ["MAE", "RMSE", "R2"]
titles  = ["Mean Absolute Error ↓", "Root Mean Squared Error ↓", "R² Score ↑"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(df))
width = 0.6
colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdd835"]

for ax, metric, title in zip(axes, metrics, titles):
    vals = df[metric].values
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v * (1.01 if metric != "R2" else 1.002), f"{v:.5f}",
                ha="center", va="bottom", fontsize=9, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=15, ha="right")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylabel(metric)
    if metric == "R2":
        ax.set_ylim(0.0, 1.05)

plt.suptitle("Model Performance Comparison Across Metrics", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
