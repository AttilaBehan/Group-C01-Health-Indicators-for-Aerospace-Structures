""" Plots all test samples with given labels from training data. """

# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

test_dir = "DCEC_Testing_Output"
test_files = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.startswith("Test_Sample") and f.endswith(".csv")])

# Loads cluster labels
with open("Cluster_Label_Mappings.json", "r") as f:
    cluster_label_map = json.load(f)

cluster_colors = {
    "Matrix cracking": 'blue',
    "Fiber-Matrix debond": 'green',
    "Fiber breakage": 'red'}

fig, axs = plt.subplots(3, 4, figsize=(24, 18))
axs = axs.flatten()

for i, test_file in enumerate(test_files):
    df = pd.read_csv(test_file)
    df = df.sort_values("Time").iloc[:10000]

    basename = os.path.basename(test_file)
    parts = basename.replace(".csv", "").split("_")
    test_sample = parts[1].replace("Sample", "")
    val_sample = parts[2].replace("Val", "")

    key = f"Val_Sample{val_sample}_Test_Sample{test_sample}"
    if key not in cluster_label_map:
        print(f"Warning: {key} not found in cluster_label_map, skipping.")
        continue
    label_map = cluster_label_map[key]

    ax = axs[i]
    for cluster_id_str, cluster_label in label_map.items():
        cluster_id = int(cluster_id_str)
        cluster_data = df[df["Cluster"] == cluster_id]
        if cluster_data.empty:
            continue
        grouped = cluster_data.groupby("Time")["Counts"].sum().reset_index()
        grouped["Cumulative_Hits"] = grouped["Counts"].cumsum()

        ax.plot(
            grouped["Time"],
            grouped["Cumulative_Hits"],
            label=cluster_label,
            color=cluster_colors.get(cluster_label, 'black'))

        # Title inside the plot
        ax.text(
            0.05, 0.95, f"Sample {test_sample}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_title(f"Test Sample {test_sample} (Val Sample {val_sample})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative AE Hits")
    ax.grid(True)

# Shared legend below the plots
handles = [plt.Line2D([0], [0], color=color, label=label) for label, color in cluster_colors.items()]
fig.legend(handles, cluster_colors.keys(), loc='lower center', ncol=3, fontsize='large', frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])    # Leave space at bottom for legend
plt.show()
