""" Plots all test samples with given labels from training data. """

# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

test_dir = "DCEC_Testing_Output"

# Sort test files based on sample number (1 to 12)
test_files = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.startswith("Test_Sample") and f.endswith(".csv")]


def extract_sample_number(filepath):
    basename = os.path.basename(filepath)
    parts = basename.replace(".csv", "").split("_")
    sample_num = int(parts[1].replace("Sample", ""))
    return sample_num


test_files = sorted(test_files, key=extract_sample_number)

# Loads cluster labels
with open("Cluster_Label_Mappings.json", "r") as f:
    cluster_label_map = json.load(f)

cluster_colors = {
    "Matrix Cracking": 'blue',
    "Fiber/Matrix Debonding": 'green',
    "Fiber Breakage": 'red',
    "Delamination": 'orange',
    "Mixed": 'purple'}

fig, axs = plt.subplots(4, 3, figsize=(18, 24))
axs = axs.flatten()

for i, test_file in enumerate(test_files):
    df = pd.read_csv(test_file)
    df = df.sort_values("Time")

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
            0.1, 0.9, f"Sample {test_sample}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative AE Hits")
    ax.grid(True)

# Shared legend below the plots
handles = [plt.Line2D([0], [0], color=color, label=label) for label, color in cluster_colors.items()]
fig.legend(handles, cluster_colors.keys(), loc='lower center', ncol=5, fontsize='large', frameon=False)

import matplotlib.ticker as ticker

ax = plt.gca()  # get current axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # force scientific notation for all ranges
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

for ax in axs:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("DCEC final test results.png")
plt.show()
