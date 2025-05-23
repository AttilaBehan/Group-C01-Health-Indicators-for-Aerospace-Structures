""" Plots to identify damage modes on training datasets.
    Given labels are inputted into Cluster_Label_Mappings.json """

# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import re


def plot_training_run(val_sample, test_sample, base_dir="DCEC_Training_Output"):
    all_files = os.listdir(base_dir)

    # Finds files formatted as Train_SampleX_ValY_TestZ.csv
    pattern = re.compile(r"Train_Sample(\d+)_Val(\d+)_Test(\d+)\.csv")

    # Find all training files that match the chosen val_sample and test_sample
    training_files = []
    training_samples = []
    for fname in all_files:
        match = pattern.match(fname)
        if match:
            train_sample_num, val_num, test_num = map(int, match.groups())
            if val_num == val_sample and test_num == test_sample:
                training_files.append(os.path.join(base_dir, fname))
                training_samples.append(train_sample_num)

    if not training_files:
        print(f"No training files found for Val{val_sample} and Test{test_sample}.")
        return

    # Sort files and sample numbers by training sample number
    training_files, training_samples = zip(*sorted(zip(training_files, training_samples), key=lambda x: x[1]))

    cluster_colors = {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'orange',
        4: 'purple'}

    fig, axs = plt.subplots(3, 4, figsize=(24, 18))
    axs = axs.flatten()

    for i, file in enumerate(training_files):
        df = pd.read_csv(file)
        df = df.sort_values("Time")

        ax = axs[i]
        for cluster in sorted(df["Cluster"].unique()):
            cluster_data = df[df["Cluster"] == cluster]
            grouped = cluster_data.groupby("Time")["Counts"].sum().reset_index()
            grouped["Cumulative_Hits"] = grouped["Counts"].cumsum()

            ax.plot(
                grouped["Time"],
                grouped["Cumulative_Hits"],
                label=f"Cluster {cluster}",
                color=cluster_colors.get(cluster, 'black')
            )

        ax.set_title(f"Training Sample {training_samples[i]}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative AE Hits")
        ax.grid(True)

    # Shared legend below the plots
    handles = [plt.Line2D([0], [0], color=color, label=f"Cluster {c}") for c, color in cluster_colors.items()]
    fig.legend(handles, [f"Cluster {c}" for c in cluster_colors.keys()],
               loc='lower center', ncol=3, fontsize='large', frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.show()


# Runs the training plots for the model which had validation sample 1 and test sample 2
plot_training_run(val_sample=1, test_sample=2)
