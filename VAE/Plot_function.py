import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_results(train_data, test_data, filepath, show):
    # Plot 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # so we can index it easily as axes[0] to axes[11]
    x = np.linspace(0, 100, len(test_data[0]))
    filepath = filepath + r"\plot.png"
    for i in range(12):
        ax = axes[i]
        
        # Plot training lines
        for j in range(11):
            ax.plot(x, train_data[i, j], color='gray', alpha=0.5, linewidth=1)

        # Plot test line
        ax.plot(x, test_data[i], color='blue', linewidth=2, label='Test')

        ax.set_title(f"Sample {i+1}")
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 1])  # adjust based on your data range


    legend_elements = [
    Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Training Data'),
    Line2D([0], [0], color='blue', lw=2, label='Testing Data')
    ]

    # ✅ Add a global legend box below the subplots
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize='large', frameon=True, bbox_to_anchor=(0.5, -0.02))

    # Layout adjustment to make space for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the bottom
    if show:
        plt.show()
    else:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Plot saved succesfully to {filepath}")

N = 120  # or whatever your time resolution is  # x-axis for all curves

# Example placeholder data
# Replace with your actual train/test data
train_data = np.random.rand(12, 11, N)
test_data = np.random.rand(12, N)
plot_results(train_data, test_data, r"C:\Users\job\Downloads", False)