import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_results(train_data, test_data, filepath, show):
    """This function plots the health indactor from a model. with one plot for each sample used for testing
        inputs:
        train_data (list of 10xtime_length), list of array: should contain the training health indidactor data for all 12 training samples
        test_data (12Xtime_lenght), array: should contain the testing health indicators for everey sample
        filepath (string): should contain the filepath where the plot should be saved if show is set to False
        show (bolean): Determines whether or not the plot will be shown and not saved or not shown but saved to the filepath specified"""
    # Plot 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # so we can index it easily as axes[0] to axes[11]
    x = np.linspace(0, 100, len(test_data[0]))
    filepath = filepath + r"\plot.png"
    for i in range(12):
        ax = axes[i]
        
        # Plot training lines
        for j in range(12):
            if i!= j: # make sure only the training samples are plotted that are not used for testing
                ax.plot(x, train_data[i, j], color='gray', alpha=0.5, linewidth=0.5)


        # Plot test line
        ax.plot(x, test_data[i], color='blue', linewidth=1, label='Test')
        ax.set_ylabel("Health")
        ax.set_xlabel("Lifetime [%]")
        ax.set_title(f"Test sample {i+1}")
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
if __name__ == "__main__":
    N = 120  # or whatever your time resolution is  # x-axis for all curves

    # Example placeholder data
    # Replace with your actual train/test data
    train_data = np.random.rand(12, 12, N)
    test_data = np.random.rand(12, N)
    plot_results(train_data, test_data, r"C:\Users\job\Downloads", True)
    print(train_data.shape, test_data.shape)