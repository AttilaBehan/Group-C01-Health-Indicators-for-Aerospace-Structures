import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

def HI_graph(X, dir="", name="", legend=True):
    #Graph of HI against cycles
    #X is numpy list of HI at different states, & name is root to save at

    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    markers = ['o', 's', '^', 'D', 'X']
    colours = ['purple', 'blue', 'red', 'green', 'orange']

    plt.figure()
    font = 20
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18

    for sample in range(len(X)):
        states = np.arange(len(X[sample]))
        cycles = states/30*100
        if str(sample) == str(name[-1]) or samples[sample] == name[:-7]:
            plt.plot(cycles, X[sample], marker=markers[sample], color=colours[sample], label="Sample "+str(sample+1) + ": Test")
        else:
            plt.plot(cycles, X[sample], marker=markers[sample], color=colours[sample], label="Sample " + str(sample+1) + ": Train")
    if legend:
        plt.legend()

    plt.xlabel('Lifetime (%)', fontsize=font)
    plt.ylabel('HI', fontsize=font)
    plt.tight_layout()
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name)
    else:
        plt.show()
    plt.close()

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    #Stacked bar chart of criteria against features
    #Features is list of feature names; Mo, Pr, Tr are numpy lists of floats in same order, dir is root to save at
    plt.figure()
    plt.bar(features, Mo, label="Mo")
    plt.bar(features, Pr, bottom=Mo, label="Pr")
    plt.bar(features, Tr, bottom=Pr+Mo, label="Tr")
    plt.legend()
    if features[0] == "050":
        plt.xlabel('Frequency (kHz)')
    else:
        plt.xlabel('Feature')
    plt.ylabel('Fitness')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name + " PC")
    else:
        plt.show()
    plt.close()


def big_plot(dir, type, transform):
    """
        Assemble grid of HI graphs

        Parameters:
        - dir (str): Directory of HI graphs
        - type (string): "DeepSAD" or "VAE"
        - transform (string): "FFT" or "HLB"

        Returns: None
    """

    # Define variables
    panels = ("0", "1", "2", "3", "4")
    freqs = ("050", "100", "125", "150", "200", "250")

    markers = ['o', 's', '^', 'D', 'X']
    colours = ['purple', 'blue', 'red', 'green', 'orange']
    labels = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']
    legend_data = [Line2D([0], [0], marker=marker, color=colour, markerfacecolor=colour, markersize=10, label=label) for marker, colour, label in zip(markers, colours, labels)]

    nrows = len(freqs)+1
    ncols = len(panels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(37, 40))  # Adjusted figure size

    # For each frequency and panel
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):
            # Generate the filename
            filename = f"{freq}kHz_{type}_{transform}_{j}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):
                # Load the image
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot
                axs[i, j].imshow(img)
                axs[i, j].axis('off')  # Hide the axes
            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    # WAE results
    for j, panel in enumerate(panels):
        filename = f"WAE_{type}_{transform}_{j}.png"
        if os.path.exists(os.path.join(dir, filename)):
            img = mpimg.imread(os.path.join(dir, filename))
            axs[-1, j].imshow(img)
            axs[-1, j].axis('off')

        else:
            axs[-1, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
            axs[-1, j].axis('off')

    # Redefine freqs to include kHz
    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold', fontsize=40)
    axs[-1, 0].annotate("Fusion", (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold', fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'    Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center', fontweight='bold', fontsize=40)

    fig.legend(handles=legend_data, loc="center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=40)

    # Adjust spacing between subplots and save
    # plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.01, top=0.98, bottom=0.05, hspace=-0.03, wspace=-0.2)

    plt.savefig(os.path.join(dir, f"BigPlot_{type}_{transform}.pdf"))
    plt.savefig(os.path.join(dir, f"BigPlot_{type}_{transform}.png"))