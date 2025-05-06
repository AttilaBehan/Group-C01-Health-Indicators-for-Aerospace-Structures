import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

# Load the data
df = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Create output directories
os.makedirs('cwt_plots', exist_ok=True)
os.makedirs('cwt_data', exist_ok=True)

# Define wavelet and scales
wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
scales = np.arange(1, 64)  # You can adjust scales based on your data

# Process each numeric column
for column in df.select_dtypes(include=[np.number]).columns:
    signal = df[column].values
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

    # Save coefficients to CSV
    coeffs_df = pd.DataFrame(np.abs(coefficients), index=scales)
    coeffs_df.to_csv(f'cwt_data/{column}_cwt.csv')

    # Plot the scalogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, len(signal), scales[-1], scales[0]], cmap='viridis', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title(f'CWT Scalogram - {column}')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.tight_layout()
    plt.savefig(f'cwt_plots/{column}_scalogram.png')
    plt.close()

print("CWT processing complete. CSVs and plots saved.")