import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def wavelet_transform(csv_path, wavelet='morl', max_scale=50, output_csv):
    """
    Performs CWT on all signal columns in a CSV (except the first time column) and exports results to a CSV.

    Parameters:
        csv_path (str): Path to the CSV file.
        wavelet (str): The name of the wavelet to use. Default is 'morl'.
        max_scale (int): Maximum scale for CWT. Default is 50.
        output_csv (str): Output CSV file path to save the magnitude of CWT coefficients.
    """
    # Load data
    df = pd.read_csv(csv_path)
    time = df.iloc[:, 0].values
    signal_columns = df.columns[1:]
    scales = np.arange(1, max_scale + 1)

    all_cwt_data = []

    for col in signal_columns:
        signal = df[col].values
        coef, freqs = pywt.cwt(signal, scales, wavelet)
        mag = np.abs(coef)  # Get magnitude of complex coefficients

        # Flatten the result and label it
        for scale_idx, scale_val in enumerate(scales):
            all_cwt_data.append({
                "Signal": col,
                "Scale": scale_val,
                "Frequency": freqs[scale_idx],
                **{f"Time_{t:.3f}": mag[scale_idx, idx] for idx, t in enumerate(time)}
            })

        # Optional: plot each scalogram (can comment this out if too many)
        plt.figure(figsize=(15, 6))
        extent = [time[0], time[-1], freqs[-1], freqs[0]]
        plt.imshow(mag, extent=extent,
                   interpolation='bilinear', cmap='jet', aspect='auto',
                   vmin=0, vmax=np.percentile(mag, 99))
        plt.title(f"CWT Scalogram - {col}")
        plt.xlabel("Time")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude")
        plt.tight_layout()
        plt.show()

    # Convert results to DataFrame
    cwt_df = pd.DataFrame(all_cwt_data)

    # Save to CSV
    cwt_df.to_csv(output_csv, index=False)
    print(f"CWT magnitudes saved to: {output_csv}")


wavelet_transform(
    csv_path=r"SP\LowLevelFeaturesSample1.csv",
    output_csv=r"cwt_output.csv"
)
