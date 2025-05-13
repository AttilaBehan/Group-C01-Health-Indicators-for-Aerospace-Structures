import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def wavelet_transform(csv_path, output_csv, wavelet='morl', max_scale=30):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(csv_path)
    time = df.iloc[:, 0].values
    signal_columns = df.columns[1:]
    scales = np.arange(1, max_scale + 1)

    # Dict to store each signal's CWT magnitude per scale & time
    cwt_results = {}

    freqs = None  # To store frequency values from CWT

    # Compute CWT for each signal
    for col in signal_columns:
        print(f"Processing: {col}")
        signal = pd.to_numeric(df[col], errors='coerce').values
        signal = np.nan_to_num(signal)

        coef, freqs = pywt.cwt(signal, scales, wavelet)
        mag = np.abs(coef)  # shape: (num_scales, num_time)

        cwt_results[col] = mag

        # # Optional: plot each scalogram
        # plt.figure(figsize=(15, 6))
        # extent = [time[0], time[-1], freqs[-1], freqs[0]]
        # plt.imshow(mag, extent=extent,
        #            interpolation='bilinear', cmap='jet', aspect='auto',
        #            vmin=0, vmax=np.percentile(mag, 99))
        # plt.title(f"CWT Scalogram - {col}")
        # plt.xlabel("Time")
        # plt.ylabel("Frequency (Hz)")
        # plt.colorbar(label="Magnitude")
        # plt.tight_layout()
        # plt.show()

    # Build output DataFrame in desired format:
    output_rows = []

    for scale_idx, freq_val in enumerate(freqs):
        for time_idx, t_val in enumerate(time):
            row = {
                "Frequency": freq_val,
                "Time": t_val
            }
            for col in signal_columns:
                row[col] = cwt_results[col][scale_idx, time_idx]
            output_rows.append(row)

    # Convert to DataFrame
    output_df = pd.DataFrame(output_rows)

    # Sort by Frequency (optional, should already be ordered)
    output_df = output_df.sort_values(by=["Frequency", "Time"]).reset_index(drop=True)

    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"CWT table saved to: {output_csv}")

'''
 #EXAMPLE USE WATCH OUT - THE OUTPUT IS VEEERY LARGE - AROUND 60 MILION LINES
wavelet_transform(
    csv_path = r"SP\LowLevelFeaturesSample1.csv",
    output_csv = r"SP\OutputCWT.csv"
)
'''