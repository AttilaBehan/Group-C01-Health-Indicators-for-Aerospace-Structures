import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def wavelet_transform(csv_path, output_csv, wavelet='morl', max_scale=30):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(csv_path)
    time = df.iloc[:, 0].values
    signal_columns = df.columns[1:]
    scales = np.arange(1, max_scale + 1)

    cwt_results = {}
    freqs = None

    for col in signal_columns:
        print(f"Processing: {os.path.basename(csv_path)} | Column: {col}")
        signal = pd.to_numeric(df[col], errors='coerce').values
        signal = np.nan_to_num(signal)

        coef, freqs = pywt.cwt(signal, scales, wavelet)
        mag = np.abs(coef)
        cwt_results[col] = mag

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

    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(by=["Frequency", "Time"]).reset_index(drop=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved CWT result: {output_csv}")

def process_folder(input_folder, output_folder, wavelet='morl', max_scale=20):
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_csv = os.path.join(output_folder, f"{file_name}_cwt.csv")
        wavelet_transform(csv_file, output_csv, wavelet=wavelet, max_scale=max_scale)

# Example usage
if __name__ == "__main__":
    input_folder = r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\Low_Features_500_500_CSV"
    output_folder = r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\output CWT"
    process_folder(input_folder, output_folder)
