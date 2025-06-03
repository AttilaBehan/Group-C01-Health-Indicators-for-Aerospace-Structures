import pywt
import numpy as np
import pandas as pd
import os
import glob

def average_in_bins(df, bin_size=100):
    n_rows = len(df)
    usable_rows = (n_rows // bin_size) * bin_size
    df = df.iloc[:usable_rows]
    binned = df.groupby(df.index // bin_size).mean()
    return binned.reset_index(drop=True)

def wavelet_transform(csv_path, output_csv, wavelet='morl', max_scale=30):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(csv_path)
    df_avg = average_in_bins(df, bin_size=100)

    time = df_avg.iloc[:, 0].values
    signal_columns = df_avg.columns[1:]
    scales = np.arange(1, max_scale + 1)

    cwt_results = {}
    freqs = None

    for col in signal_columns:
        print(f"Processing: {os.path.basename(csv_path)} | Column: {col}")
        signal = pd.to_numeric(df_avg[col], errors='coerce').values
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
    print(f"Saved averaged CWT result: {output_csv}")

def process_folder(input_folder, output_folder, wavelet='morl', max_scale=30):
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    print(f"Found {len(csv_files)} samples.")
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_csv = os.path.join(output_folder, f"{file_name}_cwt_avg.csv")
        wavelet_transform(csv_file, output_csv, wavelet=wavelet, max_scale=max_scale)

# Example usage
if __name__ == "__main__":
    input_folder = r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\Low_Features_500_500_CSV"
    output_folder = r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\CWTout"
    process_folder(input_folder, output_folder)
