from scipy.signal import hilbert
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def perform_ht(file_path, output_csv_path, plot_dir=None):
    """
    Compute Hilbert amplitude envelopes for each signal column (excluding 'Time').
    - Plot each envelope over the original signal.
    - Save all envelope data to a CSV.
    - Optionally, save plots as PNG files in `plot_dir`.
    """
    # Load CSV
    df = pd.read_csv(file_path)

    # Drop rows that are completely empty (e.g., ",,,,,,")
    df.dropna(how='all', inplace=True)

    # Handle time
    if 'Time' in df.columns:
        time = df['Time'].values
        df = df.drop(columns=['Time'])


    envelope_data = {}


    for column in df.columns:
        signal = df[column].values
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        envelope_data[f'{column}_envelope'] = envelope

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, signal, label='Original Signal', alpha=0.6)
        plt.plot(time, envelope, label='Hilbert Envelope', linewidth=2)
        plt.title(f'Hilbert Amplitude Envelope - {column}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # if plot_dir:
        #     plt.savefig(os.path.join(plot_dir, f'{column}_envelope.png'))
        #     plt.close()
        # else:
        #     plt.show()

    # Create and save DataFrame
    envelope_df = pd.DataFrame(envelope_data)
    envelope_df.insert(0, 'Time', time)
    envelope_df.to_csv(output_csv_path, index=False)

    return envelope_df


envelope_df = perform_ht(
    file_path=r"SP\LowLevelFeaturesSample1.csv",
    output_csv_path=r"SP\OutputCWT.csv",
    plot_dir=None  # Set to None to disable saving plots
)
