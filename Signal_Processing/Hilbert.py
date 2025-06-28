from scipy.signal import hilbert 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def perform_ht(input_dir, output_dir):
    """
    Compute Hilbert amplitude envelopes for each signal column (excluding 'Time').
    - Plot each envelope over the original signal.
    - Save all envelope data to a CSV.
    - Optionally, save plots as PNG files in `plot_dir`.
    """

    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file_path=os.path.join(root, sample) 
            # Load the CSV file
            print("Processing:", sample)
            df = pd.read_csv(file_path)

            # Drop rows that are completely empty (e.g., ",,,,,,")
            df.dropna(how='all', inplace=True)

            time = df['Time (cycle)'].values
            df = df.drop(columns=['Time (cycle)'])


            envelope_data = {}


            for column in df.columns:
                signal = df[column].values
                analytic_signal = hilbert(signal)
                envelope = np.abs(analytic_signal)
                envelope_data[f'{column}_envelope'] = envelope

                # # Plot
                # plt.figure(figsize=(10, 4))
                # plt.plot(time, signal, label='Original Signal', alpha=0.6)
                # plt.plot(time, envelope, label='Hilbert Envelope', linewidth=2)
                # plt.title(f'Hilbert Amplitude Envelope - {column}')
                # plt.xlabel('Time')
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.grid(True)
                # plt.tight_layout()

                # if plot_dir:
                #     plt.savefig(os.path.join(plot_dir, f'{column}_envelope.png'))
                #     plt.close()
                # else:
                #     plt.show()

            # Create and save DataFrame
            envelope_df = pd.DataFrame(envelope_data)
            envelope_df.insert(0, 'Time (cycle)', time)
            filename = os.path.join(output_dir, sample)
            envelope_df.to_csv(filename, index=False) 

    return envelope_df
