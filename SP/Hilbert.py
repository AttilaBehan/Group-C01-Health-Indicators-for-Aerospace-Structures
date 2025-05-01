import numpy as np
from scipy.signal import hilbert
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# Load data
data = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Show available column names
print("Column names:", data.columns.tolist())

# Convert each column to a NumPy array
time_array = data['Time'].to_numpy()
amplitude_array = data['Amplitude'].to_numpy()
rise_time_array = data['Rise-Time'].to_numpy()
energy_array = data['Energy'].to_numpy()
counts_array = data['Counts'].to_numpy()
duration_array = data['Duration'].to_numpy()
rms_array = data['RMS'].to_numpy()

# Function to analyze all signals with Hilbert Transform
def analyze_signal_with_hilbert(df):
    signal_columns = df.columns[1:]  # Skip 'Time' column
    fs = 1/100000
    #fs = 1 / np.mean(np.diff(df['Time']))  # Sampling frequency
    results = {}

    for col in signal_columns:
        x = df[col].to_numpy()
        analytic_signal = hilbert(x)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

        results[col] = {
            'amplitude_envelope': amplitude_envelope,
            'instantaneous_phase': instantaneous_phase,
            'instantaneous_frequency': instantaneous_frequency
        }

    return results

# Analyze signal
hilbert_results = analyze_signal_with_hilbert(data)

# Plot the results
for col, result in hilbert_results.items():
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Hilbert Transform Analysis: {col}")

    axs[0].plot(time_array, data[col], label="Original Signal")
    axs[0].plot(time_array, result['amplitude_envelope'], label="Envelope", linestyle='--')
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    axs[1].plot(time_array, result['instantaneous_phase'])
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_title("Instantaneous Phase")

    axs[2].plot(time_array[1:], result['instantaneous_frequency'])  # diff reduces length by 1
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_xlabel("Time")
    axs[2].set_title("Instantaneous Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Save all instantaneous frequencies to a single text file
time_trimmed = time_array[1:]  # Adjust for np.diff length
output_df = pd.DataFrame({'Time': time_trimmed})

for col, result in hilbert_results.items():
    output_df[col + ' InstFreq'] = result['instantaneous_frequency']

output_path = "SP\OutputHilbert.csv"
output_df.to_csv(output_path, sep='\t', index=False)
print(f"All instantaneous frequencies saved to {output_path}")