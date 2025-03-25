import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tftb.processing import smoothed_pseudo_wigner_ville, reassigned_smoothed_pseudo_wigner_ville
import matplotlib.pyplot as plt

# Data
def CSV_to_Array(file):
    """
        Converts column data of a Sample.csv into separate arrays.

        Parameters:
            - file (string): File path of the to be converted .csv file.

        Returns:
            - sample_amplitude, sample_risetime, sample_energy,
            sample_counts, sample_duration, sample_rms (1D array):
            Array containing corresponding low level feature data
    """

    sample_df = pd.read_csv(file)  # Converts csv into dataframe

    # Convert respective dataframe colum to a numpy array
    sample_cycle_time = sample_df["Time"].to_numpy()
    sample_amplitude = sample_df["Amplitude"].to_numpy()
    sample_risetime = sample_df["Rise-Time"].to_numpy()
    sample_energy = sample_df["Energy"].to_numpy()
    sample_counts = sample_df["Counts"].to_numpy()
    sample_duration = sample_df["Duration"].to_numpy()
    sample_rms = sample_df["RMS"].to_numpy()

    return sample_cycle_time, sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms

print('loading data...')
# Example raw data
cycle_numbers, sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms = CSV_to_Array('SP\\Sample1Interp.csv')
measurements = sample_amplitude
cycle_numbers = cycle_numbers

# # Convert to DataFrame
# df = pd.DataFrame({'Cycle': cycle_numbers, 'Measurement': measurements})

# # Define the number of points per cycle
# num_points_per_cycle = 2 # Adjust based on your data
# print('Interpolating cycles...')
# # Resample each cycle
# resampled_cycles = []
# resampled_measurements = []

# for cycle in df['Cycle'].unique():
#     cycle_data = df[df['Cycle'] == cycle]['Measurement'].values
#     original_x = np.linspace(0, 1, len(cycle_data))  # Original sample points
#     new_x = np.linspace(0, 1, num_points_per_cycle)  # New uniform sample points

#     # Choose interpolation method based on available data
#     if len(cycle_data) >= 4:
#         interpolator = interp1d(original_x, cycle_data, kind='cubic', fill_value="extrapolate")
#     elif len(cycle_data) == 2 or len(cycle_data) == 3:
#         interpolator = interp1d(original_x, cycle_data, kind='linear', fill_value="extrapolate")
#     else:  # If only 1 data point, repeat it
#         interpolator = lambda x: np.full_like(x, cycle_data[0])

#     # Resample
#     resampled_signal = interpolator(new_x)

#     # Store results
#     resampled_cycles.extend([cycle] * num_points_per_cycle)
#     resampled_measurements.extend(resampled_signal)

# # Convert to NumPy arrays
# uniform_cycles = np.array(resampled_cycles)
# uniform_measurements = np.array(resampled_measurements)
uniform_cycles = np.array(cycle_numbers)
uniform_measurements = np.array(measurements)

plt.subplot(1, 2, 1)
plt.plot(uniform_cycles, uniform_measurements, color='red', label='Resampled Data')
plt.xlabel('Cycles')
plt.ylabel('Amplitude')
plt.title('Resampled Data')

plt.subplot(1, 2, 2)
plt.plot(cycle_numbers, measurements, color='blue', label='Original')
plt.xlabel('Cycles')
plt.ylabel('Amplitude')
plt.title('Original Data')
plt.show()

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
axes[0].plot(uniform_cycles, uniform_measurements, color='red', label='Resampled Data')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Resampled Data')
axes[1].plot(cycle_numbers, measurements, color='blue', label='Original')
axes[1].set_xlabel('Cycles')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Original Data')
plt.tight_layout()
plt.show()



print('applying SPWVD...')
# Apply SPWVD

window_size = 500  # Size of each window
overlap_size = 250  # Overlap between windows (50%)

# Initialize lists to store the time-frequency results
tfr_list = []
times_list = []

# Process signal in overlapping windows
for start in range(0, len(uniform_measurements) - window_size, overlap_size):
    window_data = uniform_measurements[start:start + window_size]
    
    # Apply SPWVD to the window
    spwvd = smoothed_pseudo_wigner_ville(window_data)
    tfr, times, freqs = spwvd.run()
    
    # Adjust the times to reflect the actual time within the full signal
    times = times + start  # Add the starting point of the window to the times
    
    # Store the time-frequency representation and times
    tfr_list.append(tfr)
    times_list.append(times)

# Stack all the time-frequency representations from each window
tfr_combined = np.concatenate(tfr_list, axis=1)  # Combine along time axis
times_combined = np.concatenate(times_list)  # Concatenate times for all windows

# Plot the combined time-frequency representation
plt.imshow(np.abs(tfr_combined), aspect='auto', origin='lower', 
           extent=[min(times_combined), max(times_combined), min(freqs), max(freqs)])
plt.xlabel("Cycle")
plt.ylabel("Frequency (per cycle)")
plt.title("SPWVD - Cycle-Based Data")
plt.colorbar(label="Amplitude")
plt.show()


# uniform_cycles = uniform_cycles[:2000]
# uniform_measurements = uniform_measurements[:2000]
# spwvd = smoothed_pseudo_wigner_ville(uniform_measurements)
# tfr, times, freqs = spwvd.run()

# Plot the time-frequency representation
# print('plotting...')
# plt.imshow(np.abs(tfr), aspect='auto', origin='lower', extent=[min(uniform_cycles), max(uniform_cycles), min(freqs), max(freqs)])
# plt.xlabel("Cycle")
# plt.ylabel("Frequency (per cycle)")
# plt.title("SPWVD - Cycle-Based Data")
# plt.colorbar(label="Amplitude")
# plt.show()
