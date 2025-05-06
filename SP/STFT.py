import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def perform_stft(file_path, output_path, window='hann', nperseg=500, noverlap=250):
    """
    Perform STFT on all columns except 'Time' from a CSV file.
    Returns a dictionary with column names as keys and tuples of (frequencies, times, STFT values).
    Also saves the STFT results in a CSV with frequency and time bins, along with STFT magnitudes.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Drop rows that are completely empty (e.g., ",,,,,,")
    df.dropna(how='all', inplace=True)

    # Drop 'Time' column if present
    time_array = df['Time'].to_numpy()
    df = df.drop(columns=['Time'])

    results = {}
    stft_results = {}

    # Iterate over each signal column to compute STFT
    for column in df.columns:
        data = df[column].values

        # Perform STFT
        f, t, Zxx = stft(data, fs=1 / 0.5, window=window, nperseg=nperseg, noverlap=noverlap)

        # Save the results in the results dictionary
        results[column] = (f, t, np.abs(Zxx))

        # Flatten the STFT magnitudes and save for CSV output
        stft_results[column] = np.ravel(np.abs(Zxx))

    # Flatten the frequency bins (f) and times (t) as columns
    frequency_bins = np.tile(f, len(t))  # Repeat frequency bins for each time step
    time_bins = np.repeat(t, len(f))  # Repeat time bins for each frequency

    # Create the output DataFrame
    stft_df = pd.DataFrame(stft_results)

    # Add the frequency and time bins to the DataFrame
    stft_df['Frequency (Hz)'] = frequency_bins
    stft_df['Time (s)'] = time_bins

    # Reorder the columns for better readability
    columns_order = ['Frequency (Hz)', 'Time (s)'] + [col for col in df.columns]
    stft_df = stft_df[columns_order]

    # Save the results to a CSV file
    stft_df.to_csv(output_path, index=False)

    # Plot the STFT of the 'Amplitude' column if it exists
    if 'Amplitude' in results:
        f, t, Zxx = results['Amplitude']
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)), shading='gouraud')
        plt.title('STFT of Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Log Magnitude (dB)')
        plt.grid(True)
        plt.show()

    return results

# Example usage:
input_file = r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2.csv'
output_file = r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2STFT.csv'
perform_stft(input_file, output_file)






# import pandas as pd
# import numpy as np

# # Configuration
# WINDOW_SIZE_CYCLES = 500
# CYCLE_DURATION = 0.5  # seconds
# PARAMETERS = ['Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']
# AGGREGATION_METHODS = {
#     'Amplitude': 'mean',
#     'Rise-Time': 'mean',
#     'Energy': 'sum',
#     'Counts': 'sum',
#     'Duration': 'sum',
#     'RMS': 'mean'
# }

# def process_ae_data(file_path):
#     # Load and preprocess data
#     df = pd.read_csv(file_path, header=0, names=['Time'] + PARAMETERS)
#     df.dropna(how='all', inplace=True)  # Remove empty rows
#     df['Time'] = df['Time'].astype(int)  # Convert cycles to integers
#     df.sort_values('Time', inplace=True)

#     # Create windows and process
#     results = []
#     min_cycle = df['Time'].min()
#     max_cycle = df['Time'].max()

#     for window_start in range(0, max_cycle + 1, WINDOW_SIZE_CYCLES):
#         window_end = window_start + WINDOW_SIZE_CYCLES - 1
#         window_data = df[(df['Time'] >= window_start) & (df['Time'] <= window_end)]

#         if window_data.empty:
#             continue  # Skip empty windows

#         # Create uniform time series with 0-filled missing cycles
#         uniform_series = {param: np.zeros(WINDOW_SIZE_CYCLES) for param in PARAMETERS}
        
#         for cycle in range(window_start, window_end + 1):
#             cycle_data = window_data[window_data['Time'] == cycle]
#             if not cycle_data.empty:
#                 for param in PARAMETERS:
#                     if AGGREGATION_METHODS[param] == 'sum':
#                         uniform_series[param][cycle - window_start] = cycle_data[param].sum()
#                     else:
#                         uniform_series[param][cycle - window_start] = cycle_data[param].mean()

#         # Perform FFT for each parameter
#         window_results = {
#             'window_start': window_start,
#             'window_end': window_end,
#             'fft_results': {}
#         }

#         for param in PARAMETERS:
#             signal = uniform_series[param]
#             fft = np.fft.fft(signal)
#             freq = np.fft.fftfreq(WINDOW_SIZE_CYCLES, d=CYCLE_DURATION)
            
#             # Store magnitude spectrum (first half)
#             n = len(signal)
#             window_results['fft_results'][param] = {
#                 'frequencies': freq[:n//2],
#                 'magnitude': np.abs(fft)[:n//2]
#             }

#         results.append(window_results)

#     return results

# # Usage example
# if __name__ == "__main__":
#     fft_results = process_ae_data(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv')
    
#     # Example of accessing results:
#     for window in fft_results:
#         print(f"\nWindow {window['window_start']}-{window['window_end']}:")
#         for param in PARAMETERS:
#             print(f"{param} FFT peak at {window['fft_results'][param]['frequencies'][np.argmax(window['fft_results'][param]['magnitude'])]:.2f} Hz")


# #                                                       OLD V1
# # import numpy as np
# # import pandas as pd
# # from scipy.fftpack import fft
# # import matplotlib.pyplot as plt


# # df = pd.read_csv(r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv")

# # LW = 1000
# # Int = 1000

# # def fast_fourier(df, time_col="Time", amp_col="Amplitude", real_sampling_rate=None):
# #     """
# #     Computes FFT on amplitude data, assuming time is measured in cycles.

# #     Parameters:
# #     - df (pd.DataFrame): DataFrame with time and amplitude columns.
# #     - time_col (str): Name of the column containing cycle count.
# #     - amp_col (str): Name of the column containing amplitude values.
# #     - real_sampling_rate (float, optional): Actual sampling rate in Hz if known (e.g., 1000 Hz for ms-based data).
    
# #     Returns:
# #     - freq_df (pd.DataFrame): Frequency values for FFT spectrum.
# #     - amp_df (pd.DataFrame): Corresponding amplitude values.
# #     - Displays FFT plot.
    
# #     Example:
# #     freq_df, amp_df = fast_fourier(sensor_data, real_sampling_rate=1000)
# #     """
    
# #     if time_col not in df.columns or amp_col not in df.columns:
# #         raise ValueError(f"Columns '{time_col}' and '{amp_col}' must be in the DataFrame.")
    
# #     # Extract amplitude data
# #     amplitude = df[amp_col].dropna().values  # Drop NaNs if present
# #     N = len(amplitude)  # Number of data points

# #     if N < 2:
# #         raise ValueError("Not enough data points to perform FFT.")

# #     # Remove DC Offset (Prevent Infinite 0 Hz Component)
# #     amplitude = amplitude - np.mean(amplitude)  

# #     # Determine Sampling Rate
# #     if real_sampling_rate:
# #         sampling_rate = real_sampling_rate  # If user provides a real sampling rate in Hz
# #     else:
# #         # If no real sampling rate is provided, assume sampling rate = number of samples
# #         sampling_rate = N  

# #     # Compute FFT
# #     X = np.abs(fft(amplitude))  # Magnitude spectrum

# #     # Compute frequency values
# #     freq = np.fft.fftfreq(N, d=1/sampling_rate)  # Correct frequency calculation

# #     # Apply Nyquist theorem: Keep only the one-sided spectrum
# #     n_oneside = N // 2  # Keep half due to symmetry
# #     freq_oneside = freq[:n_oneside]
# #     amp_oneside = X[:n_oneside]

# #     # Convert to DataFrames
# #     freq_df = pd.DataFrame(freq_oneside, columns=["Frequency"])
# #     amp_df = pd.DataFrame(amp_oneside, columns=["Amplitude"])

# #     # Plot FFT Spectrum
# #     plt.figure(figsize=(10, 5))
# #     plt.plot(freq_df, amp_df, label="FFT Spectrum")
# #     plt.xlabel("Frequency (Hz)")
# #     plt.ylabel("Amplitude")
# #     plt.title("Frequency Spectrum of Signal")
# #     plt.legend()
# #     plt.grid()
# #     plt.show()

# #     return freq_df, amp_df

# # freq_df, amp_df = fast_fourier(df, real_sampling_rate=LW/Int)

# # freq_df.to_csv("frequency_data.csv", index=False)
# # amp_df.to_csv("amplitude_data.csv", index=False)

#                                                                       #OLD V2
# # import pandas as pd
# # import numpy as np
# # from scipy.fft import fft, fftfreq
# # import matplotlib.pyplot as plt

# # # Load your data
# # data = pd.read_csv(r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV_interp\Sample1Interp.csv")  # Assuming your CSV has 'cycle' and 'amplitude'

# # # Ensure that the data is sorted by cycle number (if not already sorted)
# # data = data.sort_values(by='Time')

# # # Identify missing cycles
# # all_cycles = np.arange(data['Time'].min(), data['Time'].max() + 1)  # All cycles in the range
# # existing_cycles = data['Time'].values  # Cycles that have data
# # missing_cycles = np.setdiff1d(all_cycles, existing_cycles)  # Missing cycles
# # print(existing_cycles)
# # # Create a new DataFrame with all cycles
# # complete_data = pd.DataFrame({'Time': all_cycles})

# # # Merge with original data to include the amplitude values
# # complete_data = pd.merge(complete_data, data, on='Time', how='left')

# # # Interpolate missing amplitude values (linear interpolation)
# # complete_data['RMS'] = complete_data['RMS'].interpolate(method='linear')



# # # Function to compute FFT
# # def compute_fft(data, sampling_rate=1.8e6): # The sampling rate has been computed using sensor datasheet * 2 (100-900kHz)
# #     # At the same time the sampling rate should be based on analog data and not the sensor but that doesn't make sense either since it gives +/- 3.73Hz
# #     N = len(data)  # Number of data points
# #     T = 1.0 / sampling_rate  # Sampling interval
# #     xf = fftfreq(N, T)[:N//2]  # Frequency bins
    
# #     # Compute the FFT
# #     yf = fft(data)
    
# #     return xf, 2.0/N * np.abs(yf[:N//2])  # Return frequency and amplitude

# # # Compute FFT for the interpolated amplitude data
# # amplitudes = complete_data['RMS'].values
# # xf, amplitude_fft = compute_fft(amplitudes)

# # # Plot the FFT result
# # plt.plot(xf, amplitude_fft)
# # plt.title("FFT of Acoustic Emission Data")
# # plt.xlabel("Frequency (Hz)")
# # plt.ylabel("RMS")
# # plt.show()
