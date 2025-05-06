import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def perform_fft(file_path, output_csv_path):
    """
    Perform FFT on all columns except 'Time' from a CSV file.
    Extract key features like Frequency, Amplitude, Rise-Time, Energy, Counts, Duration, and RMS.
    Write the results to a CSV file.
    """
    # Parameters
    sampling_period = 0.5  # seconds per cycle
    fs = 1 / sampling_period  # samples per second (Hz)

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Drop rows that are completely empty (e.g., ",,,,,,")
    df.dropna(how='all', inplace=True)

    # Drop 'Time' column if present
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])


    results = []

    for column in df.columns:
        data = df[column].values
        N = len(data)

        # Perform FFT
        data_fft = fft(data)

        # Normalize
        data_fft = data_fft / N
        if N % 2 == 0:
            data_fft[1:N//2] *= 2  # For even length, adjust the positive frequencies
        else:
            data_fft[1:(N+1)//2] *= 2  # For odd length, adjust the positive frequencies

        # Frequency bins
        freqs = fftfreq(N, d=sampling_period)

        # Only keep positive frequencies
        mask = freqs >= 0
        freqs_pos = freqs[mask]
        data_fft_pos = data_fft[mask]

        # Calculate features
        amplitude = np.abs(data_fft_pos)
        rms = np.sqrt(np.mean(np.square(amplitude)))  # RMS calculation
        energy = np.sum(np.square(amplitude))  # Energy is the sum of squared amplitudes
        rise_time = np.max(amplitude)  # Placeholder for rise-time (to be defined appropriately)
        counts = len(amplitude)  # Number of data points
        duration = N * sampling_period  # Duration in seconds

        # Append results for this column
        for freq, amp in zip(freqs_pos, amplitude):
            results.append({
                'Frequency': freq,
                'Amplitude': amp,
                'Rise-Time': rise_time,
                'Energy': energy,
                'Counts': counts,
                'Duration': duration,
                'RMS': rms
            })
    

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(output_csv_path, index=False)

    return results_df

# Example usage:
output_csv_path = r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2FFT.csv'
results_df = perform_fft(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2.csv', output_csv_path)


'''
---------------------
'''


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq

# def perform_fft(file_path):
#     """
#     Perform FFT on all columns except 'Time' from a CSV file.
#     Returns a dictionary with column names as keys and tuples of (frequencies, normalized FFT values).
#     Also plots the normalized FFT of the 'Amplitude' column.
#     """
#     # Parameters
#     sampling_period = 0.5  # seconds per cycle
#     fs = 1 / sampling_period  # samples per second (Hz)

#     # Load the CSV file
#     df = pd.read_csv(file_path)

#     # Drop 'Time' column if present
#     if 'Time' in df.columns:
#         df = df.drop(columns=['Time'])

#     results = {}

#     for column in df.columns:
#         data = df[column].values
#         N = len(data)

#         # Perform FFT
#         data_fft = fft(data)

#         # Normalize
#         data_fft = data_fft / N

#         # Frequency bins
#         freqs = fftfreq(N, d=sampling_period)

#         # Only keep positive frequencies
#         mask = freqs >= 0
#         freqs_pos = freqs[mask]
#         data_fft_pos = data_fft[mask]

#         results[column] = (freqs_pos, np.abs(data_fft_pos))



    # if 'RMS' in results:  #Time,Amplitude,Rise-Time,Energy,Counts,Duration,RMS
    #     freqs_plot, interest_plot = results['RMS']
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(freqs_plot, interest_plot)
    #     plt.title('Normalized FFT')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('RMS')
    #     plt.grid(True)
    #     plt.show()

#     return results


# results = perform_fft(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv')
