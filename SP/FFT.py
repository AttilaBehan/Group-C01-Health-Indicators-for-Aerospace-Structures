import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def perform_fft(file_path):
    """
    Perform FFT on all columns except 'Time' from a CSV file.
    Returns a dictionary with column names as keys and tuples of (frequencies, normalized FFT values).
    Also plots the normalized FFT of the 'Amplitude' column.
    """
    # Parameters
    sampling_period = 0.5  # seconds per cycle
    fs = 1 / sampling_period  # samples per second (Hz)

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Drop 'Time' column if present
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    results = {}

    for column in df.columns:
        data = df[column].values
        N = len(data)

        # Perform FFT
        data_fft = fft(data)

        # Normalize
        # data_fft = data_fft / N

        # Frequency bins
        freqs = fftfreq(N, d=sampling_period)

        # Only keep positive frequencies
        mask = freqs >= 0
        freqs_pos = freqs[mask]
        data_fft_pos = data_fft[mask]

        results[column] = (freqs_pos, np.abs(data_fft_pos))


    if 'RMS' in results:  #Time,Amplitude,Rise-Time,Energy,Counts,Duration,RMS
        freqs_plot, interest_plot = results['RMS']
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_plot, interest_plot)
        plt.title('Normalized FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('RMS')
        plt.grid(True)
        plt.show()

    return results


results = perform_fft(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv')
