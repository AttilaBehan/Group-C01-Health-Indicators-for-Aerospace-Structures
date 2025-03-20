import numpy as np
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt


df = pd.read_csv(r"Low_Features_CSV\Sample1.csv")

LW = 1000
Int = 1000

def fast_fourier(df, time_col="Time", amp_col="Amplitude", real_sampling_rate=None):
    """
    Computes FFT on amplitude data, assuming time is measured in cycles.

    Parameters:
    - df (pd.DataFrame): DataFrame with time and amplitude columns.
    - time_col (str): Name of the column containing cycle count.
    - amp_col (str): Name of the column containing amplitude values.
    - real_sampling_rate (float, optional): Actual sampling rate in Hz if known (e.g., 1000 Hz for ms-based data).
    
    Returns:
    - freq_df (pd.DataFrame): Frequency values for FFT spectrum.
    - amp_df (pd.DataFrame): Corresponding amplitude values.
    - Displays FFT plot.
    
    Example:
    freq_df, amp_df = fast_fourier(sensor_data, real_sampling_rate=1000)
    """
    
    if time_col not in df.columns or amp_col not in df.columns:
        raise ValueError(f"Columns '{time_col}' and '{amp_col}' must be in the DataFrame.")
    
    # Extract amplitude data
    amplitude = df[amp_col].dropna().values  # Drop NaNs if present
    N = len(amplitude)  # Number of data points

    if N < 2:
        raise ValueError("Not enough data points to perform FFT.")

    # Remove DC Offset (Prevent Infinite 0 Hz Component)
    amplitude = amplitude - np.mean(amplitude)  

    # Determine Sampling Rate
    if real_sampling_rate:
        sampling_rate = real_sampling_rate  # If user provides a real sampling rate in Hz
    else:
        # If no real sampling rate is provided, assume sampling rate = number of samples
        sampling_rate = N  

    # Compute FFT
    X = np.abs(fft(amplitude))  # Magnitude spectrum

    # Compute frequency values
    freq = np.fft.fftfreq(N, d=1/sampling_rate)  # Correct frequency calculation

    # Apply Nyquist theorem: Keep only the one-sided spectrum
    n_oneside = N // 2  # Keep half due to symmetry
    freq_oneside = freq[:n_oneside]
    amp_oneside = X[:n_oneside]

    # Convert to DataFrames
    freq_df = pd.DataFrame(freq_oneside, columns=["Frequency"])
    amp_df = pd.DataFrame(amp_oneside, columns=["Amplitude"])

    # Plot FFT Spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(freq_df, amp_df, label="FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Signal")
    plt.legend()
    plt.grid()
    plt.show()

    return freq_df, amp_df

freq_df, amp_df = fast_fourier(df, real_sampling_rate=LW/Int)

freq_df.to_csv("frequency_data.csv", index=False)
amp_df.to_csv("amplitude_data.csv", index=False)