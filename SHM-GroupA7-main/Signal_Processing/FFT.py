#importing libraries
import pandas as pd
import numpy as np
from numpy.fft import fft
#Following libraries were used for plotting
from scipy import stats
from matplotlib import pyplot as plt


def fast_fourier(data):
    """
        Extracts frequency domain features from sensor data using Fast Fourier Transform (FFT).

        Parameters:
        - data : A DataFrame where each column contains time-domain data.

        Returns:
        - tuple: A tuple containing two DataFrames:
            - The first DataFrame contains the frequency values.
            - The second DataFrame contains the amplitude values of the frequency components.

        Example:
        # Example usage of the function
        freq_df, amp_df = fast_fourier(sensor_data)
        """
    #initialising arrays
    freq_arr = []
    amp_arr = []
    # Iterate through sensors (Not 8 because of the actuator not being included)
    for i in range(7):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]

            X = np.abs(fft(x))  #Magnitude only
            N = len(X)
            n = np.arange(N)
            sr = 1 / 5e-7 # Sampling rate
            T = N / sr                              # total time
            freq = n / T #frequency

            # Get the one-sided spectrum
            n_oneside = N // 2 #Due to symmetry
            # get the one side frequency
            f_oneside = freq[:n_oneside]

            x_values.append(X[:n_oneside])  #not needed

            freq_arr.insert(k,f_oneside)
            amp_arr.insert(k,X[:n_oneside])
    return pd.DataFrame(freq_arr).transpose(), pd.DataFrame(amp_arr).transpose()

#This is the original
'''
# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

x_values = []
for df in dataframes:
    time = df.iloc[:, 0]
    data = df.iloc[:, 1]

    time_intervals = np.diff(time)
    mean_interval = np.mean(time_intervals)
    fs = 1 / mean_interval

    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * fs)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(time, data, label='signal')
    ax0.plot(time, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1.plot(time[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    fig.tight_layout()
    plt.show()
'''
'''
def fast_fourier(data):

    timecol = data.iloc[:,0]
    # freqcol = np.array([])
    # ampcol = np.array([])
    freq_arr = []
    amp_arr = []
    for i in range(8):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]

            X = fft(x)
            #print(X)
            N = len(X)
            n = np.arange(N)
            sr = 1 / 5e-7 # Sample rate calculation
            T = N / sr                              # total time
            freq = n / T
            #print(freq)

            # Get the one-sided spectrum
            n_oneside = N // 2
            # get the one side frequency
            f_oneside = freq[:n_oneside]

            x_values.append(X[:n_oneside])  #why do we need this line?

            # freq_arr = np.append(freq_arr, f_oneside, axis=1)
            # amp_array = np.append(amp_array, np.abs(X[:n_oneside]), axis=1)
            freq_arr.insert(k-1,f_oneside)
            amp_arr.insert(k-1,X[:n_oneside])
            # ampcol = np.append(ampcol, np.abs(X[:n_oneside]))

            # plt.figure(figsize=(12, 6))
            # plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
            # plt.xlabel('Freq (Hz)')
            # plt.ylabel('FFT Amplitude |X(freq)|')
            # plt.show()
            #print("done ",k)
        #print(x_values[0])

    #print("THIS IS THE OUTPUT!!!!!!! ", x_values)
    # print(len(x_values[1]))
    # print(len(x_values))
    return pd.DataFrame(freq_arr), pd.DataFrame(amp_arr)
'''