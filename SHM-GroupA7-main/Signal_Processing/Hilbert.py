#importing libraries
import numpy as np
from scipy.signal import hilbert, chirp
import pandas as pd
import matplotlib.pyplot as plt


# Assuming df is your dataframe read from csv
def Hilbert(data, time):
    """
        Extracts amplitude envelope and instantaneous frequency from sensor data using the Hilbert transform.
        simple explanation: It basically takes the instant frequency of a signal, So if a signal oscillates with
        increasing speed, the graph of the hilbert transform will increase in time.

        Parameters:
        - data : A DataFrame where each column contains time-domain data.
        - time (1D): The time values corresponding to the data.

        Returns:
        - DataFrame: A DataFrame containing the amplitude envelopes of the analytic signal for each data column.

        Example:
        result = Hilbert(sensor_data, time_values)
        """
    x_values = [] #check(not needed)
    time = pd.DataFrame(time)
    inst_freq_arr = [] #initialize array
    amp_arr = [] #check(not needed)
    # Iterate through sensors (Not 8 because of the actuator not being included)
    for i in range(7):
        x_values = [] #check
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]
            time_intervals = np.diff(time)
            fs = 1 / 5e-7 #sampling rate

            analytic_signal = hilbert(x) #apply Hilbert
            amplitude_envelope = np.abs(analytic_signal) #Take the absolute value
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs) #check(not needed)

            inst_freq_arr.insert(k, amplitude_envelope) #instant frequency array
    return pd.DataFrame(inst_freq_arr).transpose()

def giveTime():
    time = []
    for i in range(2000):
        time.append(i*(5e-7))
    return pd.DataFrame(time)