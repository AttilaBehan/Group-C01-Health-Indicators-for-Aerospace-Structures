import numpy as np
from scipy import stats
import pandas as pd


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
    sample_amplitude = sample_df["Amplitude"].to_numpy()
    sample_risetime = sample_df["Rise-Time"].to_numpy()
    sample_energy = sample_df["Energy"].to_numpy()
    sample_counts = sample_df["Counts"].to_numpy()
    sample_duration = sample_df["Duration"].to_numpy()
    sample_rms = sample_df["RMS"].to_numpy()

    return sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms

def Time_Domain_Features(data):
    """ Code copied/adapted from last year. """

    """
        Extracts time domain features from sensor data.

        Parameters:
            - data (1D array): Array containing data of a low-level feature.

        Returns:
            - T_features (1D array): Array containing time domain features.
    """

    # Numpy 1D array of time domain data
    T_features = np.empty(19)

    X = data

    # Mean
    T_features[0] = np.mean(X)

    # Standard deviation
    T_features[1] = np.std(X)

    # Root amplitude
    T_features[2] = ((np.mean(np.sqrt(abs(X)))) ** 2)

    # Root mean squared (RMS)
    T_features[3] = np.sqrt(np.mean(X ** 2))

    # Root standard squared (RSS)
    T_features[4] = np.sqrt(np.sum(X ** 2))

    # Peak
    T_features[5] = np.max(X)

    # Skewness
    T_features[6] = stats.skew(X)

    # Kurtosis
    T_features[7] = stats.kurtosis(X)

    # Crest factor
    T_features[8] = np.max(X) / np.sqrt(np.mean(X ** 2))

    # Clearance factor
    T_features[9] = np.max(X) / T_features[2]

    # Shape factor
    T_features[10] = np.sqrt(np.mean(X ** 2)) / np.mean(X)

    # Impulse factor
    T_features[11] = np.max(X) / np.mean(X)

    # Max-Min difference
    T_features[12] = np.max(X) - np.min(X)

    # Central moment kth order
    for k in range(3, 7):
        T_features[10+k] = np.mean((X - T_features[0])**k)

    # FM4 (close to Kurtosis)
    T_features[17] = T_features[14]/T_features[1]**4

    # Median
    T_features[18] = np.median(X)

    return T_features
