import numpy as np
from scipy import stats
import pandas as pd


                                 ## Frequency domain ##
def Frequency_domain_features(sensor):
    """
       Extracts frequency domain features from sensor data.

       Parameters:
       - sensor (1D array): Frequency domain transform of data.

       Returns:
       - F_features (1D array): Array containing frequency domain features.
    """

    #np 1D array of fft
    F_features = np.empty(14)

    #Power Spectral Density
    #S = np.abs(sensor**2)/samples
    S = sensor

    #frequency value kth spectrum line (needs adjustment)
    F = np.arange(1000, len(S)*1000+1, 1000)
    F_small = F/1000

    #Mean
    F_features[0] = np.mean(S)

    #Variance
    F_features[1] = np.var(S)

    #Skewness
    F_features[2] = stats.skew(S)

    #Kurtosis
    F_features[3] = stats.kurtosis(S)

    #P5 (Xfc)
    F_features[4] = np.sum(F * S) / np.sum(S)

    # P6
    F_features[5] = np.sqrt(np.mean( S * (F - F_features[4]) ** 2))

    #P7 (Xrmsf)
    F_features[6] = np.sqrt((np.sum(S * F_small ** 2)) / np.sum(S))*1000

    #P8
    F_features[7] = np.sqrt(np.sum(S * F_small ** 4) / np.sum(S * F_small ** 2))*1000

    #P9
    F_features[8] = np.sum(S * F_small ** 2) / (np.sqrt( np.sum(S) * np.sum(S * F_small ** 4)))

    #P10
    F_features[9] = F_features[5] / F_features[4]

    # #P11
    F_features[10] = np.mean(S * (F - F_features[4]) ** 3)/(F_features[5] ** 3)

    #P12
    F_features[11] = np.mean(S * (F - F_features[4]) ** 4)/(F_features[5] ** 4)

    #P13
    #Including forced absolute in sqrt which wasn't meant to be there
    F_features[12] = np.mean(np.sqrt(np.abs(F - F_features[4]))*S)/np.sqrt(F_features[5])

    #P14
    F_features[13] = np.sqrt(np.sum((F - F_features[4])**2*S)/np.sum(S))

    return F_features


                                   ## Time domain ##
def Time_domain_features(sensor):
    """
        Extracts time domain features from sensor data.

        Parameters:
        - sensor (1D array): Array containing sensor data.

        Returns:
        - T_features (1D array): Array containing time domain features.
    """

    # np 1D array of time domain data
    T_features = np.empty(19)

    X = sensor

    # Mean
    T_features[0] = np.mean(X)

    # Standard deviation
    T_features[1] = np.std(X)

    #Root amplitude
    T_features[2] = ((np.mean(np.sqrt(abs(X)))) ** 2)

    #Root mean squared RMS
    T_features[3] = np.sqrt(np.mean(X ** 2))

    #Root standard squared RSS
    T_features[4] = np.sqrt(np.sum(X ** 2))

    #Peak (maximum)
    T_features[5] = np.max(X)

    #Skewness
    T_features[6] = stats.skew(X)

    #Kurtosis
    T_features[7] = stats.kurtosis(X)

    #Crest factor
    T_features[8] = np.max(X) / np.sqrt(np.mean(X ** 2))

    #Clearance factor
    T_features[9] = np.max(X) / T_features[2]

    #Shape factor
    T_features[10] = np.sqrt(np.mean(X ** 2)) / np.mean(X)

    #Impulse factor
    T_features[11] = np.max(X) / np.mean(X)

    #Max-Min difference
    T_features[12] = np.max(X) - np.min(X)

    #Central moment kth order (not good enough)
    for k in range(3, 7):
        T_features[10+k] = np.mean((X - T_features[0])**k)

    #FM4 (close to kurtosis) (need central moment)
    T_features[17] = T_features[14]/T_features[1]**4

    #Median
    T_features[18] = np.median(X)

    return T_features

                                        ## STFT domain ##
def STFT_domain_features(sensor):
    """
        Extract features for STFT domain from sensor data.

        Parameters:
        - sensor (1D array): Array containing sensor data.

        Returns:
        - FT_features (1D array): Array containing features for STFT domain.
    """
    #np 1D array main features
    FT_features = np.empty(4)
    Y = sensor

    #Mean
    FT_features[0] = np.mean(Y)

    #Standard Deviation
    FT_features[1] = np.std(Y)

    #Skewness
    FT_features[2] = stats.skew(Y)

    #Kurtosis
    FT_features[3] = stats.kurtosis(Y)

    return FT_features

                        ## Correlating features with 0.95 threshold ##
def feature_correlation(features):
    """
       Filters features based on correlation coefficient threshold.

       Parameters:
       - features (2D array): Feature data for each trial.

       Returns:
       - correlation_matrix (2D array): Correlation matrix of features.
       - features (2D array): Reduced statistically significant feature array.
       - to_delete (array): Indices of features removed from the returned array.
    """

    #Calculating the correlation matrix for the feature array
    correlation_matrix = np.corrcoef(features.T)
    correlation_threshold = 0.95

    #Based on threshold create boolean matrix where True indicates a correlation above the threshold
    correlation_bool = correlation_matrix > correlation_threshold

    to_delete = []

    #Iterate over upper triangle of correlation matrix
    for column in range(len(correlation_bool)):
        for row in range(column+1, len(correlation_bool)):

            #Mark the feature for deletion if correlation is above the threshold
            if correlation_bool[column, row] == True and row not in to_delete:
                to_delete.append(row)

    to_delete.sort()

    #Delete the features from the original array based on the indices in to_delete
    features = np.delete(features, to_delete, axis=1)

    return correlation_matrix, features, np.array(to_delete)

                                    ## Converting data into features ##
def time_to_feature(data):
    """
        Converts time domain sensor data to feature data.

        Parameters:
        - data (2D array): Time domain sensor data.

        Returns:
        - features (2D array): Feature data extracted from time domain data.
    """
    # Set up array to save features
    data = np.array(data).transpose()
    features = np.empty((len(data), 19))

    #Loop through each sensor and extract time domain features
    for i in range(len(data)):
        features[i] = Time_domain_features(data[i])

    return pd.DataFrame(features).transpose()


def freq_to_feature(data):
    """
        Converts frequency domain sensor data to feature data.

        Parameters:
        - data (2D array): Frequency domain sensor data.

        Returns:
        - features (2D array): Feature data extracted from frequency domain data.
    """
    # Set up array to save features
    data = np.array(data).transpose()
    features = np.empty((len(data), 14))

    #Loop through each sensor and extract frequency domain features
    for i in range(len(data)):
        features[i] = Frequency_domain_features(data[i])

    return pd.DataFrame(features).transpose()

def STFT_to_feature(data3d):
    """
            Converts STFT domain sensor data to feature data.

            Parameters:
            - data3d (3D array): STFT domain sensor data.

            Returns:
            - features_df (DataFrame): Feature data extracted from STFT domain data.
    """
    #Setting up an output list to store feature data for every path
    out_list = []
    features = [0]*68

    #Initialize flattened version of 3D data from STFT
    #Dimension 2142 (for 17 times steps * 126 frequencies) by 56 paths
    new_flat = np.zeros((2142, 56))

    #Flatten 3D STFT data into 2D Array
    for path in range(56):
        current_path = data3d[path]
        for i in range(126):
            for j in range(17):
                new_flat[i + j * 126][path] = current_path[i][j]

    #Extracting features for each path
    for path2 in range(56):
        for i in range(17):

            #Current time step extracting 126 frequencies and computing its features
            segment = new_flat[i * 126:(i + 1) * 126, path2]
            segment_features = STFT_domain_features(segment)

            #Storing and appending features list for current path
            features[i * 4:(i + 1) * 4] = segment_features
        out_list.append(features.copy())

    features_df = pd.DataFrame(out_list).transpose()
    return features_df













## Frequency domain - Removed Features below 1.65 Fitness score####
def Frequency_domain_REDUCED_features(sensor):
    """
       Extracts frequency domain features from sensor data.

       Parameters:
       - sensor (1D array): Frequency domain transform of data.

       Returns:
       - F_features (1D array): Array containing frequency domain features.
    """

    #np 1D array of fft
    F_features = np.empty(10)

    #Power Spectral Density
    #S = np.abs(sensor**2)/samples
    S = sensor

    #frequency value kth spectrum line (needs adjustment)
    F = np.arange(1000, len(S)*1000+1, 1000)
    F_small = F/1000

    # #Mean
    # F_features[0] = np.mean(S)
    #
    # #Variance
    # F_features[1] = np.var(S)

    #Skewness
    F_features[0] = stats.skew(S)

    #Kurtosis
    F_features[1] = stats.kurtosis(S)

    #P5 (Xfc)
    F_features[2] = np.sum(F * S) / np.sum(S)

    # P6
    Not_used_now = np.sqrt(np.mean( S * (F - F_features[2]) ** 2))

    #P7 (Xrmsf)
    F_features[3] = np.sqrt((np.sum(S * F_small ** 2)) / np.sum(S))*1000

    #P8
    F_features[4] = np.sqrt(np.sum(S * F_small ** 4) / np.sum(S * F_small ** 2))*1000

    #P9
    F_features[5] = np.sum(S * F_small ** 2) / (np.sqrt( np.sum(S) * np.sum(S * F_small ** 4)))

    #P10
    F_features[6] = Not_used_now / F_features[2]

    # #P11
    F_features[7] = np.mean(S * (F - F_features[2]) ** 3)/(Not_used_now ** 3)

    #P12
    F_features[8] = np.mean(S * (F - F_features[2]) ** 4)/(Not_used_now ** 4)

    # #P13
    # #Including forced absolute in sqrt which wasn't meant to be there
    # F_features[12] = np.mean(np.sqrt(np.abs(F - F_features[4]))*S)/np.sqrt(F_features[5])

    #P14
    F_features[9] = np.sqrt(np.sum((F - F_features[2])**2*S)/np.sum(S))

    return F_features




## Time domain - Removed Features below 1.65 Fitness score##
def Time_domain_features_REDUCED(sensor):
    """
        Extracts time domain features from sensor data.

        Parameters:
        - sensor (1D array): Array containing sensor data.

        Returns:
        - T_features (1D array): Array containing time domain features.
    """

    # np 1D array of time domain data
    T_features = np.empty(14)

    X = sensor

    # Mean
    T_features[0] = np.mean(X)

    # Standard deviation
    T_features[1] = np.std(X)

    #Root amplitude
    T_features[2] = ((np.mean(np.sqrt(abs(X)))) ** 2)

    #Root mean squared RMS
    T_features[3] = np.sqrt(np.mean(X ** 2))

    #Root standard squared RSS
    T_features[4] = np.sqrt(np.sum(X ** 2))

    #Peak (maximum)
    T_features[5] = np.max(X)

    #Skewness
    T_features[6] = stats.skew(X)

    #Kurtosis
    T_features[7] = stats.kurtosis(X)

    #Crest factor
    T_features[8] = np.max(X) / np.sqrt(np.mean(X ** 2))

    #Clearance factor
    T_features[9] = np.max(X) / T_features[2]

    #Shape factor
    T_features[10] = np.sqrt(np.mean(X ** 2)) / np.mean(X)

    #Impulse factor
    T_features[11] = np.max(X) / np.mean(X)

    #Max-Min difference
    T_features[12] = np.max(X) - np.min(X)

    # #Central moment kth order (not good enough)
    # for k in range(3, 7):
    #     T_features[10+k] = np.mean((X - T_features[0])**k)
    temp_feat = np.mean((X - T_features[0])**4)

    #FM4 (close to kurtosis) (need central moment)
    T_features[13] = temp_feat/T_features[1]**4

    # #Median
    # T_features[18] = np.median(X)

    return T_features

def FFT_Feat_reduced(fftData):

    # Set up array to save features
    data_fft = np.array(fftData).transpose()
    features_fft = np.empty((len(data_fft), 10))

    # Loop through each sensor and extract frequency domain features
    for i in range(len(data_fft)):
        features_fft[i] = Frequency_domain_REDUCED_features(data_fft[i])

    return pd.DataFrame(features_fft).transpose()

def HLB_Feat_reduced(hlbData):
    # Set up array to save features
    data_hlb = np.array(hlbData).transpose()
    features_hlb = np.empty((len(data_hlb), 14))

    # Loop through each sensor and extract time domain features
    for i in range(len(data_hlb)):
        features_hlb[i] = Time_domain_features_REDUCED(data_hlb[i])

    return pd.DataFrame(features_hlb).transpose()

