import numpy as np
from scipy import stats
import pandas as pd
import math
import os
import warnings

warnings.simplefilter(action='ignore')

def time_domain_features(data):
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

def frequency_domain_features(data):
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
    S = data 

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

def extract_time_statistical_features(cycle_length, input_dir, output_dir):

    """
        Extracts time domain features from sensor data.

        Parameters:
            - cycle_length (int): Length of the cycle.
            - dir (string): Directory of the samples.

        Output:
            - CSV files containing time domain features.
    """

    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root, sample)
            
            df = pd.read_csv(file)
            print(file)
            end_val=math.ceil(df["Time"].iloc[-1] / cycle_length) * cycle_length
            cycles=np.arange(cycle_length, end_val+1, cycle_length)
            prev_index=-1

            #define the time domain features dataframe
            base_time_features=['Mean','Standard_deviation','Root_amplitude','Root_mean_squared','Root_standard_squared','Peak','Skewness','Kurtosis','Crest_factor','Clearance_factor','Shape_factor','Impulse_factor','Max_Min_difference','Central_moment_3rd_order','Central_moment_4th_order','Central_moment_5th_order','Central_moment_6th_order','FM4','Median']
            time_features=[]
            for column in df.iloc[:, 1:]:
                time_features.append([str(column)+"_"+i for i in base_time_features])
            time_features=np.concatenate(time_features)
            features_df=pd.DataFrame(columns=time_features)

            os.makedirs(output_dir, exist_ok=True)

            for cycle in cycles:
                current_row=[]
                index=df["Time"][df["Time"]<=cycle].index[-1] 
                for column in df.iloc[:, 1:]:
                    try:
                        current_row.append(time_domain_features(df[column][prev_index+1:index+1].to_numpy().flatten()))
                    except:
                        current_row.append([np.nan]*19)
                current_row=np.concatenate(current_row)
                new_row=pd.DataFrame([current_row], columns=time_features)
                features_df = pd.concat([features_df, new_row], ignore_index=True)
                prev_index=index

            features_df.insert(0, "Time (cycle)", cycles)
            features_df=features_df.dropna()
            csv_filename = os.path.join(output_dir, f"{sample[:-4]}.csv")
            features_df.to_csv(csv_filename, index=False)    

def extract_frequency_statistical_features(cycle_length, input_dir, output_dir):

    """
        Extracts time domain features from sensor data.

        Parameters:
            - cycle_length (int): Length of the cycle.
            - dir (string): Directory of the samples.

        Output:
            - CSV files containing time domain features.
    """

    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root, sample)
            
            df = pd.read_csv(file)
            print(file)
            end_val=math.ceil(df["Frequency"].iloc[-1] / cycle_length) * cycle_length
            cycles=np.arange(cycle_length, end_val+1, cycle_length)
            prev_index=-1

            #define the time domain features dataframe
            base_frequency_features=['Mean','Variance','Skewness','Kurtosis','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14']            
            frequency_features=[]
            for column in df.iloc[:, 1:]:
                frequency_features.append([str(column)+"_"+i for i in base_frequency_features])
            frequency_features=np.concatenate(frequency_features)
            features_df=pd.DataFrame(columns=frequency_features)

            os.makedirs(output_dir, exist_ok=True)

            for cycle in cycles:
                current_row=[]
                index=df["Frequency"][df["Frequency"]<=cycle].index[-1]  
                for column in df.iloc[:, 1:]:
                    try:
                        current_row.append(frequency_domain_features(df[column][prev_index+1:index+1].to_numpy().flatten()))
                    except:
                        current_row.append([np.nan]*19)
                current_row=np.concatenate(current_row)
                new_row=pd.DataFrame([current_row], columns=frequency_features)
                features_df = pd.concat([features_df, new_row], ignore_index=True)
                prev_index=index

            features_df.insert(0, "Frequency", cycles) 
            features_df=features_df.dropna()
            csv_filename = os.path.join(output_dir, f"{sample[:-4]}.csv")
            features_df.to_csv(csv_filename, index=False)    