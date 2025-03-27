import numpy as np
from scipy import stats
import pandas as pd
import math
import os
import warnings

warnings.simplefilter(action='ignore')

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
    sample_time=sample_df["Time"].to_numpy()
    sample_amplitude = sample_df["Amplitude"].to_numpy()
    sample_risetime = sample_df["Rise-Time"].to_numpy()
    sample_energy = sample_df["Energy"].to_numpy()
    sample_counts = sample_df["Counts"].to_numpy()
    sample_duration = sample_df["Duration"].to_numpy()
    sample_rms = sample_df["RMS"].to_numpy()

    return sample_time, sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms

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

def time_windows_extract(cycle_length, dir):

    """
        Extracts time domain features from sensor data.

        Parameters:
            - cycle_length (int): Length of the cycle.
            - dir (string): Directory of the samples.

        Output:
            - CSV files containing time domain features.
    """

    for root, dirs, samples in os.walk(dir):
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

            output_dir = r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Time_Domain_Features_500_500_CSV"
            os.makedirs(output_dir, exist_ok=True)

            for cycle in cycles:
                current_row=[]
                index=df["Time"][df["Time"]<=cycle].index[-1] 
                for column in df.iloc[:, 1:]:
                    try:
                        current_row.append(Time_Domain_Features(df[column][prev_index+1:index+1].to_numpy().flatten()))
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

time_windows_extract(500, r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Low_Features_500_500_CSV")