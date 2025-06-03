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

def time_frequency_domain_features(data):
    """
        Extracts time-frequency domain features from sensor data.

        Parameters:
        - data (1D array): Array containing data of a low-level feature.    

        Returns:
        - FT_features (1D array): Array containing time-frequency domain features.
    """
    #np 1D array main features
    FT_features = np.empty(4) 

    #Mean
    FT_features[0] = np.mean(data)

    #Standard Deviation
    FT_features[1] = np.std(data)

    #Skewness
    FT_features[2] = stats.skew(data)

    #Kurtosis
    FT_features[3] = stats.kurtosis(data)

    return FT_features


def extract_time_statistical_features(cycle_length, input_dir, output_dir):

    """
        Extracts time domain features from sensor data.

        Parameters:
            - cycle_length (int): Length of the cycle.
            - dir (string): Directory of the samples.

        Output:
            - CSV files containing time domain features.
    """

    vals=[]
    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root, sample)
            
            df = pd.read_csv(file)
            print("Processing:", sample) 
            end_val=math.ceil(df["Time (cycle)"].iloc[-1] / cycle_length) * cycle_length
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
                index=df["Time (cycle)"][df["Time (cycle)"]<=cycle].index[-1] 
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
            print("Processing: ", sample)
            #rows=math.ceil(endval["Endval"].iloc[int(sample[6:8])-1] / cycle_length)
            #print(rows) 
            #cycles=np.linspace(rows/len(df['Frequency (Hz)']), df['Frequency (Hz)'].iloc[-1], rows-1)  
            df.insert(0, "Time (cycle)", df["bucket"]*cycle_length+1)
            df.drop(columns=["bucket"], inplace=True) 
            end_val=math.ceil(df["Time (cycle)"].iloc[-1] / cycle_length) * cycle_length 
            cycles=np.arange(cycle_length, end_val+1, cycle_length) 
            prev_index=-1

            #define the time domain features dataframe
            base_frequency_features=['Mean','Variance','Skewness','Kurtosis','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14']            
            frequency_features=[]
            for column in df.iloc[:, 2:]:
                frequency_features.append([str(column)+"_"+i for i in base_frequency_features])
            frequency_features=np.concatenate(frequency_features)
            features_df=pd.DataFrame(columns=frequency_features) 

            os.makedirs(output_dir, exist_ok=True) 

            for cycle in cycles:
                current_row=[]
                index=df["Time (cycle)"][df["Time (cycle)"]<=cycle].index[-1]  
                for column in df.iloc[:, 2:]: 
                    try:
                        current_row.append(frequency_domain_features(df[column][prev_index+1:index+1].to_numpy().flatten()))
                    except:
                        current_row.append([np.nan]*14)
                current_row=np.concatenate(current_row)
                new_row=pd.DataFrame([current_row], columns=frequency_features)
                features_df = pd.concat([features_df, new_row], ignore_index=True)
                #print(features_df)
                prev_index=index

            #features_df.insert(0, "Time (cycle)", cycles) 
            #features_df=features_df.dropna()
            csv_filename = os.path.join(output_dir, f"{sample[6:8]}.csv")
            features_df.insert(0, "Time (cycle)", cycles)
            features_df.to_csv(csv_filename, index=False)   

        return features_df 

def new_fft_feature_extract(input_dir, output_dir):
    base_frequency_features=['Mean','Variance','Skewness','Kurtosis','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14']            
    frequency_features=[]
    columns=["Amplitude", "Rise-Time", "Energy", "Counts", "Duration", "RMS"]
    for column in columns:
        frequency_features.append([str(column)+"_"+i for i in base_frequency_features])
    frequency_features=np.concatenate(frequency_features)

    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root, sample)
            print("Processing: ", sample)
            df=pd.read_csv(file)
            cols=df.iloc[:,0]
            df=df.T
            df.columns=cols 
            df.drop(df.index[0], inplace=True)
            new_df=pd.DataFrame(columns=frequency_features)
            new_df=new_df.T
            # df.insert(0, "bucket", df.index)
            # df.index=np.arange(0, len(df))
            indexlist=np.arange(0, 7*256, 256)
            for col in df.columns:
                rowlist=[]
                for index in range(6):
                    rowlist.extend(frequency_domain_features(df[col].values[indexlist[index]:indexlist[index+1]]))
                #rowlist=
                new_df[col]=rowlist
            new_df=new_df.T
            new_df.index.name="Time (cycle)"
            filename=os.path.join(output_dir, f"Sample{sample[6:8]}.csv")
            new_df.to_csv(filename, index=True)

def extract_time_frequency_statistical_features(input_dir, output_dir):
    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root, sample)
            print("Processing: ", sample)
            
            df = pd.read_csv(file) 
            cycles=df["Frequency (Hz)"].to_numpy()
            cycles=sorted(list(set(cycles)))
            prev_index=-1

            #define the time domain features dataframe
            base_time_frequency_features=['Mean','Standard_deviation','Skewness','Kurtosis']            
            time_frequency_features=[]
            for column in df.iloc[:, 2:]:
                time_frequency_features.append([str(column)+"_"+i for i in base_time_frequency_features])
            time_frequency_features=np.concatenate(time_frequency_features)
            features_df=pd.DataFrame(columns=time_frequency_features)

            os.makedirs(output_dir, exist_ok=True) 

            for cycle in cycles:
                current_row=[]
                index=df["Frequency (Hz)"][df["Frequency (Hz)"]<=cycle].index[-1] 
                for column in df.iloc[:, 2:]: 
                    try:
                        current_row.append(time_frequency_domain_features(df[column][prev_index+1:index+1].to_numpy().flatten()))
                    except:
                        current_row.append([np.nan]*4)
                current_row=np.concatenate(current_row)
                new_row=pd.DataFrame([current_row], columns=time_frequency_features)
                features_df = pd.concat([features_df, new_row], ignore_index=True)
                #print(features_df)
                prev_index=index

            features_df.insert(0, "Frequency (Hz)", cycles) 
            csv_filename = os.path.join(output_dir, f"{sample[:-4]}.csv")
            features_df.to_csv(csv_filename, index=False)   

        #return features_df 
    
def changenames(input_dir):
    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file=os.path.join(root,sample)
            df=pd.read_csv(file)
            df=df.rename(columns={'time': 'Time (cycle)'}) #'Frequency': 'Frequency (Hz)',
            df.to_csv(os.path.join(input_dir, sample), index=False)

def transform_SPWD(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, samples in os.walk(input_dir):
        for dir in dirs:
            new_df=pd.DataFrame()
            print("Processing: ", dir)
            for root_, dirs_, attributes in os.walk(os.path.join(input_dir, dir)):
                for attribute in attributes: 
                    file=os.path.join(root_, attribute)
                    attribute=attribute[:attribute.index("_")]
                    print("Processing: ", attribute)
                    df=pd.read_csv(file)
                    matrix_df=df.drop("Time_cycle", axis=1).to_numpy()
                    matrix_df=matrix_df.flatten(order='F').reshape(-1, 1).flatten()
                    new_df[attribute]=matrix_df

                time_values=df['Time_cycle']
                frequency_values=pd.DataFrame(df.keys()[1:]).to_numpy()
                Time = np.tile(time_values, len(frequency_values))  
                Frequency = np.repeat(frequency_values, len(time_values))
                        
                new_df.insert(0, "Time (cycle)", pd.DataFrame(Time))
                new_df.insert(0, "Frequency (Hz)", pd.DataFrame(Frequency))
                filename=os.path.join(output_dir, f"{dir[:8]}.csv")
                print("Saving: ", filename)
                new_df.to_csv(filename, index=False)    
   
#transform_SPWD(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Sample03_spwvd", r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Sample03_spwvd_transformed.csv")

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

#speed=343
#wavelength=500
#frequency=speed/wavelength
#print("Frequency: ", frequency)
#extract_time_frequency_statistical_features(500, r"C:\Users\attil\Downloads\Amplitude_mean_tfr_array.csv")