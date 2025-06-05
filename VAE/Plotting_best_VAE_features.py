import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
#from Prognostic_criteria import fitness
from Selecting_few_features import fitness_metric
import matplotlib.pyplot as plt

Data_folder = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows"
Data_folder = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data"

# SP_methods = ['CWT_', 'EMD_', 'FFT_', 'HT_', 'FFT_Morteza', 'SPWVD_', 'Time_Domain_']

# Best_Feat = [['Amplitude_Standard_deviation','Amplitude_Mean','Counts_Mean','Duration_Standard_deviation','RMS_Mean','Duration_Mean','RMS_Standard_deviation'],
#              ['RMS_FM4','RMS_Kurtosis','Energy_Root_amplitude','Energy_Mean','Rise-Time_Mean','Energy_Root_mean_squared','Energy_Standard_deviation'],
#              ['Energy_Variance','Energy_Mean','Energy_P13','Counts_Variance','Energy_P10','Duration_Variance','Rise-Time_Mean'],
#              ['Rise-Time_envelope_Central_moment_6th_order','Rise-Time_envelope_Central_moment_5th_order','Rise-Time_envelope_Central_moment_4th_order','Rise-Time_envelope_Central_moment_3rd_order','RMS_envelope_Central_moment_6th_order','Rise-Time_envelope_Mean','Energy_envelope_Mean'],
#              ['Energy_Freq: f2','Energy_Freq: Mean Frequency','Amplitude_Freq: f2', 'Energy_Freq: f13','Duration_Freq: f2','Counts_Freq: f2','Energy_Freq: f10'],
#              ['Amplitude_Standard_deviation','Amplitude_Mean','RMS_Standard_deviation','Amplitude_Skewness','Rise-Time_Standard_deviation','RMS_Mean','Counts_Mean'],
#              ['RMS_Shape_factor','Rise-Time_Central_moment_6th_order','RMS_Clearance_factor','RMS_Central_moment_6th_order','RMS_Impulse_factor','Rise-Time_Central_moment_5th_order','Rise-Time_Central_moment_4th_order']
#              ]

SP_methods = ['FFT_', 'CWT_', 'EMD_', 'HT_', 'FFT_Morteza', 'SPWVD_', 'Time_Domain_']

Best_Feat = [['Energy_Variance','Energy_Mean','Energy_P13','Counts_Variance','Energy_P10','Duration_Variance','Rise-Time_Mean'],
             ['Amplitude_Standard_deviation','Amplitude_Mean','Counts_Mean','Duration_Standard_deviation','RMS_Mean','Duration_Mean','RMS_Standard_deviation'],
             ['RMS_FM4','RMS_Kurtosis','Energy_Root_amplitude','Energy_Mean','Rise-Time_Mean','Energy_Root_mean_squared','Energy_Standard_deviation'],
             ['Rise-Time_envelope_Central_moment_6th_order','Rise-Time_envelope_Central_moment_5th_order','Rise-Time_envelope_Central_moment_4th_order','Rise-Time_envelope_Central_moment_3rd_order','RMS_envelope_Central_moment_6th_order','Rise-Time_envelope_Mean','Energy_envelope_Mean'],
             ['Energy_Freq: f2','Energy_Freq: Mean Frequency','Amplitude_Freq: f2', 'Energy_Freq: f13','Duration_Freq: f2','Counts_Freq: f2','Energy_Freq: f10'],
             ['Amplitude_Standard_deviation','Amplitude_Mean','RMS_Standard_deviation','Amplitude_Skewness','Rise-Time_Standard_deviation','RMS_Mean','Counts_Mean'],
             ['RMS_Shape_factor','Rise-Time_Central_moment_6th_order','RMS_Clearance_factor','RMS_Central_moment_6th_order','RMS_Impulse_factor','Rise-Time_Central_moment_5th_order','Rise-Time_Central_moment_4th_order']
             ]

# SP_methods = ['CWT_']
# Best_Feat = [['Amplitude_Mean', 'Amplitude_Standard_deviation', 'Rise-Time_Mean', 'Rise-Time_Standard_deviation', 'Energy_Mean', 'Energy_Standard_deviation', 'Counts_Mean', 'Counts_Standard_deviation', 'Duration_Mean', 'Duration_Standard_deviation', 'RMS_Mean', 'RMS_Standard_deviation']]
    

# Best_Feat = [['Amplitude_Standard_deviation','Amplitude_Mean','Counts_Mean','Duration_Standard_deviation','RMS_Mean','Duration_Mean','RMS_Standard_deviation'],
#              ['RMS_FM4','RMS_Kurtosis','Energy_Root_amplitude','Energy_Mean','Rise-Time_Mean','Energy_Root_mean_squared','Energy_Standard_deviation'],
#              ['Energy_Variance','Energy_Mean','Energy_P13','Counts_Variance','Energy_P10','Duration_Variance','Rise-Time_Mean'],
#              ['Rise-Time_envelope_Central_moment_6th_order','Rise-Time_envelope_Central_moment_5th_order','Rise-Time_envelope_Central_moment_4th_order','Rise-Time_envelope_Central_moment_3rd_order','RMS_envelope_Central_moment_6th_order','Rise-Time_envelope_Mean','Energy_envelope_Mean'],
#              ['Amplitude_Freq: Mean Frequency','Amplitude_Freq: f2','Amplitude_Freq: f3','Amplitude_Freq: f4','Amplitude_Freq: f5','Amplitude_Freq: f6','Amplitude_Freq: f7'],
#              ['Amplitude_Standard_deviation','Amplitude_Mean','RMS_Standard_deviation','Amplitude_Skewness','Rise-Time_Standard_deviation','RMS_Mean','Counts_Mean'],
#              ['RMS_Shape_factor','Rise-Time_Central_moment_6th_order','RMS_Clearance_factor','RMS_Central_moment_6th_order','RMS_Impulse_factor','Rise-Time_Central_moment_5th_order','Rise-Time_Central_moment_4th_order']
#              ]


def plot_best_features(feature_level_data_base_path, best_features, sp_method_name):
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_samples = len(all_paths)

    sample_data_arr_lst = []
    for i, file in enumerate(all_paths):
        df = pd.read_csv(file)
        df_reduced = df[best_features]
        arr = df_reduced.values
        sample_data_arr_lst.append(arr)

    target_rows = np.max(sample_data_arr_lst[0].shape)
    num_features = len(best_features)

    feature_arr_lst = []
    for i in range(num_features):
        feature_arr = np.zeros((n_samples,target_rows))
        for j in range(n_samples):
            current_sample_data = sample_data_arr_lst[j]
            current_feature_data = current_sample_data[:,i]
            current_feature_data = current_feature_data.T
            feature_arr[j,:] = current_feature_data
        feature_arr_lst.append(feature_arr)

    time = np.linspace(0,100,target_rows)
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']

    current_feature = feature_arr_lst[2] # 0,1,2,4,6
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(time, current_feature[i,:], label=f"Sample{i+1}", color=colors[i])
        plt.xlabel('% of Lifetime')
        plt.ylim(-2.5, 3)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12, 9))
    for i in range(num_features):
        plt.subplot(2,4,i+1)
        current_feature = feature_arr_lst[i]
        for j in range(n_samples):
            plt.plot(time, current_feature[j,:], label=f"Sample{j+1}", color=colors[j])
        plt.xlabel('% of Lifetime')
        plt.title(f'{best_features[i]}')
    #plt.legend(loc='upper left')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.suptitle(f'Best {num_features} features from {sp_method_name} high features')
    plt.show()

for k in range(len(SP_methods)):
    data_base_path = os.path.join(Data_folder, SP_methods[k])
    plot_best_features(data_base_path, Best_Feat[k], sp_method_name=SP_methods[k])
    