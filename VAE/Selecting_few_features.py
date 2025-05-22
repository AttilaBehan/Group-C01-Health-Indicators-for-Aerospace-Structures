import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from Prognostic_criteria import fitness
import matplotlib.pyplot as plt

compute_fitness = True
load_fitness = True

# # FFT Morteza:
# feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
# feature_sp_method = 'FFT_Morteza'

save_bar_chart_folder = r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs"

# # Time domain features:
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Low_Features\Time_Domain_Features\Time_Domain_Features_500_500_CSV"
# feature_sp_method = 'Time_Domain_'

# # FFT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\FFT_Features_interpolated_500_500_CSV"
# feature_sp_method = 'FFT_'

# # EMD features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\EMD_Features_interpolated_500_500_CSV"
# feature_sp_method = 'EMD_'

# CWT features
feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\CWT_Features_500_500_CSV"
feature_sp_method = 'CWT_'

# # HT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Hilbert_Features_interpolated_500_500_CSV"
# feature_sp_method = 'HT_'

# # SPWVD features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\SPWVD_Features_500_500_CSV"
# feature_sp_method = 'SPWVD_'


bar_chart_output_dir = os.path.join(save_bar_chart_folder, f"{feature_sp_method}Features_fitness_bar_chart.png")
fitness_score_csv_output_dir = os.path.join(save_bar_chart_folder, f"{feature_sp_method}Features_fitness.csv")
all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
n_filepaths = len(all_paths)

df_sample1 = pd.read_csv(all_paths[0])
expected_cols = list(df_sample1.columns)
expected_cols = expected_cols[1:]

def resample_dataframe(df, target_rows):
    """Resample each column in a DataFrame to target number of rows."""
    resampled_data = {}
    for col in df.columns:
        original = df[col].values
        x_original = np.linspace(0, 1, len(original))
        x_target = np.linspace(0, 1, target_rows)
        interpolated = np.interp(x_target, x_original, original)
        resampled_data[col] = interpolated
    return pd.DataFrame(resampled_data)

def VAE_merge_data_per_timestep_new(low_feature, sample_filenames, expected_cols, target_rows):
    """
    Load multiple AE data files, resample each to have target_rows rows via interpolation.
    Combine all resampled data into a single dataframe: one row per time step.
    Shape: (n_samples * target_rows, n_features)
    """
    all_data = []

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")

        df = pd.read_csv(path)

        if 'Time' in df.columns:
            df = df.drop(columns=['Time (Cycle)'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        #df_test = df_test.drop(df_test.columns[0], axis=1)

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
        df = df[expected_cols]

        include = low_feature
        exclude = ['Cycles']
        #exclude = ['Rise', 'Energy', 'Duration', 'RMS', 'Count']

        pattern = f"({'|'.join(include)})(?!.*({'|'.join(exclude)}))"

        df_freq = df.loc[:, df.columns.str.contains(pattern, regex=True)]
        kept_columns = df_freq.columns.tolist()
        df_resampled = resample_dataframe(df_freq, target_rows)

        df = df_resampled.values
        all_data.append(df.T)

    # Stack time steps from all samples
    data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
    print(f"✅ Merged data shape: {data.shape}")

    # # Standardize feature-wise (column-wise)
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    # print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

    return all_data, kept_columns

def create_feature_fitness_array(low_feature, all_paths, expected_cols, target_rows):
    # List of arrays for samples and shape target_rows x n_features
    samples_data_lst, kept_columns = VAE_merge_data_per_timestep_new(low_feature, all_paths, expected_cols, target_rows)
    #print(f'kept columns: {kept_columns}')

    #print(f'Num features: {len(kept_columns)}')

    num_features = len(kept_columns)

    Feature_fitness_array = np.zeros((num_features, 5))

    for i in range(num_features):
        feature_array = np.zeros((n_filepaths, target_rows))
        for j in range(n_filepaths):
            current_sample_data = samples_data_lst[j]
            #print(f'shape current sample data: {current_sample_data.shape}')
            feature_array[j,:] = current_sample_data[i,:]
            #print(f'shape current sample feature data: {current_sample_data[i,:]}')
        ftn, monotonicity, trendability, prognosability, error = fitness(feature_array)
        # Store feature_array in dictionary with feature name as key
        Feature_fitness_array[i,:] = [ftn, monotonicity, trendability, prognosability, error]

    print(Feature_fitness_array)
    return Feature_fitness_array

if compute_fitness:
    #low_features_lst = [['Amplitude_Time', 'Amplitude_Freq', 'Amplitude_Physics'],['Rise-time_Time', 'Rise-time_Freq', 'Rise-time_Physics'], ['Energy_Time', 'Energy_Freq', 'Energy_Physics'], ['Counts_Time', 'Counts_Freq', 'Counts_Physics'], ['Duration_Time', 'Duration_Freq', 'Duration_Physics'], ['RMS_Time', 'RMS_Freq', 'RMS_Physics']]
    low_features_lst = [['Amplitude_'],['Rise-Time_'], ['Energy_'], ['Counts_'], ['Duration_'], ['RMS_']]
    target_rows = 300

    feature_fitness_scores_lst = []
    for k in range(6):
        low_feature_current = low_features_lst[k]
        low_feature_fitness_arr = create_feature_fitness_array(low_feature_current, all_paths, expected_cols, target_rows)
        feature_fitness_scores_lst.append(low_feature_fitness_arr)

    all_features_fitness_scores = np.vstack(feature_fitness_scores_lst)

    # Indices of fitness scores in ascending order:
    total_fitness_only = all_features_fitness_scores[:,0]
    feature_fitness_descending_indices = np.argsort(-total_fitness_only)

    print(f'\n Features with highest fitness scores: \n Feature name \t \t Fitness \t Monotonicity \t Trendability \t Prognosability \n')
    for i in range(5):
        feature_idx = feature_fitness_descending_indices[i]
        print(f' {expected_cols[feature_idx]} \t \t {all_features_fitness_scores[feature_idx,0]} \t {all_features_fitness_scores[feature_idx,1]} \t {all_features_fitness_scores[feature_idx,2]} \t {all_features_fitness_scores[feature_idx,3]} \n ')


    # Storing Features and fitnesses as CSV file:
    all_feat_fitness_scores_array_T = all_features_fitness_scores.T
    df_fitness_scores = pd.DataFrame(all_feat_fitness_scores_array_T, columns=expected_cols)
    df_fitness_scores.to_csv(fitness_score_csv_output_dir, index=False)

    print(f" \n DataFrame containing fitness scores saved to {fitness_score_csv_output_dir}")


if load_fitness:
    # Plotting data
    num_features = len(expected_cols)
    fitness_scores_df = pd.read_csv(fitness_score_csv_output_dir)
    feature_names = fitness_scores_df.columns.tolist()
    fitness_scores_arr = fitness_scores_df.to_numpy()
    fitness_scores = fitness_scores_arr[1:4,:]
    fitness_scores = fitness_scores.T

    #fitness_scores = all_features_fitness_scores[:,1:4]  # shape (num_features, 3)
    #print(f'shape fitness scores matrix: {fitness_scores.shape}')
    
    stack_labels = ['Monotonicity', 'Trendability', 'Prognosability']

    # Set positions for bars on x-axis
    x = np.arange(num_features)

    # Bar colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Initialize bottom positions for stacking
    bottom = np.zeros(num_features)

    # Create figure with wider size
    plt.figure(figsize=(num_features * 0.15, 6))  # scale 0.2 per feature — adjust as needed

    # Bar width (reduce slightly for spacing)
    #bar_width = 0.75  # default is 0.8, reduce to e.g. 0.6 for more spacing

    # Create stacked bars
    for i in range(3):
        plt.bar(x, fitness_scores[:, i], bottom=bottom, color=colors[i], label=stack_labels[i])
        bottom += fitness_scores[:, i]

    # Set x-ticks: evenly spaced, no overlap
    plt.xticks(ticks=x, labels=feature_names, rotation=90, fontsize=6, ha='center')
    # Use whole x-axis
    plt.xlim(-0.9, num_features + 0.9)

    # Add labels, legend, and title
    plt.ylabel('Fitness')
    plt.title(f'Fitness Scores for High Features from {feature_sp_method} Data')
    plt.legend()

    # Better spacing
    plt.subplots_adjust(bottom=0.3)  # more space under x labels
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Tight layout and show
    plt.tight_layout()
    plt.savefig(bar_chart_output_dir, dpi=300)
    plt.show()

    print(f'Number of features plotted: {len(feature_names)} \t Number of features given initially: {num_features}')

