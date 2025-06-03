import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from Prognostic_criteria import fitness
import matplotlib.pyplot as plt

resample_sand_save_data = False
compute_fitness = False
load_fitness = False
plot_features = False
sort_features = True

# # FFT Morteza:
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\FFT_Morteza" # r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
# feature_sp_method = 'FFT_Morteza'

save_bar_chart_folder = r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs\AST_Fitness_Graph" # r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs"
save_interp_data_folder = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows"

target_rows_num = 'interp_1200_rows_'

# # Time domain features:
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\Time_Domain_" # r"c:\Users\naomi\OneDrive\Documents\Low_Features\Time_Domain_Features\Time_Domain_Features_500_500_CSV"
# feature_sp_method = 'Time_Domain_'

# FFT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\FFT_" # r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\FFT_Features_interpolated_500_500_CSV"
# feature_sp_method = 'FFT_'
# # # feature_sp_method = 'FFT_New_prog_crit_function'

feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\FFT_new"
feature_sp_method = 'FFT_new'

# # EMD features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\EMD_" # r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\EMD_Features_interpolated_500_500_CSV"
# feature_sp_method = 'EMD_'

# # CWT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\CWT_" # r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\CWT_Features_500_500_CSV"
# feature_sp_method = 'CWT_'

# # HT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\HT_" # r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Hilbert_Features_interpolated_500_500_CSV"
# feature_sp_method = 'HT_'

# # SPWVD features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows\ast_data\SPWVD_" # r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\SPWVD_Features_500_500_CSV"
# feature_sp_method = 'SPWVD_'

# interp_sp_method_data_folder = os.path.join(save_interp_data_folder, feature_sp_method)
# os.makedirs(interp_sp_method_data_folder, exist_ok=True)

bar_chart_output_dir = os.path.join(save_bar_chart_folder, f"Interp_{feature_sp_method}Features_fitness_bar_chart.png")
fitness_score_csv_output_dir = os.path.join(save_bar_chart_folder, f"{feature_sp_method}Features_fitness.csv")
all_paths = glob.glob(feature_level_data_base_path + "/*.csv") # glob.glob(interp_sp_method_data_folder + "/*.csv")
n_filepaths = len(all_paths)

df_sample1 = pd.read_csv(all_paths[0])
expected_cols = list(df_sample1.columns)
expected_cols = expected_cols[1:]

def monotonicity(x):
    """
    Calculate monotonicity (Mo) for health indicator data.
    
    Args:
        x (np.ndarray): Health indicator data with shape (n_samples, timesteps)
        
    Returns:
        float: Monotonicity score between 0 and 1
    """
    M = x.shape[0]  # Number of samples
    Mo_values = []
    
    for j in range(M):
        sample = x[j, :]
        Nj = len(sample)
        numerator = 0
        denominator = 0
        
        for i in range(Nj):
            sum_num = 0
            sum_den = 0
            for p in range(i+1, Nj):
                time_diff = (p - i)  # Assuming regular time intervals
                value_diff = sample[p] - sample[i]
                sum_num += time_diff * np.sign(value_diff)
                sum_den += time_diff
                
            if sum_den != 0:
                numerator += sum_num / sum_den
        
        if Nj > 1:
            Mo_j = np.abs(numerator / (Nj - 1))
            Mo_values.append(Mo_j)
    
    Mo = np.mean(Mo_values) if Mo_values else 0
    return Mo

def prognosability(x):
    """
    Calculate prognosability (Pr) for health indicator data.
    
    Args:
        x (np.ndarray): Health indicator data with shape (n_samples, timesteps)
        
    Returns:
        float: Prognosability score between 0 and 1
    """
    M = x.shape[0]
    if M == 0:
        return 0
    
    # Get final values for all samples
    x_final = x[:, -1]
    mean_final = np.mean(x_final)
    
    # First term: variance of final values
    term1 = np.sqrt(np.mean(np.abs(x_final - mean_final**2)))
    
    # Second term: range of each sample
    x_ranges = np.abs(x[:, -1] - x[:, 0])
    term2 = np.mean(x_ranges)
    
    Pr = np.exp(-(term1 - term2))
    return Pr

def trendability(x):
    """
    Calculate trendability (Tr) for health indicator data.
    
    Args:
        x (np.ndarray): Health indicator data with shape (n_samples, timesteps)
        
    Returns:
        float: Trendability score between 0 and 1
    """
    M = x.shape[0]
    if M < 2:
        return 0
    
    min_corr = float('inf')
    
    # Calculate pairwise correlations
    for j in range(M):
        for k in range(j+1, M):
            # Pearson correlation
            corr_matrix = np.corrcoef(x[j], x[k])
            corr = np.abs(corr_matrix[0, 1])
            
            if corr < min_corr:
                min_corr = corr
    
    Tr = min_corr if min_corr != float('inf') else 0
    return Tr

def fitness_metric(x, a=1, b=1, c=1):
    """
    Calculate composite fitness metric for health indicators.
    
    Args:
        x (np.ndarray): Health indicator data with shape (n_samples, timesteps)
        a, b, c (float): Weighting factors for Mo, Pr, Tr (default 1 each)
        
    Returns:
        float: Fitness score between 0 and 3 (with default weights)
    """
    Mo = monotonicity(x)
    Pr = prognosability(x)
    Tr = trendability(x)
    
    ftn = a*Mo + b*Pr + c*Tr
    error = (a + b + c) / ftn
    return ftn, Mo, Tr, Pr, error

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

def VAE_merge_data_per_timestep_new(low_feature, sample_filenames, expected_cols, target_rows, interp=False):
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
        if interp:
            df_resampled = resample_dataframe(df_freq, target_rows)

            df = df_resampled.values
        else:
            df = df_freq
        all_data.append(df.T)

    # Stack time steps from all samples
    data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
    print(f"✅ Merged data shape: {data.shape}")

    # # Standardize feature-wise (column-wise)
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    # print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

    return all_data, kept_columns

def create_feature_fitness_array(low_feature, all_paths, expected_cols, target_rows, interp=False):
    # List of arrays for samples and shape target_rows x n_features
    samples_data_lst, kept_columns = VAE_merge_data_per_timestep_new(low_feature, all_paths, expected_cols, target_rows, interp)
    #print(f'kept columns: {kept_columns}')

    #print(f'Num features: {len(kept_columns)}')

    num_features = len(kept_columns)

    Feature_fitness_array = np.zeros((num_features, 5))

    for i in range(num_features):
        feature_array = np.zeros((n_filepaths, target_rows))
        for j in range(n_filepaths):
            current_sample_data = samples_data_lst[j]
            #print(f'shape current sample data: {current_sample_data.shape}')
            if isinstance(current_sample_data, pd.DataFrame):
                feature_array[j, :] = current_sample_data.iloc[i, :].values
            else:
                feature_array[j, :] = current_sample_data[i, :]
            #print(f'shape current sample feature data: {current_sample_data[i,:]}')
        ftn, monotonicity, trendability, prognosability, error = fitness(feature_array)
        #ftn, monotonicity, trendability, prognosability, error = fitness_metric(feature_array)
        # Store feature_array in dictionary with feature name as key
        Feature_fitness_array[i,:] = [ftn, monotonicity, trendability, prognosability, error]

    print(Feature_fitness_array)
    return Feature_fitness_array

def create_feature_array_for_plotting(low_feature, all_paths, expected_cols, target_rows):
    # List of arrays for samples and shape target_rows x n_features
    samples_data_lst, kept_columns = VAE_merge_data_per_timestep_new(low_feature, all_paths, expected_cols, target_rows)
    #print(f'kept columns: {kept_columns}')

    #print(f'Num features: {len(kept_columns)}')

    num_features = len(kept_columns)

    Feature_arrays = []

    for i in range(num_features):
        feature_array = np.zeros((n_filepaths, target_rows))
        for j in range(n_filepaths):
            current_sample_data = samples_data_lst[j]
            #print(f'shape current sample data: {current_sample_data.shape}')
            if isinstance(current_sample_data, pd.DataFrame):
                feature_array[j, :] = current_sample_data.iloc[i, :].values
            else:
                feature_array[j, :] = current_sample_data[i, :]
            #print(f'shape current sample feature data: {current_sample_data[i,:]}')
        #ftn, monotonicity, trendability, prognosability, error = fitness(feature_array)
        Feature_arrays.append(feature_array)
        # Store feature_array in dictionary with feature name as key
        #Feature_fitness_array[i,:] = [ftn, monotonicity, trendability, prognosability, error]

        time = np.linspace(0,100,target_rows)
        plt.plot(time, feature_array[0,:])
        plt.xlabel(f'% of lifetime')
        plt.ylabel(f'Feature {i+1}, sample 1')
        plt.show()


    #print(Feature_fitness_array)
    return Feature_arrays

# if resample_sand_save_data:
#     all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
#     n_filepaths = len(all_paths)

#     df_sample1 = pd.read_csv(all_paths[0])
#     expected_cols = list(df_sample1.columns)
#     expected_cols = expected_cols[1:]

#     target_rows = 1200

#     for i in range(n_filepaths):
#         csv_output_dir = os.path.join(interp_sp_method_data_folder, f"Sample_{i+1}Features_{target_rows_num}.csv")
#         df = pd.read_csv(all_paths[i])
#         resampled_df = resample_dataframe(df, target_rows)
#         print(resampled_df.head)
#         resampled_df.to_csv(csv_output_dir, index=False)
#         print(f" \n Resampled DataFrame saved to {csv_output_dir}")



if compute_fitness:
    #low_features_lst = [['Amplitude_Time', 'Amplitude_Freq', 'Amplitude_Physics'],['Rise-time_Time', 'Rise-time_Freq', 'Rise-time_Physics'], ['Energy_Time', 'Energy_Freq', 'Energy_Physics'], ['Counts_Time', 'Counts_Freq', 'Counts_Physics'], ['Duration_Time', 'Duration_Freq', 'Duration_Physics'], ['RMS_Time', 'RMS_Freq', 'RMS_Physics']]
    low_features_lst = [['Amplitude_'],['Rise-Time_','Rise-time_'], ['Energy_'], ['Counts_'], ['Duration_'], ['RMS_']]

    target_rows = 1200

    feature_fitness_scores_lst = []
    for k in range(6):
        low_feature_current = low_features_lst[k]
        low_feature_fitness_arr = create_feature_fitness_array(low_feature_current, all_paths, expected_cols, target_rows, interp=False)
        feature_fitness_scores_lst.append(low_feature_fitness_arr)

    all_features_fitness_scores = np.vstack(feature_fitness_scores_lst)

    # Indices of fitness scores in ascending order:
    total_fitness_only = all_features_fitness_scores[:,0]
    feature_fitness_descending_indices = np.argsort(-total_fitness_only)

    print(f'\n Features with highest fitness scores: \n Feature name \t \t Fitness \t Monotonicity \t Trendability \t Prognosability \n')
    for i in range(7):
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
    num_features = len(feature_names)
    fitness_scores_arr = fitness_scores_df.to_numpy()

    all_features_fitness_scores = fitness_scores_arr.T

    # Indices of fitness scores in ascending order:
    total_fitness_only = fitness_scores_arr[0,:]
    feature_fitness_descending_indices = np.argsort(-total_fitness_only)

    print(f'\n Features with highest fitness scores: \n Feature name \t \t Fitness \t Monotonicity \t Trendability \t Prognosability \n')
    for i in range(7):
        feature_idx = feature_fitness_descending_indices[i]
        print(f' {feature_names[feature_idx]} \t \t {all_features_fitness_scores[feature_idx,0]} \t {all_features_fitness_scores[feature_idx,1]} \t {all_features_fitness_scores[feature_idx,2]} \t {all_features_fitness_scores[feature_idx,3]} \n ')

    MoTrPr_scores = fitness_scores_arr[1:4,:]
    MoTrPr_scores_T = MoTrPr_scores.T

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
        plt.bar(x, MoTrPr_scores_T[:, i], bottom=bottom, color=colors[i], label=stack_labels[i])
        bottom += MoTrPr_scores_T[:, i]

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

if plot_features:
    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    best_features = ['Time (cycle)', 'Energy_Variance', 'Energy_Mean', 'Energy_P13', 'Counts_Variance', 'Energy_P10']
    features_data, kept_cols = VAE_merge_data_per_timestep_new(best_features, all_paths, expected_cols, target_rows)
    print(f'shape data for one samples feature wise data: {features_data[0].shape}')
    print(f'kept cols: {kept_cols}')

    num_features = len(kept_cols)

    Feature_fitness_array = np.zeros((num_features, 5))

    feature_wise_data = []

    for i in range(num_features):
        feature_array = np.zeros((n_filepaths, target_rows))
        for j in range(n_filepaths):
            current_sample_data = features_data[j]
            #print(f'shape current sample data: {current_sample_data.shape}')
            if isinstance(current_sample_data, pd.DataFrame):
                feature_array[j, :] = current_sample_data.iloc[i, :].values
            else:
                feature_array[j, :] = current_sample_data[i, :]
        ftn, monotonicity, trendability, prognosability, error = fitness(feature_array)
        #ftn, monotonicity, trendability, prognosability, error = fitness_metric(feature_array)
        feature_wise_data.append(feature_array)
        # Store feature_array in dictionary with feature name as key
        Feature_fitness_array[i,:] = [ftn, monotonicity, trendability, prognosability, error]

    time = np.linspace(0,100,target_rows)
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    for i in range(num_features):
        plt.subplot(2,3,i+1)
        features = feature_wise_data[0]
        for j in range(n_filepaths):
            plt.plot(time, features[j,:], label=f'Sample{j+1}', color=colors[j])
        plt.xlabel('% of Lifetime')
        plt.ylabel(f'{kept_cols[i]}')
    plt.legend()
    plt.title(f'Best {num_features} features from {feature_sp_method} high features')
    plt.show()

    Feature_arrays = create_feature_array_for_plotting(best_features, all_paths, expected_cols, target_rows)

# Sorting csv files with fitness per feature
def sort_columns_by_first_row(csv_path):
    df = pd.read_csv(csv_path)
    sorted_cols = df.iloc[0].sort_values(ascending=False).index.tolist()
    df_sorted = df[sorted_cols]

    df_sorted.to_csv(csv_path, index=False)
    print('Sorted CSV file')
    

if sort_features:
    ast_folder = r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs\AST_Fitness_Graph"
    non_ast_folder = r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs"
    filepaths = glob.glob(non_ast_folder + "/*.csv") 
    for file in filepaths:
        sort_columns_by_first_row(file)




# import glob
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import os
# import numpy as np
# from Prognostic_criteria import fitness
# import matplotlib.pyplot as plt

# compute_fitness = True
# load_fitness = True

# # # FFT Morteza:
# # feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
# # feature_sp_method = 'FFT_Morteza'

# save_bar_chart_folder = r"c:\Users\naomi\OneDrive\Documents\High_Features_Fitness_Graphs"

# # # Time domain features:
# # feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Low_Features\Time_Domain_Features\Time_Domain_Features_500_500_CSV"
# # feature_sp_method = 'Time_Domain_'

# # # FFT features
# # feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\FFT_Features_interpolated_500_500_CSV"
# # feature_sp_method = 'FFT_'

# # # EMD features
# # feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\EMD_Features_interpolated_500_500_CSV"
# # feature_sp_method = 'EMD_'

# # CWT features
# feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\CWT_Features_500_500_CSV"
# feature_sp_method = 'CWT_'

# # # HT features
# # feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Hilbert_Features_interpolated_500_500_CSV"
# # feature_sp_method = 'HT_'

# # # SPWVD features
# # feature_level_data_base_path = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\SPWVD_Features_500_500_CSV"
# # feature_sp_method = 'SPWVD_'


# bar_chart_output_dir = os.path.join(save_bar_chart_folder, f"{feature_sp_method}Features_fitness_bar_chart.png")
# fitness_score_csv_output_dir = os.path.join(save_bar_chart_folder, f"{feature_sp_method}Features_fitness.csv")
# all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
# n_filepaths = len(all_paths)

# df_sample1 = pd.read_csv(all_paths[0])
# expected_cols = list(df_sample1.columns)
# expected_cols = expected_cols[1:]

# def resample_dataframe(df, target_rows):
#     """Resample each column in a DataFrame to target number of rows."""
#     resampled_data = {}
#     for col in df.columns:
#         original = df[col].values
#         x_original = np.linspace(0, 1, len(original))
#         x_target = np.linspace(0, 1, target_rows)
#         interpolated = np.interp(x_target, x_original, original)
#         resampled_data[col] = interpolated
#     return pd.DataFrame(resampled_data)

# def VAE_merge_data_per_timestep_new(low_feature, sample_filenames, expected_cols, target_rows):
#     """
#     Load multiple AE data files, resample each to have target_rows rows via interpolation.
#     Combine all resampled data into a single dataframe: one row per time step.
#     Shape: (n_samples * target_rows, n_features)
#     """
#     all_data = []

#     for path in sample_filenames:
#         print(f"Reading and resampling: {os.path.basename(path)}")

#         df = pd.read_csv(path)

#         if 'Time' in df.columns:
#             df = df.drop(columns=['Time (Cycle)'])
#         if 'Unnamed: 0' in df.columns:
#             df = df.drop(columns=['Unnamed: 0'])

#         #df_test = df_test.drop(df_test.columns[0], axis=1)

#         missing = [col for col in expected_cols if col not in df.columns]
#         if missing:
#             raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
#         df = df[expected_cols]

#         include = low_feature
#         exclude = ['Cycles']
#         #exclude = ['Rise', 'Energy', 'Duration', 'RMS', 'Count']

#         pattern = f"({'|'.join(include)})(?!.*({'|'.join(exclude)}))"

#         df_freq = df.loc[:, df.columns.str.contains(pattern, regex=True)]
#         kept_columns = df_freq.columns.tolist()
#         df_resampled = resample_dataframe(df_freq, target_rows)

#         df = df_resampled.values
#         all_data.append(df.T)

#     # Stack time steps from all samples
#     data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
#     print(f"✅ Merged data shape: {data.shape}")

#     # # Standardize feature-wise (column-wise)
#     # scaler = StandardScaler()
#     # data_scaled = scaler.fit_transform(data)
#     # print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

#     return all_data, kept_columns

# def create_feature_fitness_array(low_feature, all_paths, expected_cols, target_rows):
#     # List of arrays for samples and shape target_rows x n_features
#     samples_data_lst, kept_columns = VAE_merge_data_per_timestep_new(low_feature, all_paths, expected_cols, target_rows)
#     #print(f'kept columns: {kept_columns}')

#     #print(f'Num features: {len(kept_columns)}')

#     num_features = len(kept_columns)

#     Feature_fitness_array = np.zeros((num_features, 5))

#     for i in range(num_features):
#         feature_array = np.zeros((n_filepaths, target_rows))
#         for j in range(n_filepaths):
#             current_sample_data = samples_data_lst[j]
#             #print(f'shape current sample data: {current_sample_data.shape}')
#             feature_array[j,:] = current_sample_data[i,:]
#             #print(f'shape current sample feature data: {current_sample_data[i,:]}')
#         ftn, monotonicity, trendability, prognosability, error = fitness(feature_array)
#         # Store feature_array in dictionary with feature name as key
#         Feature_fitness_array[i,:] = [ftn, monotonicity, trendability, prognosability, error]

#     print(Feature_fitness_array)
#     return Feature_fitness_array

# if compute_fitness:
#     #low_features_lst = [['Amplitude_Time', 'Amplitude_Freq', 'Amplitude_Physics'],['Rise-time_Time', 'Rise-time_Freq', 'Rise-time_Physics'], ['Energy_Time', 'Energy_Freq', 'Energy_Physics'], ['Counts_Time', 'Counts_Freq', 'Counts_Physics'], ['Duration_Time', 'Duration_Freq', 'Duration_Physics'], ['RMS_Time', 'RMS_Freq', 'RMS_Physics']]
#     low_features_lst = [['Amplitude_'],['Rise-Time_'], ['Energy_'], ['Counts_'], ['Duration_'], ['RMS_']]
#     target_rows = 300

#     feature_fitness_scores_lst = []
#     for k in range(6):
#         low_feature_current = low_features_lst[k]
#         low_feature_fitness_arr = create_feature_fitness_array(low_feature_current, all_paths, expected_cols, target_rows)
#         feature_fitness_scores_lst.append(low_feature_fitness_arr)

#     all_features_fitness_scores = np.vstack(feature_fitness_scores_lst)

#     # Indices of fitness scores in ascending order:
#     total_fitness_only = all_features_fitness_scores[:,0]
#     feature_fitness_descending_indices = np.argsort(-total_fitness_only)

#     print(f'\n Features with highest fitness scores: \n Feature name \t \t Fitness \t Monotonicity \t Trendability \t Prognosability \n')
#     for i in range(5):
#         feature_idx = feature_fitness_descending_indices[i]
#         print(f' {expected_cols[feature_idx]} \t \t {all_features_fitness_scores[feature_idx,0]} \t {all_features_fitness_scores[feature_idx,1]} \t {all_features_fitness_scores[feature_idx,2]} \t {all_features_fitness_scores[feature_idx,3]} \n ')


#     # Storing Features and fitnesses as CSV file:
#     all_feat_fitness_scores_array_T = all_features_fitness_scores.T
#     df_fitness_scores = pd.DataFrame(all_feat_fitness_scores_array_T, columns=expected_cols)
#     df_fitness_scores.to_csv(fitness_score_csv_output_dir, index=False)

#     print(f" \n DataFrame containing fitness scores saved to {fitness_score_csv_output_dir}")


# if load_fitness:
#     # Plotting data
#     num_features = len(expected_cols)
#     fitness_scores_df = pd.read_csv(fitness_score_csv_output_dir)
#     feature_names = fitness_scores_df.columns.tolist()
#     fitness_scores_arr = fitness_scores_df.to_numpy()
#     fitness_scores = fitness_scores_arr[1:4,:]
#     fitness_scores = fitness_scores.T

#     #fitness_scores = all_features_fitness_scores[:,1:4]  # shape (num_features, 3)
#     #print(f'shape fitness scores matrix: {fitness_scores.shape}')
    
#     stack_labels = ['Monotonicity', 'Trendability', 'Prognosability']

#     # Set positions for bars on x-axis
#     x = np.arange(num_features)

#     # Bar colors
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

#     # Initialize bottom positions for stacking
#     bottom = np.zeros(num_features)

#     # Create figure with wider size
#     plt.figure(figsize=(num_features * 0.15, 6))  # scale 0.2 per feature — adjust as needed

#     # Bar width (reduce slightly for spacing)
#     #bar_width = 0.75  # default is 0.8, reduce to e.g. 0.6 for more spacing

#     # Create stacked bars
#     for i in range(3):
#         plt.bar(x, fitness_scores[:, i], bottom=bottom, color=colors[i], label=stack_labels[i])
#         bottom += fitness_scores[:, i]

#     # Set x-ticks: evenly spaced, no overlap
#     plt.xticks(ticks=x, labels=feature_names, rotation=90, fontsize=6, ha='center')
#     # Use whole x-axis
#     plt.xlim(-0.9, num_features + 0.9)

#     # Add labels, legend, and title
#     plt.ylabel('Fitness')
#     plt.title(f'Fitness Scores for High Features from {feature_sp_method} Data')
#     plt.legend()

#     # Better spacing
#     plt.subplots_adjust(bottom=0.3)  # more space under x labels
#     plt.grid(axis='y', linestyle='--', alpha=0.5)

#     # Tight layout and show
#     plt.tight_layout()
#     plt.savefig(bar_chart_output_dir, dpi=300)
#     plt.show()

#     print(f'Number of features plotted: {len(feature_names)} \t Number of features given initially: {num_features}')

