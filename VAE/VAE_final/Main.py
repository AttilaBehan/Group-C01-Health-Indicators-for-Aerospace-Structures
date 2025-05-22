import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.interpolate as interp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from Prog_crit import fitness, scale_exact
from File_handling import VAE_merge_data_per_timestep, resample_dataframe
from Train import VAE_train
from Bayesian_optimization import VAE_optimize_hyperparameters
from Plot_function import plot_results
from Optuna_optimization import optimize_hyperparameters_optuna
from Model_architecture import VAE, VAE_Seed
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tqdm import tqdm
import glob
# from pathlib import Path

# # Get the parent directory of the current file's directory
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

#vae_seed = 42
random.seed(VAE_Seed.vae_seed)
tf.random.set_seed(VAE_Seed.vae_seed)
np.random.seed(VAE_Seed.vae_seed)

# Training_data_folder
train_paths_folder = r"VAE_AE_DATA"
# Get a list of CSV file paths in the folder
train_paths = glob.glob(train_paths_folder + "/*.csv")

df_sample1 = pd.read_csv(train_paths[0])
expected_cols = list(df_sample1.columns)
expected_cols = expected_cols[1:]

target_rows = 300

''' STRUCTURE OF THE CODE:

    1. Functions defined: 
    2. Seeds defined'''

''' Functions needed:
    
    - Function to read data
    - VAE model
    - Training VAE
    - Calculatating HI and fitness score
    - Optimizing hyperparameters (uses training function)
    - Applying optimization to all data
    - Storing hyperparameters
    - Storing optimized model'''

''' FUNCTIONS:'''

# VAE merge data function and inputs for current dataset:
target_rows = 300
num_features=201
hidden_2 = 10




''' Extra callback function for hyperparameter optimization progress report'''
def enhanced_callback(res):
    print(f"Iteration {len(res.x_iters)} - Best error: {res.fun:.4f}")

# Define space for Bayesian hyperparameter optimiation 
space = [
        Integer(70, 140, name='hidden_1'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 1000, name='epochs'),
        Integer(8, 32, name="hidden_2"),
        Real(0.05, 0.6, name='reloss_coeff'),
        Real(0.1, 1.0, name='klloss_coeff'),
        Real(2.6, 4, name='moloss_coeff')
    ]

# Use the decorator to automatically convert parameters to keyword arguments
# @use_named_args(space) # converts positional arguments to keyword arguments

train_once = False
if __name__ == "__main__" and train_once:
    # Variables:
    target_rows=300
    hidden_1 = 50
    batch_size = 300
    learning_rate = 0.005
    epochs = 550
    hidden_2 = 10
    reloss_coeff = 0.075
    klloss_coeff = 1.6
    moloss_coeff = 2.8

    #expected_cols = ['Amplitude_Time: Mean','Amplitude_Time: Standard Deviation','Amplitude_Time: Root Amplitude','Amplitude_Time: Root Mean Square','Amplitude_Time: Root Sum of Squares','Amplitude_Time: Peak','Amplitude_Time: Skewness','Amplitude_Time: Kurtosis','Amplitude_Time: Crest factor','Amplitude_Time: Clearance factor','Amplitude_Time: Shape factor','Amplitude_Time: Impulse factor','Amplitude_Time: Maximum to minimum difference','Amplitude_Time: FM4','Amplitude_Time: Median','Energy_Time: Mean','Energy_Time: Standard Deviation','Energy_Time: Root Amplitude','Energy_Time: Root Mean Square','Energy_Time: Root Sum of Squares','Energy_Time: Peak','Energy_Time: Skewness','Energy_Time: Kurtosis','Energy_Time: Crest factor','Energy_Time: Clearance factor','Energy_Time: Shape factor','Energy_Time: Impulse factor','Energy_Time: Maximum to minimum difference','Energy_Time: Median']
    #expected_cols_freq = ['Energy_Freq: Mean Frequency','Energy_Freq: f2','Energy_Freq: f3','Energy_Freq: f4','Energy_Freq: f5','Energy_Freq: f6','Energy_Freq: f7','Energy_Freq: f8','Energy_Freq: f9','Energy_Freq: f10','Energy_Freq: f11','Energy_Freq: f12','Energy_Freq: f13','Energy_Freq: f14','Energy_Physics: Cumulative energy']
    feature_level_data_base_path = r"VAE_AE_DATA"
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_filepaths = len(all_paths)

    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    num_features = len(expected_cols)
    # Leave-one-out split
    test_path = all_paths[2]
    #val_path_idx = (i+5)%(int(n_filepaths))
    val_path = all_paths[6]
    #val_id = all_ids[val_path_idx]
    train_paths = [p for j, p in enumerate(all_paths) if j != 2 and j!=6]

    # Load and flatten (merge) training data csv files, resampling to 300 rows
    vae_train_data, vae_scaler = VAE_merge_data_per_timestep(train_paths, expected_cols, target_rows)

    # Load expected colums of test data excluding time
    df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
    df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
    df_test = df_test[expected_cols]
    df_val = df_val[expected_cols]

    df_test_resampled = resample_dataframe(df_test, target_rows)
    df_val_resampled = resample_dataframe(df_val, target_rows)

    vae_test_data = df_test_resampled.values
    vae_val_data = df_val_resampled.values

    # Standardize val and test data
    vae_test_data = vae_scaler.transform(vae_test_data)
    vae_val_data = vae_scaler.transform(vae_val_data)

    # Train model
    hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2=10, target_rows=target_rows)
    
    hi_train = hi_train.reshape(-1, target_rows)

    # # Print 
    # print(f'Epoch losses: {epoch_losses}')
    # print(f'\n HI_train shape: {hi_train.shape}, \n HI_train: {hi_train}')
    # print(f'\n HI_test shape: {hi_test.shape}, \n HI_test: {hi_test}')
    # print(f'\n HI_val shape: {hi_val.shape}, \n HI_val: {hi_val}')

    # Plot HI graph
    filepath = r"Test_HI_graph.png"
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    x = np.linspace(0,100,target_rows)
    plt.plot(x, hi_train[0].reshape(-1,1), color='r', label='Sample 1')
    plt.plot(x, hi_train[1].reshape(-1,1), color='lime', label='Sample 2')
    plt.plot(x, hi_test.reshape((-1,1)), color='b', label='Sample 3 ')
    plt.plot(x, hi_val.reshape((-1,1)), color='violet', label='Sample 6 ')
    for i in range(2,5):
        plt.plot(x, hi_train[i].reshape(-1,1), color=colors[i], label=f'Sample {i+2}') 
    for i in range(5,10):
        plt.plot(x, hi_train[i].reshape(-1,1), color=colors[i], label=f'Sample {i+3}')
    plt.title('Test panel = 2')
    plt.xlabel('Lifetime (%)')
    plt.ylabel('HI')
    plt.legend()
    plt.savefig(filepath)
    plt.show()

'''ATTEMPTING TO IMPLEMENT OPTIMIZATION'''
optimizing = True
if __name__ == "__main__" and optimizing:
    folder_store_hyperparameters = r"VAE_AE_DATA"

    target_rows = 201
    batch_size = 300
    n_calls_per_sample = 12
    feature_level_data_base_path = r"VAE_AE_DATA"
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_filepaths = len(all_paths)

    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    num_features = len(expected_cols)
    # Define space over which hyperparameter optimization will be performed (hyperparameter range and name set)
    # space = [
    #     Integer(40, 60, name='hidden_1'),
    #     Real(0.001, 0.01, name='learning_rate'),
    #     Integer(500, 600, name='epochs'),
    #     Real(0.05, 0.1, name='reloss_coeff'),
    #     Real(1.4, 1.8, name='klloss_coeff'),
    #     Real(2.6, 3, name='moloss_coeff')
    # ]

    # Use the decorator to automatically convert parameters proposed by optimizer to keyword arguments for objective funtion
    # [50, 8, 0.003, 550, 0.06, 1.6, 2.8] → hidden_1=50, batch_size=8, ...
    #@use_named_args(space)
    while True:
        var = int(input("Select 1 to run Bayesian or 2 to run optuna optimization"))
        if var == 1:
            VAE_optimize_hyperparameters(folder_store_hyperparameters, expected_cols, all_paths, n_calls_per_sample, target_rows, space, batch_size)
            break
        if var == 2:
            optimize_hyperparameters_optuna(folder_store_hyperparameters, expected_cols, all_paths, n_calls_per_sample, target_rows, num_features)
            break



''' CHECK THIS OUT LATER'''
code_og = False
if __name__ == "__main__" and code_og:
    base_path = "VAE_AE_DATA"
    sample_ids = [f"Sample{i}Interp.csv" for i in range(1, 13)]
    sample_paths = [os.path.join(base_path, sid) for sid in sample_ids]

    all_data = VAE_merge_data_per_timestep(sample_paths)
    scaler = StandardScaler().fit(all_data)
    all_data_scaled = scaler.transform(all_data).astype(np.float32)

    hidden_1 = 50
    batch_size = 300
    learning_rate = 0.0055
    epochs = 550
    reloss_coeff = 0.075
    klloss_coeff = 1.6
    moloss_coeff = 2.8

    hi_full = np.zeros((12, 1, 12000))
    fitness_scores = []
    csv_dir = base_path

    os.makedirs(csv_dir, exist_ok=True)

    for idx, sid in enumerate(tqdm(sample_ids, desc="Running VAE folds", unit="fold")):
        test_path = sample_paths[idx]
        train_paths = [p for j, p in enumerate(sample_paths) if j != idx]
        train_data = VAE_merge_data_per_timestep(train_paths)
        train_data_scaled = scaler.transform(train_data).astype(np.float32)

        df_test = pd.read_csv(test_path)
        if 'Time' in df_test.columns:
            df_test = df_test.drop(columns=['Time'])
        if 'Unnamed: 0' in df_test.columns:
            df_test = df_test.drop(columns=['Unnamed: 0'])
        expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_resampled = pd.DataFrame()
        for col in df_test.columns:
            original = df_test[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, 12000)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated
        test_data = df_resampled.values.flatten(order='C').reshape(1, -1)
        test_data_scaled = scaler.transform(test_data).astype(np.float32)

        hi_train, hi_test = VAE_train(train_data_scaled, val_data_scaled, test_data_scaled, hidden_1, batch_size, learning_rate, 
                                      epochs, reloss_coeff, klloss_coeff, moloss_coeff, hidden_2=10, target_rows=12000, num_features=5)

        train_hi_min = np.mean(hi_train, axis=1, keepdims=True)
        train_hi_max = np.max(hi_train, axis=1, keepdims=True)
        hi_test_normalized = (hi_test - train_hi_min[0:1]) / (train_hi_max[0:1] - train_hi_min[0:1] + 1e-8)
        hi_full[idx, 0, :] = hi_test_normalized[0]

        x = np.linspace(0, 100, 12000)
        fig = plt.figure()
        for i in range(hi_train.shape[0]):
            train_hi_normalized = (hi_train[i] - train_hi_min[i]) / (train_hi_max[i] - train_hi_min[i] + 1e-8)
            plt.plot(x, train_hi_normalized, color="gray", alpha=0.4, label="Train" if i == 0 else "")
        plt.plot(x, hi_test_normalized[0], color="tab:blue", linewidth=2, label=f"{sid[:-4]} (Test)")
        plt.xlabel("Lifetime (%)")
        plt.ylabel("Health Indicator")
        plt.title(f"Health Indicator - {sid[:-4]}")
        plt.legend()
        plt.savefig(os.path.join(csv_dir, f"HI_graph_{sid[:-4]}.png"))
        plt.close(fig)

        hi_train_compressed = np.array([scale_exact(hi, minimum=30) for hi in hi_train])
        ftn, mo, tr, pr, err = fitness(hi_train_compressed)
        sample_num = int(sid.replace("Sample", "").replace("Interp.csv", ""))
        fitness_scores.append([sample_num, ftn, mo, tr, pr, err])

    try:
        np.save(os.path.join(csv_dir, f"VAE_AE_seed_{VAE_Seed.vae_seed}.npy"), hi_full)
        df_fitness = pd.DataFrame(fitness_scores, columns=['Sample_ID', 'Fitness_Score', 'Monotonicity', 'Trendability', 'Prognosability', 'Error'])
        df_fitness.to_csv(os.path.join(csv_dir, 'fitness_scores.csv'), index=False)
        print(f"\n✅ All folds completed. Saved HI array to {os.path.join(csv_dir, f'VAE_AE_seed_{VAE_Seed.vae_seed}.npy')}")
        print(f"✅ Saved fitness scores to {os.path.join(csv_dir, 'fitness_scores.csv')}")
    except Exception as e:
        print(f"Error saving files: {e}")