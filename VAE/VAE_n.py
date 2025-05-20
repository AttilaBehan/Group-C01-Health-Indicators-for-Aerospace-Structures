import tensorflow as tf
import pandas as pd
import ast
from time import time
import numpy as np
import scipy.interpolate as interp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from Prog_crit import fitness, test_fitness, scale_exact
from File_handling import VAE_merge_data_per_timestep, resample_dataframe
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tqdm import tqdm
from functools import partial
import glob
import inspect
import sys
# from pathlib import Path

# # Get the parent directory of the current file's directory
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

#vae_seed = 42
random.seed(VAE_Seed.vae_seed)
tf.random.set_seed(VAE_Seed.vae_seed)
np.random.seed(VAE_Seed.vae_seed)

# Training_data_folder
train_paths_folder = r"C:\Users\job\OneDrive - Delft University of Technology\Documents\GitHub\Group-C01-Health-Indicators-for-Aerospace-Structures\VAE_AE_DATA"
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

''' HI calculator based on reconstruction errors, 

    per timestep health scores: detect degradation at specific times, allows to check for monotonicity (penalize health decreases over time in VAE_loss)'''
def compute_health_indicator(x, x_recon, k=1.0, target_rows=300, num_features=201):
    ''' x, x_recon should have same shape and be 2D tensors
        k = sensitivity parameter (larger values penalize errors more)'''
    #print(f'x shape: {x.shape}')
    if x.shape[0]==target_rows:
        x_reshaped = tf.convert_to_tensor(x, dtype=tf.float64)
        x_recon_reshaped = tf.convert_to_tensor(x_recon, dtype=tf.float32)
        # Make sure two x tensors have same float type:
        x_reshaped = tf.cast(x_reshaped, tf.float32)
        errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=1) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
        health = tf.exp(-k * errors)  # Shape (1, target_rows)
    else:
        x_reshaped = tf.reshape(x, (-1, target_rows, num_features))  # Reshape to 3D tensor and separate features again
        x_recon_reshaped = tf.reshape(x_recon, (-1, target_rows, num_features))
        # Make sure two x tensors have same float type:
        x_reshaped = tf.cast(x_reshaped, tf.float32)
        errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=2) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
        health = tf.exp(-k * errors)  # Shape (n_samples, target_rows)
    return health




''' Extra callback function for hyperparameter optimization progress report'''
def enhanced_callback(res):
    print(f"Iteration {len(res.x_iters)} - Best error: {res.fun:.4f}")

# Define space for Bayesian hyperparameter optimiation 
space = [
        Integer(70, 140, name='hidden_1'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 1000, name='epochs'),
        Real(0.05, 0.6, name='reloss_coeff'),
        Real(0.1, 1.0, name='klloss_coeff'),
        Real(2.6, 4, name='moloss_coeff')
    ]

# Use the decorator to automatically convert parameters to keyword arguments
@use_named_args(space) # converts positional arguments to keyword arguments


def VAE_objective(params, batch_size):
    """
    Objective function for optimizing VAE hyperparameters.

    Parameters:
        - params (list): List of hyperparameter values in the order:
            [hidden_1, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff]

    Returns:
        - error (float): Error from fitness function (3 / fitness)
    """

    # Unpack parameters
    hidden_1, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff = params

    # Reproducibility
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Print parameters being tested
    print(
        f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

    # Train VAE and obtain HIs for train and test data
    hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(
        vae_train_data, vae_val_data, vae_test_data, 
        hidden_1, batch_size, learning_rate, epochs, 
        reloss_coeff, klloss_coeff, moloss_coeff, 
        num_features, hidden_2, target_rows)

    # Stack train and test HIs
    hi_all = np.vstack((hi_train, hi_test, hi_val))

    # Handle single-dimension case if necessary
    if hi_test.shape[1] == 1:
        hi_test = np.tile(hi_test, (1, -1))  

    # Compute fitness
    ftn, monotonicity, trendability, prognosability, error = fitness(hi_all)

    # Output error value (3 / fitness)
    print("Error: ", error)

    return error

def VAE_objective_with_data(params, vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq):
    return VAE_objective(params, batch_size)

# # Define space over which hyperparameter optimization will be performed (hyperparameter range and name set)
# space = [
#     Integer(40, 60, name='hidden_1'),
#     Real(0.001, 0.01, name='learning_rate'),
#     Integer(500, 600, name='epochs'),
#     Real(0.05, 0.1, name='reloss_coeff'),
#     Real(1.4, 1.8, name='klloss_coeff'),
#     Real(2.6, 3, name='moloss_coeff')
# ]

# # Use the decorator to automatically convert parameters proposed by optimizer to keyword arguments for objective funtion
# # [50, 8, 0.003, 550, 0.06, 1.6, 2.8] → hidden_1=50, batch_size=8, ...
# @use_named_args(space)


    

''' LOOK AT THIS LATER'''
def plot_images(SP_Method_file_type, dir, panels, freqs, seed=VAE_Seed.vae_seed):
    """
    Plot 5x6 figure by merging existing graphs for all folds

    Parameters:
        - seed (int): Seed for reproducibility and filename
        - SP_Method_file_type (str): Indicates whether FFT or HLB data is being processed
        - dir (str): CSV root folder directory
        - panels, freqs (lists of str): List of panels/samples
    Returns: None
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Creating the 5x6 figure Output directory
    filedir = os.path.join(dir, f"big_VAE_graph_{SP_Method_file_type}_seed_{seed}")

    # Initializing the figure
    nrows = len(freqs)
    ncols = len(panels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))

    # Iterate over all folds of panel and frequency
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):

            # Create the filename for each individual graph
            filename = f"HI_graph_{freq}_{panel}_{SP_Method_file_type}_seed_{vae_seed}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):

                # Load the individual graph
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot and hide the axis
                axs[i, j].imshow(img)
                axs[i, j].axis('off')

            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    # Change freqs for labelling
    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold',
                    fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center',
                    fontweight='bold', fontsize=40)

    # Adjust spacing between subplots and save figure
    plt.tight_layout()
    plt.savefig(filedir)

def VAE_save_results(fitness_all, fitness_test, panel, freq, SP_Method_file_type, dir, freqs, seed=VAE_Seed.vae_seed):
    """
    Save VAE results to a CSV file

    Parameters:
        - fitness_all (float): Evaluation of fitness of all 5 HIs
        - fitness_test (float): Evaluation of fitness only for test HI
        - panel (str): Indicates test panel being processed
        - freq (str): Indicates frequency being processed
        - SP_Method_file_type (str): Indicates whether FFT or HLB data is being processed
        - seed (int): Seed for reproducibility and filename
        - dir (str): CSV root folder directory
    Returns: None
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Create filenames for the fitness-test and fitness-all CSV files
    filename_test = os.path.join(dir, f"fitness-test-{SP_Method_file_type}-seed-{seed}.csv")
    filename_all = os.path.join(dir, f"fitness-all-{SP_Method_file_type}-seed-{seed}.csv")

    # Create the fitness-test file if it does not exist or load the existing fitness-test file
    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs) # Creates rows of df labeled with freqs
    else:
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists in fitness-test df (creates new col if not)
    if panel not in df.columns:
        df[panel] = None

    df.loc[freq, panel] = str(fitness_test)

    df.to_csv(filename_test)

    # Create the fitness-all file if it does not exist or load the existing fitness-all file
    if not os.path.exists(filename_all):
        df = pd.DataFrame(index=freqs)
    else:
        df = pd.read_csv(filename_all, index_col=0)

    # Ensure that the panel column exists in fitness-all
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new results
    df.loc[freq, panel] = str(fitness_all)

    # Save the dataframe to a fitness-all CSV
    df.to_csv(filename_all)



''' PLOTS GRAPH OF HI's vs % OF LIFETIME FOR DIFFERENT TEST SAMPLES'''
def plot_HI_graph(HI_all, dataset_name, sp_method_name, folder_output, show_plot=True, n_plot_rows=4, n_plot_col=3):
    ''' Takes data from array HI_all, which contains stacked arrays of HI vs lifetime for samples
        Plots graph for each value of test panel
        
        Inputs:
            - HI_all (arr): stacked arrays of HI vs lifetime for samples
            - dataset_name (str): name to identify data for graph title
            - sp_method_name (str): which sp method was used on data
            - n_plot_rows, n_plot_cols (int): dimensions of subplots (product=n_samples)
            
        Outputs:
            - Prints graphs and saves figure to folder_output'''
    n_samples = n_plot_rows*n_plot_col
    x = np.linspace(0,100, HI_all.shape[1])
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    # Create grid of subplots
    fig, axes = plt.subplots(n_plot_rows, n_plot_col, figsize=(15, 12))

    # Flatten the axes array for simple iteration
    axes = axes.flatten()
    # Plot each row of y against x
    for i, ax in enumerate(axes):
        y = HI_all[i*n_samples:i*n_samples+n_samples,:]
        for j in range(y.shape[0]):  # Plot all rows/samples
            ax.plot(x, y[j,:], color=colors[j], label=f'Sample{j+1}')
            ax.set_title(f"Test sample {j + 1}")
            ax.grid(True)

    # Add a single legend and big title for all subplots
    fig.legend(loc='center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.suptitle(f'HIs constructed by VAE using {dataset_name} and {sp_method_name} features')
    plt.tight_layout()

    # Save the figure to file
    filename = f'HI_graphs_VAE_{dataset_name}_{sp_method_name}.png'
    file_path = os.path.join(folder_output, filename)
    plt.savefig(file_path)
    if show_plot:
        plt.show()

''' Trying out full training and optimization run'''

train_once = True
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
    feature_level_data_base_path = r"C:\Users\job\OneDrive - Delft University of Technology\Documents\GitHub\Group-C01-Health-Indicators-for-Aerospace-Structures\VAE_AE_DATA"
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
    vae_train_data, vae_scaler = VAE_merge_data_per_timestep_new(train_paths, expected_cols, target_rows)

    # Load expected colums of test data excluding time
    df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
    df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
    df_test = df_test[expected_cols]
    df_val = df_val[expected_cols]

    # df_test_resampled = pd.DataFrame()
    # df_val_resampled = pd.DataFrame()
    # for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
    #     original = df_test[col].values
    #     og = df_val[col].values
    #     x_original = np.linspace(0, 1, len(original))
    #     x_val_original = np.linspace(0,1,len(og))
    #     x_target = np.linspace(0, 1, target_rows)
    #     interpolated = np.interp(x_target, x_original, original)
    #     interp = np.interp(x_target, x_val_original, og)
    #     df_test_resampled[col] = interpolated
    #     df_val_resampled[col] = interp

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
    filepath = r"C:\Users\job\Downloads\Test_HI_graph.png"
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
optimizing = False
if __name__ == "__main__" and optimizing:
    folder_store_hyperparameters = r"C:\Users\job\Downloads\Low_Features"

    target_rows = 300
    batch_size = 300
    n_calls_per_sample = 12
    feature_level_data_base_path = r"C:\Users\job\OneDrive - Delft University of Technology\Documents\GitHub\Group-C01-Health-Indicators-for-Aerospace-Structures\VAE_AE_DATA"
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

    VAE_optimize_hyperparameters(folder_store_hyperparameters, expected_cols, all_paths, n_calls_per_sample=n_calls_per_sample, target_rows=target_rows)



''' CHECK THIS OUT LATER'''
code_og = False
if __name__ == "__main__" and code_og:
    base_path = "C:/Users/AJEBr/OneDrive/Documents/Aerospace/BsC year 2/VAE_Project/VAE_AE_DATA"
    sample_ids = [f"Sample{i}Interp.csv" for i in range(1, 13)]
    sample_paths = [os.path.join(base_path, sid) for sid in sample_ids]

    all_data = VAE_merge_data_per_timestep(sample_paths)
    scaler = StandardScaler().fit(all_data)
    all_data_scaled = scaler.transform(all_data).astype(np.float32)

    hidden_1 = 50
    batch_size = 5
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