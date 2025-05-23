import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from Model_architecture import VAE_Seed, VAE
from skopt import gp_minimize
from functools import partial
from skopt.space import Real, Integer
import inspect
from File_handling import VAE_merge_data_per_timestep, resample_dataframe
from Train import VAE_train
from Prognostic_criteria import fitness

def VAE_hyperparameter_optimisation(vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, n_calls, space, batch_size, target_rows):
    """
    Optimize VAE hyperparameters using gp_minimize, a Gaussian process-based minimization algorithm

    Parameters:
        - vae_train_data (np.ndarray): Data used for training, with shape (num_samples, num_features)
        - vae_test_data (np.ndarray): Data used for testing, with shape (num_samples, num_features)
        - vae_scaler (sklearn.preprocessing.StandardScaler): Scaler object for standardization
        - vae_pca (sklearn.decomposition.PCA): PCA object to apply PCA
        - vae_seed (int): Seed for reproducibility
        - file_type (str): Identifier for FFT or HLB data
        - panel (str): Identifier for test panel of fold
        - freq (str): Identifier for frequency of fold
        - csv_dir (str): Directory containing data and hyperparameters
        - n_calls (int): Number of optimization calls per fold
    Returns:
        - opt_parameters (list): List containing the best parameters found for that fold, and the error value (3 / fitness)
    """

    # Make results reproducable by setting the same random seed for tensorflow and numpy
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Create a partial function that includes the data, fixed variables
    objective_with_data = partial(
        VAE_objective_with_data,
        vae_train_data=vae_train_data,
        vae_val_data=vae_val_data,
        vae_test_data=vae_test_data,
        file_type=file_type,
        panel=panel,
        freq=freq,
        target_rows=target_rows,
        batch_size=batch_size
    )

    try:
        # Run optimization using gp_minimize
        res_gp = gp_minimize(
            func=objective_with_data,  # Function to minimize (with data injected)
            dimensions=space,           # Hyperparameter space
            n_calls=n_calls,            # Number of optimization calls
            random_state=VAE_Seed.vae_seed,
            n_jobs=-1,                  # Use all cores for parallel processing
            verbose=True
        )
        # Extract results
        opt_parameters = {
            'best_params': res_gp.x,
            'best_error': res_gp.fun,
            'all_params': res_gp.x_iters,
            'all_errors': res_gp.func_vals
        }

        print(f"\nBest parameters: {res_gp.x}")
        print(f"Best error: {res_gp.fun:.4f}")

        # Print original opt_parameters output
        opt_parameters = [res_gp.x, res_gp.fun]
        print(f'Object type opt_parametes: {type(opt_parameters)}')
        
        return opt_parameters

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None
    
    ''' Uses VAE_hyperparameter_optimization() in loop using LOOCV'''

def VAE_optimize_hyperparameters(folder_save_opt_param_csv, expected_cols, filepaths, n_calls_per_sample, target_rows, space, batch_size):
    """
    Run leave-one-out cross-validation on 12 samples to optimize VAE hyperparameters.
    Saves the best set of hyperparameters per test sample in a CSV.
    """
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)
    batch_size = batch_size
    # Declares these variables as global, a.k.a. the function will use and modify these global variables (Defined elsewhere)
    global vae_train_data, vae_val_data, vae_test_data, vae_scaler, file_type, panel, freq

    # Converts csv_dir into an absolute path — ensures reliable file handling regardless of how the path was passed in.
    #csv_dir = os.path.abspath(csv_dir)

    n_filepaths = len(filepaths)

    # Creates list of sample id names and list of filepaths to save all csv files
    all_ids = [f"Sample{i}" for i in range(1, int(n_filepaths+1))]
    all_paths = filepaths #  [os.path.join(csv_dir, f"{sid}Interp.csv") for sid in all_ids]

    # initiate list to store hyperparams for each sample
    results = []

    # Start looping through samples
    for i, test_id in enumerate(all_ids):
        print(f"\nOptimizing hyperparams: TEST={test_id}")

        # Sets panel global variable to current sample name and resets freq and file_type (not used)
        panel = test_id  # Legacy naming
        freq = None
        file_type = None

        # Leave-one-out split
        test_path = all_paths[i]
        val_path_idx = (i+5)%(int(n_filepaths))
        val_path = all_paths[val_path_idx]
        val_id = all_ids[val_path_idx]
        train_paths = [p for j, p in enumerate(all_paths) if j != i and j!=val_path_idx]

        # Load and flatten (merge) training data csv files, resampling to 12000 rows
        vae_train_data, vae_scaler = VAE_merge_data_per_timestep(train_paths, expected_cols, target_rows)

        # Load expected colums of test data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
        df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
        #expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_val = df_val[expected_cols]

        # df_test_resampled = pd.DataFrame()
        # df_val_resampled = pd.DataFrame()
        # for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
        #     original = df_test[col].values
        #     og = df_val[col].values
        #     x_original = np.linspace(0, 1, len(original))
        #     x_val_original = np.linspace(0, 1, len(og))
        #     x_target = np.linspace(0, 1, target_rows)
        #     interpolated = np.interp(x_target, x_original, original)
        #     interp = np.interp(x_target, x_val_original, og)
        #     df_test_resampled[col] = interpolated
        #     df_val_resampled[col] = interp

        # Resample both DataFrames using function
        df_test_resampled = resample_dataframe(df_test, target_rows)
        df_val_resampled = resample_dataframe(df_val, target_rows)

        # If using old MERGE_DATA - Row major order flattening into 1D array (Row1, Row2, Row3... successive), then reshapes to go from one row to one column
        vae_test_data = df_test_resampled.values
        vae_val_data = df_val_resampled.values

        # Standardize val and test data
        vae_test_data = vae_scaler.transform(vae_test_data)
        vae_val_data = vae_scaler.transform(vae_val_data)

        print("Space definition right before optimization call:", space)
        print("VAE_hyperparameter_optimisation signature:", inspect.signature(VAE_hyperparameter_optimisation))

        # Optimize - Runs optimization funtion to tune hyperparameters over 'n_calls_per_sample' trials
        best_params, best_errors = VAE_hyperparameter_optimisation(vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, n_calls_per_sample, space, batch_size, target_rows)
        # best_params = opt_hyperparameters[0]
        # best_error = opt_hyperparameters[1]

        # Stores tuple of: test_id, hyperparametes, and error in results list
        results.append((test_id, best_params, best_errors)) 

    # Save results in df (save list of tuples in df with 3 cols) -> save df to csv file
    df_out = pd.DataFrame(results, columns=["test_panel_id", "params", "error"])
    df_out.to_csv(os.path.join(folder_save_opt_param_csv, "hyperparameters-opt-samples.csv"))
    print(f"\n✅ Saved best parameters to {os.path.join(folder_save_opt_param_csv, 'hyperparameters-opt-samples.csv')}")
    return best_params


def VAE_objective(params, batch_size, target_rows):
    """
    Objective function for optimizing VAE hyperparameters.

    Parameters:
        - params (list): List of hyperparameter values in the order:
            [hidden_1, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff]

    Returns:
        - error (float): Error from fitness function (3 / fitness)
    """

    # Unpack parameters
    hidden_1, learning_rate, epochs, hidden_2, reloss_coeff, klloss_coeff, moloss_coeff = params

    # Reproducibility
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Print parameters being tested
    print(
        f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}, hidden_2 = {hidden_2}")

    # Train VAE and obtain HIs for train and test data
    hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(
        vae_train_data, vae_val_data, vae_test_data, 
        hidden_1, batch_size, learning_rate, epochs, 
        reloss_coeff, klloss_coeff, moloss_coeff, 
        hidden_2, target_rows)

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

def VAE_objective_with_data(params, batch_size, vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, target_rows):
    return VAE_objective(params, batch_size, target_rows)