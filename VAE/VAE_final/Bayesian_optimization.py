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

def VAE_hyperparameter_optimisation(vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, n_calls, space, batch_size, target_rows, num_features):
    """
    Optimize VAE hyperparameters using gp_minimize, a Gaussian process-based minimization algorithm

    Parameters:
        - vae_train_data (np.ndarray): Data used for training, with shape (num_samples, num_features)
        - vae_val_data (np.ndarray): Data used for validation, with shape (num_samples, num_features)
        - vae_test_data (np.ndarray): Data used for testing, with shape (num_samples, num_features)
        - file_type (str): Identifier for FFT or HLB data
        - panel (str): Identifier for test panel of fold
        - freq (str): Identifier for frequency of fold
        - n_calls (int): Number of optimization calls per fold
        - space (list): Hyperparameter search space
        - batch_size (int): Batch size for training
        - target_rows (int): Number of timesteps per sample
        - num_features (int): Number of features per timestep
    Returns:
        - opt_parameters (list): List containing the best parameters found for that fold, and the error value (3 / fitness)
    """
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Create a partial function that includes the data and fixed variables
    objective_with_data = partial(
        VAE_objective_with_data,
        vae_train_data=vae_train_data,
        vae_val_data=vae_val_data,
        vae_test_data=vae_test_data,
        file_type=file_type,
        panel=panel,
        freq=freq,
        target_rows=target_rows,
        batch_size=batch_size,
        num_features=num_features
    )

    try:
        # Run optimization using gp_minimize
        res_gp = gp_minimize(
            func=objective_with_data,  # Function to minimize (with data injected)
            dimensions=space,          # Hyperparameter space
            n_calls=n_calls,           # Number of optimization calls
            random_state=VAE_Seed.vae_seed,
            n_jobs=-1,                 # Use all cores for parallel processing
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
        print(f'Object type opt_parameters: {type(opt_parameters)}')
        
        return opt_parameters

    except Exception as e:
        print(f"Optimization failed: {e}")
        raise e

def VAE_optimize_hyperparameters(folder_save_opt_param_csv, expected_cols, filepaths, n_calls_per_sample, target_rows, space, batch_size, num_features):
    """
    Run leave-one-out cross-validation on 12 samples to optimize VAE hyperparameters.
    Saves the best set of hyperparameters per test sample in a CSV.
    """
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    global vae_train_data, vae_val_data, vae_test_data, vae_scaler, file_type, panel, freq

    n_filepaths = len(filepaths)
    all_ids = [f"Sample{i}" for i in range(1, int(n_filepaths+1))]
    all_paths = filepaths

    results = []

    for i, test_id in enumerate(all_ids):
        print(f"\nOptimizing hyperparams: TEST={test_id}")
        panel = test_id
        freq = None
        file_type = None

        # Leave-one-out split
        test_path = all_paths[i]
        val_path_idx = (i+5) % (int(n_filepaths))
        val_path = all_paths[val_path_idx]
        val_id = all_ids[val_path_idx]
        train_paths = [p for j, p in enumerate(all_paths) if j != i and j != val_path_idx]

        # Load and flatten (merge) training data csv files, resampling to target_rows
        vae_train_data, vae_scaler = VAE_merge_data_per_timestep(train_paths, expected_cols, target_rows)

        # Load expected columns of test and validation data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Unnamed: 0'])
        df_val = pd.read_csv(val_path).drop(columns=['Unnamed: 0'])
        df_test = df_test[expected_cols]
        df_val = df_val[expected_cols]

        df_test_resampled = resample_dataframe(df_test, target_rows)
        df_val_resampled = resample_dataframe(df_val, target_rows)

        vae_test_data = df_test_resampled.values
        vae_val_data = df_val_resampled.values

        # Standardize val and test data
        vae_test_data = vae_scaler.transform(vae_test_data)
        vae_val_data = vae_scaler.transform(vae_val_data)

        # Reshape data to 3D: (n_samples, target_rows, num_features)
        vae_train_data = vae_train_data.reshape(-1, target_rows, num_features)
        vae_val_data = vae_val_data.reshape(-1, target_rows, num_features)
        vae_test_data = vae_test_data.reshape(-1, target_rows, num_features)

        print("Space definition right before optimization call:", space)
        print("VAE_hyperparameter_optimisation signature:", inspect.signature(VAE_hyperparameter_optimisation))

        # Optimize
        best_params, best_errors = VAE_hyperparameter_optimisation(
            vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, 
            n_calls_per_sample, space, batch_size, target_rows, num_features
        )
        results.append((test_id, best_params, best_errors))

    df_out = pd.DataFrame(results, columns=["test_panel_id", "params", "error"])
    df_out.to_csv(os.path.join(folder_save_opt_param_csv, "hyperparameters-opt-samples.csv"))
    print(f"\n✅ Saved best parameters to {os.path.join(folder_save_opt_param_csv, 'hyperparameters-opt-samples.csv')}")
    return best_params


def VAE_objective(params, batch_size, target_rows, num_features):
    hidden_1, learning_rate, epochs, hidden_2, reloss_coeff, klloss_coeff, moloss_coeff = params
    # Convert integer parameters to Python int
    hidden_1 = int(hidden_1.item()) if isinstance(hidden_1, np.integer) else int(hidden_1)
    epochs = int(epochs.item()) if isinstance(epochs, np.integer) else int(epochs)
    hidden_2 = int(hidden_2.item()) if isinstance(hidden_2, np.integer) else int(hidden_2)
    target_rows = int(target_rows)
    num_features = int(num_features)
    batch_size = int(batch_size)

    #bug checking
    print(f"Params: hidden_1={hidden_1} ({type(hidden_1)}), hidden_2={hidden_2} ({type(hidden_2)}), "
          f"epochs={epochs} ({type(epochs)}), target_rows={target_rows} ({type(target_rows)}), "
          f"num_features={num_features} ({type(num_features)}), batch_size={batch_size} ({type(batch_size)})")
    if not all(isinstance(x, int) for x in [hidden_1, hidden_2, epochs, target_rows, num_features, batch_size]):
        raise ValueError("All integer parameters must be Python int")
    print(f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
          f"epochs={epochs}, hidden_2={hidden_2}, reloss_coeff={reloss_coeff}, "
          f"klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")
    
    
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)
    hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(
        vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, 
        epochs, reloss_coeff, klloss_coeff, moloss_coeff, hidden_2, target_rows, num_features
    )
    hi_all = np.vstack((hi_train, hi_test, hi_val))
    if hi_test.shape[1] == 1:
        hi_test = np.tile(hi_test, (1, -1))  
    ftn, monotonicity, trendability, prognosability, error = fitness(hi_all)
    print(f"Error: {error}")
    return error
# def VAE_objective(params, batch_size, target_rows, num_features):
#     """
#     Objective function for optimizing VAE hyperparameters.

#     Parameters:
#         - params (list): List of hyperparameter values in the order:
#             [hidden_1, learning_rate, epochs, hidden_2, reloss_coeff, klloss_coeff, moloss_coeff]
#         - batch_size (int): Batch size for training
#         - target_rows (int): Number of timesteps per sample
#         - num_features (int): Number of features per timestep

#     Returns:
#         - error (float): Error from fitness function (3 / fitness)
#     """
#     hidden_1, learning_rate, epochs, hidden_2, reloss_coeff, klloss_coeff, moloss_coeff = params
#     random.seed(VAE_Seed.vae_seed)
#     tf.random.set_seed(VAE_Seed.vae_seed)
#     np.random.seed(VAE_Seed.vae_seed)

#     print(f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
#           f"epochs={epochs}, hidden_2={hidden_2}, reloss_coeff={reloss_coeff}, "
#           f"klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

#     hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(
#         vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, 
#         epochs, reloss_coeff, klloss_coeff, moloss_coeff, hidden_2, target_rows, num_features
#     )

#     hi_all = np.vstack((hi_train, hi_test, hi_val))
#     if hi_test.shape[1] == 1:
#         hi_test = np.tile(hi_test, (1, -1))  
#     ftn, monotonicity, trendability, prognosability, error = fitness(hi_all)
#     print("Error: ", error)
#     return error



def VAE_objective_with_data(params, batch_size, vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, target_rows, num_features):
    return VAE_objective(params, batch_size, target_rows, num_features)