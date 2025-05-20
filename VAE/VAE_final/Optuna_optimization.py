import optuna
from Model_architecture import VAE_Seed
from Train import VAE_train
import numpy as np
from Prog_crit import fitness
import random
from File_handling import VAE_merge_data_per_timestep,resample_dataframe
import pandas as pd
import tensorflow as tf
import inspect
import os

def optimize_hyperparameters_optuna(
    num_features, target_rows, vae_train_data, vae_val_data, vae_test_data,
    n_trials=40,
    direction='minimize' , 
):
    """
    Optimize VAE hyperparameters using Optuna's TPE sampler and pruning.
    Returns best_params dict and best_value.
    """
    def objective(trial):
        # Suggest hyperparameters
        hidden_1     = trial.suggest_int('hidden_1',     40, 120)
        learning_rate= trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
        epochs       = trial.suggest_int('epochs',       500, 1000)
        reloss_coeff = trial.suggest_uniform('reloss_coeff', 0.05, 0.6)
        klloss_coeff = trial.suggest_uniform('klloss_coeff', 1.4, 1.8)
        moloss_coeff = trial.suggest_uniform('moloss_coeff', 2.6, 4.0)
        
        #these are added hyperparameters, these can be removed if necessary
        hidden_2 = trial.suggest_int('hidden_2',8,32)
        batch_size = trial.suggest_int('batch_size',100,1000)

        # Train VAE with these params
        hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(target_rows,
            vae_train_data, vae_val_data, vae_test_data,
            hidden_1, batch_size, learning_rate, epochs,
            reloss_coeff, klloss_coeff, moloss_coeff,
            num_features, hidden_2=hidden_2,
            )

        # Compute fitness error on stacked health indicators
        hi_all = np.vstack((hi_train, hi_test, hi_val))
        _, _, _, _, error = fitness(hi_all)
        trial.report(error, step=0)
        return error

    # Create study with TPE sampler and median pruner
    sampler = optuna.samplers.TPESampler(seed=VAE_Seed.vae_seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=None)

    print("Best trial:", study.best_trial.params)
    return study.best_trial.params, study.best_value

def VAE_optimize_hyperparameters_optuna(folder_save_opt_param_csv, expected_cols, filepaths, n_calls_per_sample, target_rows, num_features):
    """
    Run leave-one-out cross-validation on 12 samples to optimize VAE hyperparameters.
    Saves the best set of hyperparameters per test sample in a CSV.
    """
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)
    
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

        print("VAE_hyperparameter_optimisation signature:", inspect.signature(optimize_hyperparameters_optuna))

        # Optimize - Runs optimization funtion to tune hyperparameters over 'n_calls_per_sample' trials
        best_params, best_error = optimize_hyperparameters_optuna(num_features, target_rows, vae_train_data, vae_val_data, vae_test_data)
        # best_params = opt_hyperparameters[0]
        # best_error = opt_hyperparameters[1]

        # Stores tuple of: test_id, hyperparametes, and error in results list
        results.append((test_id, best_params, best_error)) 

    # Save results in df (save list of tuples in df with 3 cols) -> save df to csv file
    df_out = pd.DataFrame(results, columns=["test_panel_id", "params", "error"])
    df_out.to_csv(os.path.join(folder_save_opt_param_csv, "hyperparameters-opt-samples.csv"))
    print(f"\n✅ Saved best parameters to {os.path.join(folder_save_opt_param_csv, 'hyperparameters-opt-samples.csv')}")