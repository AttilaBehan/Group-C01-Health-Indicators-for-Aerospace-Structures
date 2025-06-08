import numpy as np
import pandas as pd
import os
import glob

compute = True

''' Input features standardized to standard normal distribution - for every feature use all data for that feature'''

def adaptive_standardize(feature_trajectory):
    """
    Standardize a feature trajectory adaptively (online) for each timestep.
    
    Args:
        feature_trajectory: 1D array of raw feature values over time (shape [n_timesteps]).
    
    Returns:
        x_ast: Adaptively standardized trajectory (same shape as input).
        mus: Mean values at each step.
        sigmas: Std dev values at each step.
    """
    x_ast = []
    mus, sigmas = [], []
    
    for i in range(1, len(feature_trajectory) + 1):
        # Slice data up to current timestep (t_1 to t_i)
        x_window = feature_trajectory[:i]
        
        # Compute mean and std
        mu_i = np.mean(x_window)
        sigma_i = np.std(x_window) if i > 1 else 1.0  # Avoid division by zero
        
        # Standardize current value
        x_i_ast = (feature_trajectory[i-1] - mu_i) / sigma_i if sigma_i != 0 else 0.0
        
        x_ast.append(x_i_ast)
        mus.append(mu_i)
        sigmas.append(sigma_i)
    
    return np.array(x_ast), np.array(mus), np.array(sigmas)

def adaptive_standardize_dataset(X):
    """
    Standardize all samples and features adaptively.
    
    Args:
        X: 3D array of shape (n_samples, n_timesteps, n_features).
    
    Returns:
        X_ast: Standardized array (same shape as X).
    """
    n_samples, n_timesteps, n_features = X.shape
    X_ast = np.zeros_like(X)
    
    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            trajectory = X[sample_idx, :, feature_idx]
            x_ast, _, _ = adaptive_standardize(trajectory)
            X_ast[sample_idx, :, feature_idx] = x_ast
    
    return X_ast

def load_samples_data_into_3D_array(csv_files):
    n_samples = len(csv_files)

    sample_data = pd.read_csv(csv_files[0]).values
    sample_data = sample_data[:,1:]
    n_timesteps, n_features = sample_data.shape

    # Initialize 3D array (pre-allocate memory)
    data_3d = np.zeros((n_samples, n_timesteps, n_features))

    # Load each CSV into the 3D array
    for i, csv_file in enumerate(csv_files):
        arr = pd.read_csv(csv_file).values
        arr = arr[:,1:]  # Shape (n_timesteps, n_features)
        #print(arr)
        data_3d[i] = arr # pd.read_csv(csv_file, header=None).values[:,1:]  # Shape (n_timesteps, n_features)
    
    return data_3d

if compute:
    all_sp_features_folder = r"c:\Users\naomi\OneDrive\Documents\Extracted_High_Features_data\Interpolated_to_equal_rows"
    print(f'Select SP method for features: \n 1. FFT \n 2. EMD \n 3. HT \n 4. CWT \n 5. SPWVD \n 6. Time Domain features \n 7. FFT Morteza')

    SP_method_idx = input('Enter SP method number')
    SP_method_idx = int(SP_method_idx)

    SP_folders = ['FFT_', 'EMD_', 'HT_', 'CWT_', 'SPWVD_', 'Time_Domain_', 'FFT_Morteza']

    Data_folder = os.path.join(all_sp_features_folder, SP_folders[SP_method_idx-1])

    all_sample_paths = glob.glob(Data_folder + "/*.csv")
    #print(all_sample_paths)

    df_sample1 = pd.read_csv(all_sample_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]

    # Load data without time column into 3D array shape (samples,timesteps,features)
    data_3d = load_samples_data_into_3D_array(all_sample_paths)
    #print(data_3d)

    data_3d_ast = adaptive_standardize_dataset(data_3d)

    # Create foler for all ast data
    ast_data_folder = os.path.join(all_sp_features_folder, 'ast_data')
    os.makedirs(ast_data_folder, exist_ok=True)

    # Create folder in folder for specific SP method data
    ast_sp_folder = os.path.join(ast_data_folder, SP_folders[SP_method_idx-1])
    os.makedirs(ast_sp_folder, exist_ok=True)

    # Save all AST data for samples to CSV
    for i in range(data_3d_ast.shape[0]):
        df_ast = pd.DataFrame(data_3d_ast[i,:,:], columns=expected_cols)
        ast_csv_output_dir = os.path.join(ast_sp_folder, f"Sample{i+1}AST_Features.csv")
        df_ast.to_csv(ast_csv_output_dir, index=False)
        
        print(f" \n DataFrame containing ast data for sample {i+1} {SP_folders[SP_method_idx-1]} saved to {ast_csv_output_dir}")

    



