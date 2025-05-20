import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler




def VAE_merge_data_per_timestep(sample_filenames, expected_cols, target_rows):
    """
    Load multiple AE data files, resample each to have target_rows rows via interpolation.
    Combine all resampled data into a single dataframe: one row per time step.
    Shape: (n_samples * target_rows, n_features)
    """
    all_data = []

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")

        df = pd.read_csv(path)

        # Column cleanup
        cols_to_drop = ['Time (Cycle)', 'Unnamed: 0', 'Time']  # Combine checks
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # if 'Time' in df.columns:
        #     df = df.drop(columns=['Time (Cycle)'])
        # if 'Unnamed: 0' in df.columns:
        #     df = df.drop(columns=['Unnamed: 0'])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
        df = df[expected_cols]

        # Resample each feature independently
        df_resampled = resample_dataframe(df, target_rows)

        # df_resampled = pd.DataFrame()
        # for col in df.columns:
        #     original = df[col].values
        #     x_original = np.linspace(0, 1, len(original))
        #     x_target = np.linspace(0, 1, target_rows)
        #     interpolated = np.interp(x_target, x_original, original)
        #     df_resampled[col] = interpolated

        all_data.append(df_resampled)

    # Stack time steps from all samples
    data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
    print(f"✅ Merged data shape: {data.shape}")

    # Standardize feature-wise (column-wise)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

    return data_scaled, scaler

''' Resampling test and validation data'''
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