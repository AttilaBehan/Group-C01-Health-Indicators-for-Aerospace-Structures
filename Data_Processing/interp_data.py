import pandas as pd
import numpy as np
import os

# def interpolate_missing_cycles(file_path, save_path, column, cycle_length, max_gap=5000000000000):
#     """
#     Interpolates missing cycles but avoids filling large gaps.
    
#     Parameters:
#         file_path (str): Path to the CSV file.
#         save_path (str): Path to save the interpolated file.
#         column (str): The name of the time column.
#         max_gap (int): Maximum allowed gap size for interpolation.
    
#     Returns:
#         pd.DataFrame: Interpolated DataFrame.
#     """
#     # Load the CSV file
#     df = pd.read_csv(file_path)

#     # Ensure the Time column is numeric (handles cases where it's read as string)
#     df[column] = pd.to_numeric(df[column], errors='coerce')

#     # Drop rows where Time is NaN
#     df = df.dropna(subset=[column])

#     # Convert Time to integer safely (after ensuring no NaNs)
#     df[column] = df[column].astype(float)

#     # Create a full range of time values
#     full_time_range = pd.DataFrame({column: np.arange(df[column].min(), df[column].max() + 1,cycle_length)})
#     # Merge with existing data
#     df_interpolated = pd.merge(full_time_range, df, on=column, how='left')
    
#     # Identify gaps
#     df_interpolated["Gap"] = df_interpolated[column].diff()
    
#     # Only interpolate small gaps
#     df_interpolated.iloc[:, 1:] = df_interpolated.iloc[:, 1:].interpolate(method='linear', limit=max_gap)
    
#     # Fill any remaining NaNs at the edges
#     df_interpolated.iloc[:, 1:] = df_interpolated.iloc[:, 1:].ffill(limit_area='outside').bfill(limit_area='outside')
    
#     # Remove the gap column
#     df_interpolated.drop(columns=["Gap"], inplace=True)
#     # Save the cleaned dataset
#     df_interpolated.to_csv(save_path, index=False)
    
#     return df_interpolated 

def interpolate_missing_cycles(file_path, save_path, column, cycle_length):
    """
    Interpolates all missing cycles by generating a full time range and filling in all gaps.

    Parameters:
        file_path (str): Path to the CSV file.
        save_path (str): Path to save the interpolated file.
        column (str): The name of the time column.
        cycle_length (int or float): Expected interval between time points.
    
    Returns:
        pd.DataFrame: Fully interpolated DataFrame.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure time column is numeric
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(float)
    df=df.dropna()

    # Generate a complete time range
    full_time = pd.DataFrame({column: np.arange(df[column].min(), df[column].max() + cycle_length, cycle_length)})

    # Merge with full time range
    df_full = pd.merge(full_time, df, on=column, how='left')

    # Interpolate all missing values (excluding time column)
    for col in df_full.columns:
        if col == column:
            continue
        df_full[col] = df_full[col].interpolate(method='linear', limit_direction='both') 

    # Save the interpolated data
    df_full.to_csv(save_path, index=False)

    return df_full


def get_missing_cycles(input_dir, output_dir, column, cycle_length): 
    """
    Interpolates missing cycles in all CSV files within a directory.
    
    Parameters:
        input_dir (str): Directory containing the input CSV files.
        output_dir (str): Directory to save the interpolated CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, samples in os.walk(input_dir): 
        for sample in samples:
            print("Processing: ", sample)
            file=os.path.join(root, sample) 
            save_path = os.path.join(output_dir, sample)
            interpolate_missing_cycles(file, save_path, column, cycle_length)#, max_gap=50000000)

