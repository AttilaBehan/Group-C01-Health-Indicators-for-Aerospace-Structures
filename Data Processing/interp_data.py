import pandas as pd
import numpy as np
import os

def interpolate_missing_cycles(file_path, save_path, time_column='Time (cycle)', max_gap=5):
    """
    Interpolates missing cycles but avoids filling large gaps.
    
    Parameters:
        file_path (str): Path to the CSV file.
        save_path (str): Path to save the interpolated file.
        time_column (str): The name of the time column.
        max_gap (int): Maximum allowed gap size for interpolation.
    
    Returns:
        pd.DataFrame: Interpolated DataFrame.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure the Time column is numeric (handles cases where it's read as string)
    df[time_column] = pd.to_numeric(df[time_column], errors='coerce')

    # Drop rows where Time is NaN
    df = df.dropna(subset=[time_column])

    # Convert Time to integer safely (after ensuring no NaNs)
    df[time_column] = df[time_column].astype(int)

    # Create a full range of time values
    full_time_range = pd.DataFrame({time_column: range(df[time_column].min(), df[time_column].max() + 1,500)})
    # Merge with existing data
    df_interpolated = pd.merge(full_time_range, df, on=time_column, how='left')
    
    # Identify gaps
    df_interpolated["Gap"] = df_interpolated[time_column].diff()
    
    # Only interpolate small gaps
    df_interpolated.iloc[:, 1:] = df_interpolated.iloc[:, 1:].interpolate(method='linear')#, limit=max_gap)
    
    # Fill any remaining NaNs at the edges
    df_interpolated = df_interpolated.bfill().ffill()
    
    # Remove the gap column
    df_interpolated.drop(columns=["Gap"], inplace=True)
    # Save the cleaned dataset
    df_interpolated.to_csv(save_path, index=False)
    
    return df_interpolated

input_dir = r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Time_Domain_Features_500_500_CSV"
output_dir = r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Time_Domain_Interpolated_Features_500_500_CSV"
os.makedirs(output_dir, exist_ok=True)

#for root, dirs, samples in os.walk(input_dir): 
#    for sample in samples:
#        file=os.path.join(root, sample)
#        save_path = os.path.join(output_dir, sample)
#        interpolate_missing_cycles(file, save_path)

def linear_interpolate_5D(A, B, t):
    """
    Performs linear interpolation between two 5D points.
    
    Args:
        A (np.array): A 5D point (shape: (5,))
        B (np.array): Another 5D point (shape: (5,))
        t (float): Interpolation factor (0 <= t <= 1)
    
    Returns:
        np.array: Interpolated 5D point
    """
    return (1 - t) * A + t * B

# Example usage  
A = np.array([1500,	72.2,	7.212489168,	72.10982505,	72.37990052,	102.360637,	77.3])  # First 5D point
B = np.array([7500,	79.25,	7.424621202,	79.16295645,	79.42370553,	112.3220815,	84.5])  # Second 5D point
t = 0.5  # Interpolation factor (midway)

result = linear_interpolate_5D(A, B, t)
print(result)