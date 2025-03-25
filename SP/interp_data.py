import pandas as pd
import numpy as np

def interpolate_missing_cycles(file_path, save_path, time_column='Time', max_gap=5):
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
    full_time_range = pd.DataFrame({time_column: range(df[time_column].min(), df[time_column].max() + 1)})
    
    # Merge with existing data
    df_interpolated = pd.merge(full_time_range, df, on=time_column, how='left')
    
    # Identify gaps
    df_interpolated["Gap"] = df_interpolated[time_column].diff()
    
    # Only interpolate small gaps
    df_interpolated.iloc[:, 1:] = df_interpolated.iloc[:, 1:].interpolate(method='linear', limit=max_gap)
    
    # Fill any remaining NaNs at the edges
    df_interpolated = df_interpolated.bfill().ffill()
    
    # Remove the gap column
    df_interpolated.drop(columns=["Gap"], inplace=True)
    
    # Save the cleaned dataset
    df_interpolated.to_csv(save_path, index=False)
    
    return df_interpolated


# Example usage
file_path = r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample11.csv"
save_path = r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV_interp\Sample11Interp.csv"

interpolated_df = interpolate_missing_cycles(file_path, save_path)

print("Interpolation complete. Saved to:", save_path)
