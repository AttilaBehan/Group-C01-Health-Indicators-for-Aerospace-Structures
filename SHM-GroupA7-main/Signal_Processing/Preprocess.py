import scipy.io
import pandas as pd
import numpy as np
import os
import csv
import pyexcel as p

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


def matToCsv(dir):
    """
        Converts the directory of raw data in .mat format to a new directory of .csv files
        with -CSV appended before the extension.

        Parameters:
            dir (str): Directory of the input data, with files of .mat format
        Returns:
            none
        """
    # name the new directory with 'PZT-CSV' instead of 'PZT'
    csv_dir = dir.replace('PZT', 'PZT-CSV')

    # Create the output directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Iterates over all the frequency files in the directory
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            # Process only directories that end with "cycles"
            if name.endswith("cycles"):
                # Update the root path for the new CSV directory
                root_new = root.replace('PZT', 'PZT-CSV')
                for i in range(10):
                    root_new = root_new.replace(f'State_{i}_', f'State_0{i}_')

                # Create new output subdirectory if it doesn't exist
                if not os.path.exists(root_new):
                    os.makedirs(root_new)

                # Combine data from all .mat files in the current directory
                arrayfile = combine_act(os.path.join(root, name))
                # Drop the 'Sensor_0' column
                arrayfile = arrayfile.drop(columns='Sensor_0', axis=1)
                # Create the path for the new CSV file
                csv_file_path = os.path.join(root_new, f"{name.replace('_5cycles', '')}.csv")
                # Replace '\50kHz' with '\050kHz' in the file path
                csv_file_path = csv_file_path.replace(r'\50kHz', r'\050kHz')

                # Save the DataFrame to a CSV file
                arrayfile.to_csv(csv_file_path, index=False)


def combine_act(freq):
    """
        Combines data from .mat files in the given directory into a single DataFrame.

        Parameters:
            freq (str): Directory containing the .mat files
        Returns:
            combined_df (pd.DataFrame): Combined data from all .mat files as a DataFrame
    """
    freq_arr = []
    # Iterate over all files in the given directory
    for root, dirs, files in os.walk(freq):
        for name in files:
            # Process only .mat files that end with 'rep_1.mat'
            if name.endswith('rep_1.mat'):
                locat = os.path.join(root, name)
                actuatorNum = int(locat[-25])
                mat_array = scipy.io.loadmat(os.path.join(root, name))

                # Iterating over the different keys(TriggerInfo and Data files)
                for key, value in mat_array.items():
                    if not key.startswith('_') and isinstance(value, (list, tuple, np.ndarray)):

                        # Ensure the array is 1D
                        flattened_data = np.ravel(value)

                        # Check if the length is divisible by 9
                        if len(flattened_data) % 9 == 0:
                            # Reshape the array into 9 columns
                            reshaped_data = np.reshape(flattened_data, (-1, 9))

                            # Create a DataFrame with 9 columns
                            data = pd.DataFrame(reshaped_data, columns=[f'Sensor_{i}' for i in range(9)])
                            # Drop the column corresponding to the current actuator
                            data = data.drop(data.columns[actuatorNum], axis=1)
                            # inserts data in the correct order (in terms of which sensor acts as actuator) into a list
                            freq_arr.insert(actuatorNum - 1, data)

                        elif (key == "Trigger_Info"):
                            # Create a DataFrame for the Trigger_Info key
                            data = pd.DataFrame(flattened_data)
                        else:
                            pass
                            print(f"Data format invalid for key {key}. Skipping.")
                # combines all the entries in the list into a single dataframe
                combined_df = pd.concat(freq_arr, axis=1)

    return combined_df
