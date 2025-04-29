import pandas as pd
import numpy as np
import math
''' 
The SPWVD code requires some additional processing and windowing which is done here, this code has the following structure:

    1. Input and output file names defined as well as which data columns to use
    2. Downsampling parameters defined
    3. Functions for smoothing, downsampling and windowing, and only downsampling defined
    4. Functions applied to all input files and downsampled files saved'''

# 1. Input and output file names defined as well as which data columns to use
csv_files = ["SP\\Sample1Interp.csv", "SP\\Sample1Interp.csv"]
smoothing_output_filenames = ["Sample1_cycle_level_features_smoothed.csv", "Sample1_cycle_level_features_smoothed.csv"]
windowed_output_filenames = ["Sample1_window_level_features_smoothed.csv", "Sample1_window_level_features_smoothed.csv"]
relevant_col_names = ['Time','Amplitude_mean','Amplitude_std','Amplitude_max','Amplitude_min','Energy_sum','Energy_mean','Counts_sum','Duration_mean','RMS_mean','Rise-Time_mean','Amplitude_mean_smoothed','Amplitude_std_smoothed','Amplitude_max_smoothed','Amplitude_min_smoothed','Energy_sum_smoothed','Energy_mean_smoothed','Counts_sum_smoothed','Duration_mean_smoothed','RMS_mean_smoothed','Rise-Time_mean_smoothed']

# 2. Downsampling parameters defined
downsample_factor=10 
truncation_loc=40000 
overlap_window=200

# 3. Functions for smoothing, downsampling and windowing, and only downsampling defined
def generate_smoothed_df(filename, output_filename, smoothing_window_size=5):
    # Load dataframe 
    df = pd.read_csv(filename)

    # Step 1: Group by cycle and compute features per cycle
    grouped = df.groupby('Time')

    # Define aggregations for potentially usefull features
    agg_df = grouped.agg({
        'Amplitude': ['mean', 'std', 'max', 'min'],
        'Energy': ['sum', 'mean'],
        'Counts': ['sum'],
        'Duration': ['mean'],
        'RMS': ['mean'],
        'Rise-Time': ['mean'],
    })

    # Step 2: Flatten the MultiIndex column names
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df = agg_df.reset_index()

    # Step 3: Add event count per cycle
    agg_df['Num_Events'] = grouped.size().values

    # Step 4: Handle NaNs (for std when only one AE event per cycle)
    agg_df.fillna(0, inplace=True)  # Or use a different strategy like interpolation

    # Step 5: Apply rolling smoothing (default is 5 cycle window)
    window_size = smoothing_window_size
    smoothed_df = agg_df.copy()
    rolling_cols = [col for col in agg_df.columns if col not in ['Time']]  # exclude 'Time'

    for col in rolling_cols:
        smoothed_df[f'{col}_smoothed'] = agg_df[col].rolling(window=window_size, min_periods=1).mean()

    # Step 6: Preview first few rows
    print(smoothed_df.head())

    # Step 7: Save to CSV
    smoothed_df.to_csv(output_filename, index=False)

def get_downsampled_and_windowed_smoothe_data(csv_filename, csv_output_filename, relevant_col_names, downsample_factor, truncation_loc, overlap_window):
    # Load data
    smoothed_df = pd.read_csv(csv_filename)
    time_cycles = smoothed_df['Time'].values

    # Define number of datapoints and number of windows data is split into
    N_cycles_total = time_cycles.shape[0]
    print('total n of cycles:', N_cycles_total)
    N_cycle_windows = math.ceil(N_cycles_total/(truncation_loc-overlap_window))
    print('N_cycle_windows:', N_cycle_windows)

    # Check downsampling factor matches with number of data points per window
    if truncation_loc%downsample_factor!=0:
        print('Error: number of samples per window is not an integer -> (change downsampling factor to factor of number of samples per window)')

    # Creates array of start and stop indices of each window of data
    Cycle_windows = np.zeros((N_cycle_windows, 2))
    Cycle_windows[0, 1] = truncation_loc
    for i in range(1,N_cycle_windows):
        print('i = ', i)
        if i!=(N_cycle_windows-1):
            j = i-1
            start = int(Cycle_windows[j, 1] - overlap_window)
            stop = int(start + truncation_loc)
        elif i==(N_cycle_windows-1):
            stop = N_cycles_total
            start = int(N_cycles_total - truncation_loc)
        Cycle_windows[i,0] = start
        Cycle_windows[i,1] = stop

    # Create arrays of data divided in windows using start and stop indices
    for col_name in relevant_col_names:
        signal = smoothed_df[col_name].values
        signal = np.nan_to_num(signal)  # Replace NaNs or infs

        # Creates array of segments of data and time
        downsampled_signals = np.zeros((N_cycle_windows, int(truncation_loc/downsample_factor))) 
        downsampled_signals[0,:] = signal[:truncation_loc:downsample_factor]
        # print('Downsampled signal:',downsampled_signals)
        for i in range(1,N_cycle_windows):
            #print('i = ', i)
            start, stop = Cycle_windows[i,:]
            start = int(start)
            stop = int(stop)
            downsampled_signals[i,:] = signal[start:stop:downsample_factor]
        
        # Create df with first col indicating the data window to which the data point belongs and add flattened version of the signal arrays next to this
        if col_name=='Time':
            n_rows, n_cols = downsampled_signals.shape

            # Create first signal column
            signal_values = downsampled_signals.flatten()

            # Create the 'Data_Window_idx' column by repeating the row indices for each value in that row
            window_indices = np.repeat(np.arange(n_rows), n_cols)

            # Create the DataFrame
            windowed_df = pd.DataFrame({
                'Data_Window_idx': window_indices,
                'Time_cycle': signal_values
            })
            print(windowed_df)

        else:
            windowed_df[col_name] = downsampled_signals.flatten()

    windowed_df.to_csv(csv_output_filename, index=False) # Doesn't add the df row index to csv file
    print(f"Downsampled and windowed data saved as {csv_output_filename}")

def get_simple_downsampled_smoothe_data(csv_filename, csv_output_filename, relevant_col_names, downsample_factor):
    # Load data
    smoothed_df = pd.read_csv(csv_filename)
    df_selected = smoothed_df[relevant_col_names]

    # Downsample and reset indices
    df_downsampled = df_selected.iloc[::downsample_factor]
    df_downsampled = df_downsampled.reset_index(drop=True)

    df_downsampled.to_csv(csv_output_filename, index=False) # Doesn't add the df row index to csv file
    print(f"Downsampled data saved as {csv_output_filename}")


# 4. Functions applied to all input files and downsampled files saved
for i, csv_file in enumerate(csv_files):
    smoothing_output_filename = smoothing_output_filenames[i]
    windowed_output_filename = windowed_output_filenames[i]
    generate_smoothed_df(csv_file, smoothing_output_filename)
    get_downsampled_and_windowed_smoothe_data(smoothing_output_filename, windowed_output_filename, relevant_col_names, downsample_factor, truncation_loc, overlap_window)