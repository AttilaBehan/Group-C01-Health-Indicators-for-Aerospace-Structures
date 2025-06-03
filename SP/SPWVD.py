import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from Feature_Extraction import CSV_to_Array
import numpy as np
import math 
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from tftb.processing import smoothed_pseudo_wigner_ville
from sklearn.preprocessing import StandardScaler
from scipy.signal import get_window 
#from Variables import SharedData 

''' To Do: add downsampling factors to class '''

# Downsampling parameters from class
downsample_factor=20 
truncation_loc=40000 
overlap_window=200

# Make this true if you want to plot one of the color plots to check
plot_visual = False

''' This file contains the main function which computes the SPWVD and one which applies this to the windowed 
                                        csv files of the various signals which have been fully preprocessed. 

    It currently has the following structure:
     
      1. Imported downsample factor (I am going to change the format of this later to store these values elsewhere in a class)
      2. Array of file names to be processed and one of signal columns to process
      3. Definition of the sampling frequency based on the individual cycle length and downsampling factor 
      4. Main SPWVD funtion, apply function, and fuction which applies main function to windowed data files
                ^^^ as of now the function is not complete (only calculates it for the first window, I will complete this nexrt session)'''

# Windowed data file names
windowing_output_folder = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Low_features_windowed_fully_preprocessed"
# Get a list of CSV file paths in the folder
Windowed_filenames = glob.glob(windowing_output_folder + "/*.csv")
# Folder to store results in:
spwvd_results_folder = r"C:\Users\naomi\OneDrive\Documents\Low_Features\SPWVD_results"
#print(Windowed_filenames)
#Windowed_filenames = Windowed_filenames[:1]
# For testing code:
#Windowed_filenames = ['Sample1_window_level_features_smoothed.csv', 'Sample1_window_level_features_smoothed.csv']

# Signal columns to be used
relevant_col_names = ['Amplitude_mean','Energy_mean','Counts_sum','Duration_mean','RMS_mean','Rise-Time_mean']
#relevant_col_names = ['Amplitude_mean']
smoothed_rel_col_names = ['Data_Window_idx','Time_cycle','Amplitude_mean_smoothed','Energy_mean_smoothed','Counts_sum_smoothed','Duration_mean_smoothed','RMS_mean_smoothed','Rise-Time_mean_smoothed']

# Sampling frequency
fs = 1/(0.5*downsample_factor)

# Main SPWVD function:

def spwvd(x, fs, window_time=None, window_freq=None, Ntime=None, Nfreq=None):
    """
    Smoothed Pseudo Wigner-Ville Distribution
    
    Parameters:
    x - input signal
    fs - sampling frequency
    window_time - time smoothing window (default: 128-point Hamming)
    window_freq - frequency smoothing window (default: 128-point Hamming)
    Ntime - time window length
    Nfreq - frequency window length
    """
    N = len(x)
    
    # Default parameters
    if Ntime is None:
        Ntime = min(129, N//4)
    if Nfreq is None:
        Nfreq = min(129, N//4)
    if window_time is None:
        window_time = get_window('hamming', Ntime)
    if window_freq is None:
        window_freq = get_window('hamming', Nfreq)
    
    # Normalize windows
    window_time = window_time / np.sum(window_time)
    window_freq = window_freq / np.sum(window_freq)
    
    # Initialize tfr output array
    tfr = np.zeros((N, N), dtype=complex)
    
    # Compute analytic signal
    x_analytic = np.fft.fft(x)
    x_analytic[N//2+1:] = 0
    x_analytic = np.fft.ifft(x_analytic)
    
    # Compute Wigner-Ville with smoothing
    ''' tau = time lag, nu = freq lag '''

    # Time smoothing
    for n in range(N):
        tau_max = min(n, N-1-n, Ntime//2) # Maximum possible lag without out-of-bounds
        tau = np.arange(-tau_max, tau_max+1) # Symmetric lags around n
        product = x_analytic[n + tau] * np.conj(x_analytic[n - tau]) # Autocorrelation
        tfr[n, :] = np.fft.fft(product * window_time[tau_max + tau], N) # Windowed & Fourier-transformed
    
    # Frequency smoothing
    for m in range(N):
        nu_max = min(m, N-1-m, Nfreq//2) # Limits to avoid boundary issues
        nu = np.arange(-nu_max, nu_max+1) # Frequency shifts
        tfr[:, m] = np.convolve(tfr[:, m], window_freq[nu_max + nu], mode='same')
    
    # Compile output time-frequency representation
    t = np.arange(N) / fs
    f = np.fft.fftfreq(N, 1/fs)[:N//2]
    tfr = tfr[:, :N//2]
    
    return t, f, np.abs(tfr)

def group_spwvd(group):
    return spwvd(group.values, fs)

def apply_SPWVD_to_windowed_data(filename, output_folder, truncation_loc, downsample_factor, overlap_window, relevant_col_names):
    # Load dataframe 
    df = pd.read_csv(filename)
    N_windows = int(np.max(df['Data_Window_idx'].values) + 1)

    # Reconstruct time array without overlap but with downsampling
    N_cycles_total = int(np.max(df['Time_cycle'].values))
    time_array_reconstructed = np.arange(1,N_cycles_total,downsample_factor)

    # Determine which part to truncate off final window with overlap
    last_window_idx = df['Data_Window_idx'].max()
    second_last_window_idx = last_window_idx - 1

    last_window_times = df.loc[df['Data_Window_idx'] == last_window_idx, 'Time_cycle'].reset_index(drop=True)
    second_last_window_times = df.loc[df['Data_Window_idx'] == second_last_window_idx, 'Time_cycle'].values
    sec_last_window_last_time = max(second_last_window_times)

    for i, val in enumerate(last_window_times):
        if val>sec_last_window_last_time:
            N_cut_fin_wind = i
            break

    # Find dimensions of window outputs
    time_dim = time_array_reconstructed.shape[0] # = int(truncation_loc/downsample_factor*N_windows)
    freq_dim = int(truncation_loc/downsample_factor/2)
    window_time_dim  = int(truncation_loc/downsample_factor)
    
    # Create folder for current file/sample results

    base_filename = os.path.basename(filename)
    # Extract the 'Sample1' part for any number
    sample_number = base_filename.split("Interp")[0]
    base_output_folder = output_folder
    extract_to_folder = os.path.join(base_output_folder, f"{sample_number}_spwvd")
    # Create the folder if it doesn't exist
    os.makedirs(extract_to_folder, exist_ok=True)

    print(f"\n Created (or verified existing) folder: {extract_to_folder}")

    print(f'\n Computing SPWVD for {sample_number}...')

    # Create file names each feature column tfr array to csv file to be stored
    tfr_feature_filenames = [
        os.path.splitext(os.path.basename(col_name))[0] + "_tfr_array.csv"
        for col_name in relevant_col_names
    ]

    # Compute SPWVD of each feature
    for i, col in enumerate(relevant_col_names):
        print(f'\n Processing feature {col}')
        # Group by window and compute SPWVD per window (spwvd_results is a Pandas Series with each row being a tuple (t, f, tfr))
        spwvd_results = df.groupby('Data_Window_idx')[col].apply(group_spwvd)

        # Initialize array for columns to store frequency in (to add next to time) and array for main tfr
        frequency = []
        tfr_concat = np.zeros((time_dim, freq_dim))

        # SPWVD per window
        for j in range(N_windows):
            # accessing different windows results
            window_t, window_f, window_tfr = spwvd_results.iloc[j]

            group_current = df.groupby('Data_Window_idx').get_group(j)
            window_time_array = group_current['Time_cycle'].values

            time = window_time_array
            tfr = window_tfr
            f = window_f

            #print('j=',j)
            #print(f'Time: {time}, \n Frequencies: {f}, \n TFR: {tfr}')
            if j==0:
                frequency = f
                start_time = 0
                end_time = int(truncation_loc/downsample_factor)
            
            # Add tfr array to total array
            if j>0 and j<(N_windows-1) and overlap_window>0:
                # Cut off overlap window
                cut_off_rows = int(overlap_window/downsample_factor)
                tfr = tfr[cut_off_rows:,:]
                start_time = end_time
                end_time = start_time + int(truncation_loc/downsample_factor-cut_off_rows)
                time = time[cut_off_rows:]
            if j==(N_windows-1):
                tfr = tfr[N_cut_fin_wind:,:]
                end_time = int(time_dim)
                start_time = end_time - tfr.shape[0]
                time = time[N_cut_fin_wind:]

            # Add non-overlapping parts of tfr to main array
            #print(f'Truncated: \n Time: {time}, \n Frequencies: {f}, \n TFR: {tfr}')

            tfr_concat[start_time:end_time,:] = tfr
            time_array_reconstructed[start_time:end_time] = time

            # print('Shape of tfr:', tfr.shape)
            # print('Shape of time:', time.shape)
            # print('Shape of freqs:', f.shape)
            # print('tfr', tfr)

            if plot_visual and j==0:
                print(f'\n Plotting logarithmic scale graph...')
                # Plot logarithmic scale
                plt.pcolormesh(time, f, 10 * np.log10(tfr.T), shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [cycles]')
                plt.title(f'Smoothed Pseudo Wigner-Ville Distribution of {col} (First window)')
                plt.colorbar(label='Magnitude [dB]')
                #plt.ylim(0, 300)
                plt.show()

                print(f'\n Plotting real scale graph...')
                # Plot normal scale
                plt.pcolormesh(time, f, tfr.T, shading='auto')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [cycles]')
                plt.title(f'Smoothed Pseudo Wigner-Ville Distribution of {col} (First window)')
                plt.colorbar(label='Magnitude')
                #plt.ylim(0, 300)
                plt.show()
        
        # Create dataframe with columns time, then all frequencies in increasing order (to save big tfr matrix as csv)
        df_time = pd.DataFrame({'Time_cycle': time_array_reconstructed})
        tfr_df = pd.DataFrame(tfr_concat, columns=frequency)
        final_df = pd.concat([df_time, tfr_df], axis=1)

        # Save tfr dataframe to csv file
        output_filename = tfr_feature_filenames[i]
        full_output_path = os.path.join(extract_to_folder, output_filename)
        final_df.to_csv(full_output_path, index=False)
        print(f"\n Time frequency array of feature: {col}, saved as {output_filename} in {full_output_path}")
            
        
                
        
    

#apply_SPWVD_to_windowed_data('Sample1_window_level_features_smoothed.csv')

for i, file in enumerate(Windowed_filenames): 
    print(f'\n Processing data file {i+1} out of {len(Windowed_filenames)}')
    apply_SPWVD_to_windowed_data(file, spwvd_results_folder, truncation_loc, downsample_factor, overlap_window, relevant_col_names)

print(f'\n All data processed and saved to {r"C:\Users\naomi\OneDrive\Documents\Low_Features\SPWVD_results"}')