import sys
import os

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
from Data_processing_SPWVD import downsample_factor, truncation_loc, overlap_window

# Make this true if you want to plot one of the color plots to check
plot_visual = True

''' This file contains the main function which computes the SPWVD and one which applies this to the windowed 
                                        csv files of the various signals which have been fully preprocessed. 

    It currently has the following structure:
     
      1. Imported downsample factor (I am going to change the format of this later to store these values elsewhere in a class)
      2. Array of file names to be processed and one of signal columns to process
      3. Definition of the sampling frequency based on the individual cycle length and downsampling factor 
      4. Main SPWVD funtion, apply function, and fuction which applies main function to windowed data files
                ^^^ as of now the function is not complete (only calculates it for the first window, I will complete this nexrt session)'''

# Windowed data file names
Windowed_filenames = ['Sample1_window_level_features_smoothed.csv', 'Sample1_window_level_features_smoothed.csv']

# Signal columns to be used
relevant_col_names = ['Amplitude_mean','Amplitude_std','Amplitude_max','Amplitude_min','Energy_sum','Energy_mean','Counts_sum','Duration_mean','RMS_mean','Rise-Time_mean','Amplitude_mean_smoothed','Amplitude_std_smoothed','Amplitude_max_smoothed','Amplitude_min_smoothed','Energy_sum_smoothed','Energy_mean_smoothed','Counts_sum_smoothed','Duration_mean_smoothed','RMS_mean_smoothed','Rise-Time_mean_smoothed']
relevant_col_names = ['Energy_sum_smoothed','Amplitude_mean_smoothed','Amplitude_std_smoothed']

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

def apply_SPWVD_to_windowed_data(filename):
    # Load dataframe 
    df = pd.read_csv(filename)

    N_windows = int(np.max(df['Data_Window_idx'].values) + 1)

    # Initialize place to store 

    for i, col in enumerate(relevant_col_names):
        # Step 1: Group by window and compute SPWVD per window (spwvd_results is a Pandas Series with each row being a tuple (t, f, tfr))
        spwvd_results = df.groupby('Data_Window_idx')[col].apply(group_spwvd)

        # Step 2: Initialize array for column to store frequency in (to add next to time)
        frequency = []
        time_dim = int(truncation_loc/downsample_factor)
        freq_dim = int(time_dim/2)
        tfr_per_window = torch.zeros(N_windows, time_dim, freq_dim)
        for j in range(N_windows):
            # accessing different windows results
            window_t, window_f, window_tfr = spwvd_results.iloc[j]

            group_current = df.groupby('Data_Window_idx').get_group(j)
            window_time_array = group_current['Time_cycle'].values

            time = window_time_array
            tfr = window_tfr
            f = window_f

            frequency.append(f)
            tfr_tensor = torch.from_numpy(tfr).float()
            tfr_per_window[j] = tfr_tensor

            print('Shape of tfr:', tfr.shape)
            print('tfr', tfr)

            if plot_visual and j==0:
                # Plot logarithmic scale
                plt.pcolormesh(time, f, 10 * np.log10(tfr.T), shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [cycles]')
                plt.title(f'Smoothed Pseudo Wigner-Ville Distribution of {col} (First window)')
                plt.colorbar(label='Magnitude [dB]')
                #plt.ylim(0, 300)
                plt.show()

                # Plot normal scale
                plt.pcolormesh(time, f, tfr.T, shading='auto')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [cycles]')
                plt.title(f'Smoothed Pseudo Wigner-Ville Distribution of {col} (First window)')
                plt.colorbar(label='Magnitude')
                #plt.ylim(0, 300)
                plt.show()
            
        
                
        
    

apply_SPWVD_to_windowed_data('Sample1_window_level_features_smoothed.csv')

# # Using SPWVD function (only on first window)
# time = downsampled_time_cycles[0]
# time.reshape(-1, 1)

# signal = downsampled_signals[0]
# signal.reshape(-1, 1)
# print(time.shape, signal.shape)

# fs = 1/(0.5*downsample_factor)

# t, f, tfr = spwvd(signal, fs)

# # Plot
# plt.pcolormesh(time, f, 10 * np.log10(tfr.T), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [cycles]')
# plt.title('Smoothed Pseudo Wigner-Ville Distribution (Custom Implementation)')
# plt.colorbar(label='Magnitude [dB]')
# #plt.ylim(0, 300)
# plt.show()

# # Plot
# plt.pcolormesh(time, f, tfr.T, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [cycles]')
# plt.title('Smoothed Pseudo Wigner-Ville Distribution (Custom Implementation)')
# plt.colorbar(label='Magnitude')
# #plt.ylim(0, 300)
# plt.show()

CAI_signal = False
Real_data = True
Cycle_windowing = True

if CAI_signal:
    data = pd.read_csv('SP\\CAI_test_dataset_spec2_mts.csv')


if Real_data:
    # Load data
    smoothed_df = pd.read_csv("Sample1_cycle_level_features_smoothed.csv")

    # Applying SPWVD:
    col_names = ['Time','Amplitude_mean','Amplitude_std','Amplitude_max','Amplitude_min','Energy_sum','Energy_mean','Counts_sum','Duration_mean','RMS_mean','Rise-Time_mean','Num_Events','Amplitude_mean_smoothed','Amplitude_std_smoothed','Amplitude_max_smoothed','Amplitude_min_smoothed','Energy_sum_smoothed','Energy_mean_smoothed','Counts_sum_smoothed','Duration_mean_smoothed','RMS_mean_smoothed','Rise-Time_mean_smoothed','Num_Events_smoothed']

    relevant_col_names = ['Time','Amplitude_mean','Amplitude_std','Amplitude_max','Amplitude_min','Energy_sum','Energy_mean','Counts_sum','Duration_mean','RMS_mean','Rise-Time_mean','Amplitude_mean_smoothed','Amplitude_std_smoothed','Amplitude_max_smoothed','Amplitude_min_smoothed','Energy_sum_smoothed','Energy_mean_smoothed','Counts_sum_smoothed','Duration_mean_smoothed','RMS_mean_smoothed','Rise-Time_mean_smoothed']
    # Example: using smoothed energy values as the signal
    signal = smoothed_df['Counts_sum_smoothed'].values
    signal = np.nan_to_num(signal)  # Replace NaNs or infs
    time_cycles = smoothed_df['Time'].values
    N_cycles_total = time_cycles.shape[0]
    print('total n of cycles:', N_cycles_total)

    # Downsampling and dividing signal into segments
    downsample_factor = 10
    if Cycle_windowing:
        truncation_loc = 40000
        if truncation_loc%downsample_factor!=0:
            print('Error: number of samples per window is not an integer -> (change downsampling factor to factor of number of samples per window)')
        overlap_window = 200
        N_cycle_windows = math.ceil(N_cycles_total/(truncation_loc-overlap_window))

        print('N_cycle_windows:', N_cycle_windows)

        # Creates array of start and stop indices of each window of data
        Cycle_windows = np.zeros((N_cycle_windows, 2))
        Cycle_windows[0, 1] = truncation_loc

        # Creates array of segments of data and time
        downsampled_signals = np.zeros((N_cycle_windows, int(truncation_loc/downsample_factor))) 
        downsampled_signals[0,:] = signal[:truncation_loc:downsample_factor]
        downsampled_time_cycles = np.zeros((N_cycle_windows, int(truncation_loc/downsample_factor))) 
        downsampled_time_cycles[0,:] = time_cycles[:truncation_loc:downsample_factor]
        # print('Downsampled signal and time')
        # print(downsampled_signals)
        # print(downsampled_time_cycles)
        for i in range(1,N_cycle_windows):
            print('i = ', i)
            if i!=(N_cycle_windows-1):
                j = i-1
                start = Cycle_windows[j, 1] - overlap_window
                start = int(start)
                stop = start + truncation_loc
                stop = int(stop)
                # Cycle_windows[i,0] = start
                # Cycle_windows[i,1] = stop
                # # print(signal[start:stop:downsample_factor])
                # # print('shape',signal[start:stop:downsample_factor].shape)
                # downsampled_signals[i,:] = signal[start:stop:downsample_factor]
                # downsampled_time_cycles[i,:] = time_cycles[start:stop:downsample_factor]
            elif i==(N_cycle_windows-1):
                stop = N_cycles_total
                start = int(N_cycles_total - truncation_loc)
            Cycle_windows[i,0] = start
            Cycle_windows[i,1] = stop
            # print(signal[start:stop:downsample_factor])
            # print('shape',signal[start:stop:downsample_factor].shape)
            downsampled_signals[i,:] = signal[start:stop:downsample_factor]
            downsampled_time_cycles[i,:] = time_cycles[start:stop:downsample_factor]
        print('downsampled signals:', downsampled_signals)
        # print('downsampled time cycles:', downsampled_time_cycles)
        
    else:
        downsampled_signal = signal[::downsample_factor]
        downsampled_time_cycles = time_cycles[::downsample_factor]
        print('downsampled signal:', downsampled_signal)
        

    # Apply SPWVD to the downsampled signal    

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

        # Time smoothing
        ''' 
        n = current point in time
        tau = range of time lags

        '''
        for n in range(N):
            tau_max = min(n, N-1-n, Ntime//2) # Maximum possible lag without out-of-bounds
            tau = np.arange(-tau_max, tau_max+1) # Symmetric lags around n
            indices = n + tau
            product = x_analytic[n + tau] * np.conj(x_analytic[n - tau]) # Autocorrelation
            tfr[n, :] = np.fft.fft(product * window_time[tau_max + tau], N) # Windowed & Fourier-transformed
        
        # Frequency smoothing
        for m in range(N):
            nu_max = min(m, N-1-m, Nfreq//2) # Limits to avoid boundary issues
            nu = np.arange(-nu_max, nu_max+1) # Frequency shifts
            tfr[:, m] = np.convolve(tfr[:, m], window_freq[nu_max + nu], mode='same')
        
        # Time-frequency representation
        t = np.arange(N) / fs
        f = np.fft.fftfreq(N, 1/fs)[:N//2]
        tfr = tfr[:, :N//2]
        
        return t, f, np.abs(tfr)

    # Using SPWVD function (only on first window)
    time = downsampled_time_cycles[0]
    time.reshape(-1, 1)
    
    signal = downsampled_signals[0]
    signal.reshape(-1, 1)
    print(time.shape, signal.shape)

    fs = 1/(0.5*downsample_factor)

    t, f, tfr = spwvd(signal, fs)

    # Plot
    plt.pcolormesh(time, f, 10 * np.log10(tfr.T), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [cycles]')
    plt.title('Smoothed Pseudo Wigner-Ville Distribution (Custom Implementation)')
    plt.colorbar(label='Magnitude [dB]')
    #plt.ylim(0, 300)
    plt.show()

    # Plot
    plt.pcolormesh(time, f, tfr.T, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [cycles]')
    plt.title('Smoothed Pseudo Wigner-Ville Distribution (Custom Implementation)')
    plt.colorbar(label='Magnitude')
    #plt.ylim(0, 300)
    plt.show()

