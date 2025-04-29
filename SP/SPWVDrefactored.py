import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from Feature_Extraction import CSV_to_Array
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from tftb.processing import smoothed_pseudo_wigner_ville
from sklearn.preprocessing import StandardScaler
from scipy.signal import get_window

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

