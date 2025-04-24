import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from Feature_Extraction import CSV_to_Array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from tftb.processing import smoothed_pseudo_wigner_ville
from sklearn.preprocessing import StandardScaler
from scipy.signal import get_window

random_signal = False
CAI_signal = True
Real_data = True

if CAI_signal:
    data = pd.read_csv('SP\\CAI_test_dataset_spec2_mts.csv')

print('Reading CSV file...')
#sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms = CSV_to_Array('SP\\LowLevelFeaturesSample1.csv')
df = pd.read_csv('SP\\Sample1Interp.csv')

df = df.iloc[:, 1:]  # Keep all columns except first one (empty)
print(df)

df = df.iloc[:6000]
print(df.size)
df_not_to_normalize = df.drop(columns=['Amplitude'])

# Select the 'Amplitude' column
df_to_normalize = df['Amplitude']

# Reshape it to a 2D array (n_samples, 1 feature)
df_to_normalize = df_to_normalize.values.reshape(-1, 1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply fit_transform to normalize the data
df_normalized = scaler.fit_transform(df_to_normalize)

# If you want to convert it back to a pandas DataFrame (with the same index and column name)
df_normalized = pd.DataFrame(df_normalized, columns=['Amplitude'], index=df.index)
df_normalized['Time'] = df['Time']
#df_normalized = df_normalized[::10]
print(df_normalized)

print('Data collected')

'''
Not own work, put own code into chatgpt and got my code improved:
"Let's refactor your Wigner-Ville Distribution (WVD) and Smoothed Pseudo Wigner-Ville 
Distribution (SPWVD) implementations properly using an FFT-based approach. 
This will make the computation much more efficient, accurate, and numerically stable.

Key improvements:
    - Use FFT instead of direct summation for computing WVD efficiently.
    - Use a proper lag structure: Instead of iterating over tau manually, we'll handle it using signal padding and FFT shifting.
    - Fix the SPWVD smoothing by applying Gaussian windows correctly in time and frequency domains.
    - Optimize Memory Usage by avoiding unnecessary loops.
    
What's improved:
    - FFT-based WVD Computation: Instead of nested loops, we use the Hilbert transform and FFT to compute the WVD much faster.
    - Proper Time-Frequency Representation: The lag structure is handled automatically instead of using manual index shifting.
    - Efficient Gaussian Smoothing: We apply separate Gaussian filters along the time and frequency axes for correct smoothing.
    - Prevents Out-of-Bounds Errors: We zero-pad the signal and ensure no indexing issues occur.
    
Performance gain:
    - Using FFT-based computation makes this implementation orders of magnitude faster than your original nested loops method.
    - For large signals, this is crucial for real-time applications."
'''

''' 
HOW IT WOKRS 

1. Turns real valued AE signal into analytic signal (complex valued and non-negative frequencies) (Hilbert Transform removes 
    negative frquencies and gives instantaneous amplitude and phase for time-freq analysis)
2. Create WVD matrix size of signal (2D, rows = time, col = freq) to store time-freq val for each instant
3. Loop over each time and calc auto-correlation in lag, uses WV formula W(t,τ)=x(t+τ)⋅x*(t-τ)
    for each time t compute how signal correllates with itself at delays +-tau (captures how energy spreads in time and freq)
    Values stored in lag domain (τ-axis) for each time (%N maps -ve lags into matrix index range safely)
4. Apply FFT across lag axis (convert lag-> freq) (W in time vs lag rn not frequency, apply FFT across columns (lags) to transform into time vs freq)
    fftshift centers the freq axis
5. Smoothing time domain (bc WVD can have interference terms or artifacts, reduces noise-like interference)
    Use Gaussian filter, usually used in SPWVD
6. SMoothing frequency domain (to suppress sharp fluctuations that don't correspond to real features in the AE data)
'''

def wvd_og(signal):
    """
    Wigner-Ville Distribution (WVD) using FFT across lag axis.
    """
    N = len(signal)
    x = hilbert(signal)  # Analytic signal
    W = np.zeros((N, N), dtype=complex)

    for t in range(N):
        for tau in range(-min(t, N - t - 1), min(t, N - t - 1) + 1):
            t_plus = t + tau
            t_minus = t - tau
            if 0 <= t_plus < N and 0 <= t_minus < N:
                W[t, tau % N] = x[t_plus] * np.conj(x[t_minus])

    # Apply FFT along the lag axis (columns) → frequency domain
    W = np.fft.fftshift(np.fft.fft(W, axis=1), axes=1)
    return np.real(W)

def spwvd_og(signal, time_smoothing=15, freq_smoothing=15):
    """
    Smoothed Pseudo Wigner-Ville Distribution (SPWVD) with Gaussian smoothing.
    """
    N = len(signal)
    W = wvd_og(signal)

    # Time smoothing (along time axis)
    time_kernel = gaussian(N, std=time_smoothing)
    time_kernel /= np.sum(time_kernel)
    W_smoothed_time = convolve1d(W, time_kernel, axis=0, mode='constant')

    # Frequency smoothing (along frequency axis)
    freq_kernel = gaussian(N, std=freq_smoothing)
    freq_kernel /= np.sum(freq_kernel)
    W_smoothed = convolve1d(W_smoothed_time, freq_kernel, axis=1, mode='constant')

    return W_smoothed

def wvd_simple(signal):
    """
    WVD using zero-padded lag computation with FFT.
    Easier to understand but not optimal in speed.
    """
    N = len(signal)
    x = hilbert(signal)
    W = np.zeros((N, N), dtype=complex)

    for t in range(N):
        for tau in range(-N//2, N//2):
            t1 = t + tau
            t2 = t - tau
            if 0 <= t1 < N and 0 <= t2 < N:
                W[t, tau % N] = x[t1] * np.conj(x[t2])
    
    W = np.fft.fftshift(np.fft.fft(W, axis=1), axes=1)
    return np.real(W)

def spwvd_simple(signal, time_smoothing=15, freq_smoothing=15):
    W = wvd_simple(signal)

    time_kernel = gaussian(W.shape[0], std=time_smoothing)
    time_kernel /= np.sum(time_kernel)
    W_time_smoothed = convolve1d(W, time_kernel, axis=0, mode='constant')

    freq_kernel = gaussian(W.shape[1], std=freq_smoothing)
    freq_kernel /= np.sum(freq_kernel)
    W_freq_smoothed = convolve1d(W_time_smoothed, freq_kernel, axis=1, mode='constant')

    return W_freq_smoothed


if random_signal:
    # Test signal
    fs = 1000
    t = np.linspace(0, 1, fs)
    x = chirp(t, f0=100, f1=400, t1=1, method='quadratic') + chirp(t, f0=350, f1=50, t1=1, method='quadratic')
    #print('signal', x)
    N = len(x)
    freqs = np.linspace(0, fs, N)

    # SPWVD Matrices
    spwvd_matrix_v1 = spwvd_og(x, time_smoothing=20, freq_smoothing=20)
    magnitude = np.abs(spwvd_matrix_v1)
    spwvd_matrix_v2 = spwvd_simple(x, time_smoothing=20, freq_smoothing=20)
    wvd_matrix = wvd_og(x)
    # Plot
    print('Plotting results...')

    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(spwvd_matrix_v1), origin='lower', aspect='auto', cmap='jet')
    plt.title('SPWVD – Sample 1 Energy (FFT-based)')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Compute SPWVD
    freq_bins = 4096
    twindow = np.array([0.001])
    fwindow = np.array([0.001])
    #tfr, t_vals, f_vals, _ = smoothed_pseudo_wigner_ville(df["Amplitude"], timestamps=df["Time"], freq_bins=freq_bins, twindow=twindow, fwindow=fwindow)
    print('Starting SPWVD on test...')
    tfr = smoothed_pseudo_wigner_ville(x, timestamps=t, freq_bins=freq_bins, twindow=twindow, fwindow=fwindow)
    print('Second SPWVD completed')
    print(tfr.shape)
    print(type(tfr))
    print(tfr)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(tfr), aspect='auto', origin='lower', cmap='jet', extent=[0, tfr.shape[1], 0, tfr.shape[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (samples)')
    plt.ylabel('Frequency bins')
    plt.title('Time-Frequency Representation (TFR)')
    plt.show()


if Real_data:
    # Load AE data (replace with your actual file path)
    df = pd.read_csv("SP\\Sample1Interp.csv")

    # Step 1: Group by cycle (Time column) and compute per-cycle features
    grouped = df.groupby('Time')

    # Define aggregations for useful features
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

    # Step 3: Add AE event count per cycle
    agg_df['Num_Events'] = grouped.size().values

    # Step 4: Handle NaNs (e.g., std when only one AE event per cycle)
    agg_df.fillna(0, inplace=True)  # Or use a different strategy like interpolation

    # Step 5: Apply rolling smoothing (e.g., 5-cycle window)
    window_size = 5
    smoothed_df = agg_df.copy()
    rolling_cols = [col for col in agg_df.columns if col not in ['Time']]  # exclude 'Time'

    for col in rolling_cols:
        smoothed_df[f'{col}_smoothed'] = agg_df[col].rolling(window=window_size, min_periods=1).mean()

    # Step 6: Preview
    print(smoothed_df.head())

    # Optional: Save to CSV
    smoothed_df.to_csv("Sample1_cycle_level_features_smoothed.csv", index=False)
    

    # Applying SPWVD:

    # Example: using smoothed energy values as the signal
    signal = smoothed_df['Energy_sum_smoothed'].values
    signal = np.nan_to_num(signal)  # Replace NaNs or infs
    downsample_factor = 10
    trunkation_loc = 4000
    downsampled_signal = signal[:trunkation_loc]
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
        
        # Initialize output
        tfr = np.zeros((N, N), dtype=complex)
        
        # Compute analytic signal
        x_analytic = np.fft.fft(x)
        x_analytic[N//2+1:] = 0
        x_analytic = np.fft.ifft(x_analytic)
        
        # Compute Wigner-Ville with smoothing
        for n in range(N):
            tau_max = min(n, N-1-n, Ntime//2)
            tau = np.arange(-tau_max, tau_max+1)
            indices = n + tau
            product = x_analytic[n + tau] * np.conj(x_analytic[n - tau])
            tfr[n, :] = np.fft.fft(product * window_time[tau_max + tau], N)
        
        # Frequency smoothing
        for m in range(N):
            nu_max = min(m, N-1-m, Nfreq//2)
            nu = np.arange(-nu_max, nu_max+1)
            tfr[:, m] = np.convolve(tfr[:, m], window_freq[nu_max + nu], mode='same')
        
        # Time-frequency representation
        t = np.arange(N) / fs
        f = np.fft.fftfreq(N, 1/fs)[:N//2]
        tfr = tfr[:, :N//2]
        
        return t, f, np.abs(tfr)

    # Example usage
    fs = downsampled_signal.shape[0]
    print('fs:', fs)
    time = np.arange(1, fs+1)
    time.reshape(-1, 1)
    
    signal = downsampled_signal
    signal.reshape(-1, 1)
    print(time.shape, signal.shape)

    fs = 2

    t, f, tfr = spwvd(signal, fs)

    # Plot
    plt.pcolormesh(time, f, 10 * np.log10(tfr.T), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [cycles]')
    plt.title('Smoothed Pseudo Wigner-Ville Distribution (Custom Implementation)')
    plt.colorbar(label='Power [dB]')
    #plt.ylim(0, 300)
    plt.show()

    result = smoothed_pseudo_wigner_ville(downsampled_signal)

    # Inspect the result
    print(type(result))  # Check the type of the result to understand the structure

    # It's likely a tuple with the first element as the time-frequency representation (TFR)
    tfr = result[0]  # Time-frequency representation
    # Inspect the shape of tfr
    print(f"Shape of tfr: {tfr.shape}")
    # If tfr is 1D, reshape it to 2D for plotting
    if tfr.ndim == 1:
        tfr = np.expand_dims(tfr, axis=0)

    t = result[1]    # Time axis
    f = result[2]    # Frequency axis

    #tfr, t, f = smoothed_pseudo_wigner_ville(downsampled_signal)

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(tfr), shading='auto')
    plt.xlabel("Cycle")
    plt.ylabel("Pseudo-frequency (cycles⁻¹)")
    plt.title("Smoothed Pseudo Wigner-Ville Distribution (SPWVD)")
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()






# Try real data
print('Starting SPWVD...')
# fs_Sample1 = len(sample_energy)/2931*2
# spwvd_Sample1 = spwvd(df["Amplitude"], time_smoothing=20, freq_smoothing=15)
# magnitude_Sample1 = np.abs(spwvd_Sample1)

# Compute SPWVD
freq_bins = 128  # 16384  # np.linspace(0, 100, 256)
twindow = np.array([0.1])
fwindow = np.array([0.1])
#tfr, t_vals, f_vals, _ = smoothed_pseudo_wigner_ville(df["Amplitude"], timestamps=df["Time"], freq_bins=freq_bins, twindow=twindow, fwindow=fwindow)
print('SPWVD completed')
print('Starting second SPWVD...')
tfr = smoothed_pseudo_wigner_ville(df["Energy"], timestamps=df["Time"], freq_bins=freq_bins, twindow=twindow, fwindow=fwindow)
print('Second SPWVD completed')
print(tfr.shape)
print(type(tfr))
print(tfr)

# # Apply inverse transformation to the reshaped data
tfr_original_scale = scaler.inverse_transform(tfr)



plt.figure(figsize=(10, 6))
plt.imshow(np.abs(tfr_original_scale), aspect='auto', origin='lower', cmap='jet', extent=[0, tfr.shape[1], 0, tfr.shape[0]])
plt.colorbar(label='Magnitude')
plt.xlabel('Time (samples)')
plt.ylabel('Frequency bins')
plt.title('Time-Frequency Representation (TFR)')
plt.show()
# Plot
# plt.pcolormesh(t_vals, f_vals, np.abs(tfr), shading='auto')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Cycles/sample)")
# plt.title("Smoothed Pseudo Wigner-Ville Distribution (SPWVD)")
# plt.colorbar(label="Amplitude")
# plt.show()
