import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Feature_Extraction import CSV_to_Array
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d

print('Reading CSV file')
sample_amplitude, sample_risetime, sample_energy, sample_counts, sample_duration, sample_rms = CSV_to_Array('SP\\LowLevelFeaturesSample1.csv')

sample_amplitude = sample_amplitude[:2000]
sample_duration = sample_duration[150:2000]
sample_energy = sample_energy[:7002]
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

def wvd(signal, fs):
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

def spwvd(signal, fs, time_smoothing=15, freq_smoothing=15):
    """
    Smoothed Pseudo Wigner-Ville Distribution (SPWVD) with Gaussian smoothing.
    """
    N = len(signal)
    W = wvd(signal, fs)

    # Time smoothing (along time axis)
    time_kernel = gaussian(N, std=time_smoothing)
    time_kernel /= np.sum(time_kernel)
    W_smoothed_time = convolve1d(W, time_kernel, axis=0, mode='constant')

    # Frequency smoothing (along frequency axis)
    freq_kernel = gaussian(N, std=freq_smoothing)
    freq_kernel /= np.sum(freq_kernel)
    W_smoothed = convolve1d(W_smoothed_time, freq_kernel, axis=1, mode='constant')

    return W_smoothed

def wvd_simple(signal, fs):
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

def spwvd_simple(signal, fs, time_smoothing=15, freq_smoothing=15):
    W = wvd_simple(signal, fs)

    time_kernel = gaussian(W.shape[0], std=time_smoothing)
    time_kernel /= np.sum(time_kernel)
    W_time_smoothed = convolve1d(W, time_kernel, axis=0, mode='constant')

    freq_kernel = gaussian(W.shape[1], std=freq_smoothing)
    freq_kernel /= np.sum(freq_kernel)
    W_freq_smoothed = convolve1d(W_time_smoothed, freq_kernel, axis=1, mode='constant')

    return W_freq_smoothed


# Test signal
fs = 1000
t = np.linspace(0, 1, fs)
x = chirp(t, f0=100, f1=400, t1=1, method='quadratic') + chirp(t, f0=350, f1=50, t1=1, method='quadratic')
#print('signal', x)
N = len(x)
freqs = np.linspace(0, fs, N)

# SPWVD Matrices
# spwvd_matrix_v1 = spwvd(x, fs, time_smoothing=20, freq_smoothing=20)
# magnitude = np.abs(spwvd_matrix_v1)
# spwvd_matrix_v2 = spwvd_simple(x, fs, time_smoothing=20, freq_smoothing=20)
# wvd_matrix = wvd(x, fs)

# Try real data
print('Starting SPWVD...')
fs_Sample1 = len(sample_energy)/2931*2
spwvd_Sample1 = spwvd(sample_energy, fs_Sample1, time_smoothing=20, freq_smoothing=15)
magnitude_Sample1 = np.abs(spwvd_Sample1)
print('SPWVD completed')

# Features

# # --- Instat=ntaneous energy over time ---
# energy_over_time = magnitude.sum(axis=1)

# # --- Instantaneous Frequency (energy-weighted mean frequency) ---
# weighted_freq = (magnitude * freqs[np.newaxis, :]).sum(axis=1) / (energy_over_time + 1e-10)

# # --- Peak Frequency over time (frequency with highest energy at each time slice) ---
# peak_freq = freqs[np.argmax(magnitude, axis=1)]

# # --- Frequency Bandwidth over time ---
# # Compute standard deviation (energy-weighted)
# mean_sq = (magnitude * (freqs[np.newaxis, :] ** 2)).sum(axis=1) / (energy_over_time + 1e-10)
# bandwidth = np.sqrt(mean_sq - weighted_freq**2)

# # --- Spectral Entropy over time ---
# prob_dist = magnitude / (magnitude.sum(axis=1, keepdims=True) + 1e-10)
# entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10), axis=1)

# # --- Frequency Centroid over entire signal (already done) ---
# freq_centroid = (magnitude * freqs[np.newaxis, :]).sum() / (magnitude.sum() + 1e-10)


# Plotting SPWVD and WVD
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(spwvd_matrix_v1), origin='lower', aspect='auto',
#            extent=[0, 1, 0, fs], cmap='jet')
# plt.title('SPWVD – Version 1 (FFT-based)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(wvd_matrix), origin='lower', aspect='auto',
#            extent=[0, 1, 0, fs], cmap='jet')
# plt.title('WVD – Version 1 (FFT-based)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

print('Plotting results...')

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(spwvd_Sample1), origin='lower', aspect='auto',
           extent=[0, 1, 0, fs_Sample1], cmap='jet')
plt.title('SPWVD – Sample 1 Energy (FFT-based)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar()

plt.tight_layout()
plt.show()

# Plot features

# fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

# axs[0].plot(t, energy_over_time)
# axs[0].set_ylabel('Energy')
# axs[0].set_title('Instantaneous Energy')

# axs[1].plot(t, weighted_freq)
# axs[1].set_ylabel('Freq [Hz]')
# axs[1].set_title('Instantaneous Frequency (Mean)')

# axs[2].plot(t, peak_freq)
# axs[2].set_ylabel('Freq [Hz]')
# axs[2].set_title('Peak Frequency')

# axs[3].plot(t, bandwidth)
# axs[3].set_ylabel('Hz')
# axs[3].set_title('Frequency Bandwidth')

# axs[4].plot(t, entropy)
# axs[4].set_ylabel('Entropy')
# axs[4].set_title('Spectral Entropy')
# axs[4].set_xlabel('Time [s]')

# plt.tight_layout()
# plt.show()

# # --- Print the frequency centroid ---
# print(f"Overall Frequency Centroid: {freq_centroid:.2f} Hz")


# def wvd(x, fs):
#     """
#     Compute the Wigner-Ville Distribution (WVD) using FFT for efficiency.
    
#     Parameters:
#         x (numpy array): Input signal (1D).
#         fs (float): Sampling frequency.
    
#     Returns:
#         numpy array: Time-frequency representation (WVD).
#     """
#     N = len(x)
#     x = np.pad(x, (N//2, N//2), mode='constant')  # Zero-padding
#     X = hilbert(x)  # Compute the analytic signal
    
#     WVD = np.zeros((N, N), dtype=complex)  # Time-frequency matrix

#     for t in range(N):  # Time index
#         tau_max = min(t, N - t)  # Prevent out-of-bounds indexing
#         signal_product = X[t + tau_max::-1][:2*tau_max + 1] * np.conj(X[t - tau_max:t + tau_max + 1])
#         WVD[t, :] = np.fft.fftshift(np.fft.fft(signal_product, N))  # FFT for frequency domain
    
#     return np.abs(WVD)

# def spwvd(x, fs, time_smoothing=10, freq_smoothing=10):
#     """
#     Compute the Smoothed Pseudo Wigner-Ville Distribution (SPWVD).
    
#     Parameters:
#         x (numpy array): Input signal (1D).
#         fs (float): Sampling frequency.
#         time_smoothing (int): Gaussian smoothing in time domain.
#         freq_smoothing (int): Gaussian smoothing in frequency domain.
    
#     Returns:
#         numpy array: Smoothed Time-Frequency representation.
#     """
#     WVD = wvd(x, fs)  # Compute raw WVD
    
#     # Apply time smoothing (Gaussian kernel)
#     time_kernel = gaussian(WVD.shape[0], time_smoothing) / np.sum(gaussian(WVD.shape[0], time_smoothing))
#     WVD_smoothed_time = np.apply_along_axis(lambda m: np.convolve(m, time_kernel, mode='same'), axis=0, arr=WVD)
    
#     # Apply frequency smoothing (Gaussian kernel)
#     freq_kernel = gaussian(WVD.shape[1], freq_smoothing) / np.sum(gaussian(WVD.shape[1], freq_smoothing))
#     SPWVD = np.apply_along_axis(lambda m: np.convolve(m, freq_kernel, mode='same'), axis=1, arr=WVD_smoothed_time)
    
#     return SPWVD

# # Example: Chirp signal
# fs = 1000  # Sampling frequency
# t = np.linspace(0, 1, fs)  # Time vector
# x = chirp(t, f0=100, t1=1, f1=400, method='quadratic') + chirp(t, f0=350, t1=1, f1=50, method='quadratic')

# # Compute SPWVD
# spwvd_result = spwvd(x, fs, time_smoothing=20, freq_smoothing=20)

# # Plot the SPWVD
# plt.figure(figsize=(10, 6))
# plt.imshow(spwvd_result, aspect='auto', cmap='jet', origin='lower', extent=[0, 1, 0, fs / 2])
# plt.title('Smoothed Pseudo Wigner-Ville Distribution (SPWVD)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Magnitude')
# plt.show()


# '''

# New chatgpt optimized method

# "Let's refactor your Wigner-Ville Distribution (WVD) and Smoothed Pseudo Wigner-Ville Distribution (SPWVD) 
# implementations in a cleaner, more efficient, and mathematically correct way, using FFT-based computation, 
# which is standard in time-frequency analysis.

# Key improvements:
#     - Use lag-domain formulation.
#     - Apply FFT over the lag dimension to transform to frequency domain.
#     - Ensure correct zero-padding and indexing.
#     - Use Gaussian smoothing via convolution in both time and frequency directions.

# Benefits of refactoring:
#     - Robust index handling (no IndexError).
#     - FFT-based frequency domain representation (standard method).
#     - Efficient and clean smoothing using convolve1d.
#     - Easy to adapt to other kernel types if needed."

# '''

# def wvd_new(signal, fs):
#     """
#     Efficient Wigner-Ville Distribution using FFT-based method.
#     """
#     N = len(signal)
#     wvd_matrix = np.zeros((N, N), dtype=complex)

#     x = signal
#     for t in range(N):
#         tau_max = min(t, N - t - 1)
#         for tau in range(-tau_max, tau_max + 1):
#             n1 = t + tau
#             n2 = t - tau
#             if 0 <= n1 < N and 0 <= n2 < N:
#                 wvd_matrix[t, tau % N] = x[n1] * np.conj(x[n2])

#     # FFT along the lag (2nd) dimension to get frequency representation
#     wvd_matrix = np.fft.fftshift(np.fft.fft(wvd_matrix, axis=1), axes=1)
#     return np.real(wvd_matrix)

# def spwvd_new(signal, fs, time_smoothing=15, freq_smoothing=15):
#     """
#     Smoothed Pseudo Wigner-Ville Distribution with Gaussian smoothing.
#     """
#     N = len(signal)
#     # Step 1: Compute WVD
#     W = wvd_new(signal, fs)

#     # Step 2: Time smoothing (Gaussian along time axis)
#     time_kernel = gaussian(N, std=time_smoothing)
#     time_kernel /= np.sum(time_kernel)
#     W_smoothed_time = convolve1d(W, time_kernel, axis=0, mode='constant')

#     # Step 3: Frequency smoothing (Gaussian along frequency axis)
#     freq_kernel = gaussian(N, std=freq_smoothing)
#     freq_kernel /= np.sum(freq_kernel)
#     W_smoothed = convolve1d(W_smoothed_time, freq_kernel, axis=1, mode='constant')

#     return W_smoothed

# # Create a multi-component chirp signal
# fs = 1000  # Hz
# t = np.linspace(0, 1, fs)
# x = chirp(t, f0=100, f1=400, t1=1, method='quadratic') + chirp(t, f0=350, f1=50, t1=1, method='quadratic')

# # Compute new SPWVD
# spwvd_matrix = spwvd_new(x, fs, time_smoothing=20, freq_smoothing=20)

# # Old SPWVD
# spwvd_result = spwvd(x, fs, time_smoothing=20, freq_smoothing=20)

# # Plotting new SPWVD
# plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(spwvd_matrix.T), origin='lower', aspect='auto',
#            extent=[0, 1, 0, fs], cmap='jet')
# plt.title('New Smoothed Pseudo Wigner-Ville Distribution (SPWVD)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Magnitude')
# plt.tight_layout()
# plt.show()

# # Plot the old SPWVD
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(spwvd_result, aspect='auto', cmap='jet', origin='lower', extent=[0, 1, 0, fs / 2])
# plt.title('Old Smoothed Pseudo Wigner-Ville Distribution (SPWVD)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Magnitude')
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(spwvd_matrix.T), origin='lower', aspect='auto',
#            extent=[0, 1, 0, fs], cmap='jet')  # Time frequency distribution might have complex values, take abs of matrix vals
# plt.title('New Smoothed Pseudo Wigner-Ville Distribution (SPWVD)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Magnitude')
# plt.tight_layout()
# plt.show()

