import numpy as np
#import tftb
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

''' SORRY THIS SCRIPT IS RLLY MESSY, JUST TRYING STUFF OUT, DON'T USE THIS FOR ANYTHING'''



def wvd(x, fs):
    """
    Compute the Wigner-Ville Distribution (WVD) of a signal.
    
    x: The signal (1D array)
    fs: Sampling frequency
    
    Returns: WVD of the signal as a 2D array (time-frequency representation)
    """
    N = len(x)
    W = np.zeros((N, N), dtype=complex)
    time = np.arange(N) / fs

    # Wigner-Ville distribution computation
    # for t in range(N):
    #     for tau in range(-t, N-t):
    #         W[t, tau] = np.sum(x[t + tau] * np.conj(x[t - tau]) * np.exp(-2j * np.pi * fs * tau))
    
    for t in range(N):
        for tau in range(-N//2, N//2):
            t_plus_tau = t + tau
            t_minus_tau = t - tau

            if 0 <= t_plus_tau < N and 0 <= t_minus_tau < N:
                W[t, tau % N] = x[t_plus_tau] * np.conj(x[t_minus_tau]) * np.exp(-2j * np.pi * fs * tau)

    return W

def spwvd(x, fs, time_smoothing=10, freq_smoothing=10):
    """
    Compute the Smoothed Pseudo Wigner-Ville Distribution (SPWVD).
    
    x: The signal (1D array)
    fs: Sampling frequency
    time_smoothing: Time-domain smoothing (Gaussian width)
    freq_smoothing: Frequency-domain smoothing (Gaussian width)
    
    Returns: SPWVD of the signal as a 2D array (time-frequency representation)
    """
    # Step 1: Compute the Wigner-Ville Distribution (WVD)
    W = wvd(x, fs)
    
    # Step 2: Apply time-domain smoothing (Gaussian kernel)
    time_kernel = gaussian(len(x), time_smoothing)
    W_smoothed_time = np.zeros_like(W)
    for f in range(len(x)):
        W_smoothed_time[f] = np.convolve(W[f], time_kernel, mode='same')
    
    # Step 3: Apply frequency-domain smoothing (Gaussian kernel)
    freq_kernel = gaussian(len(x), freq_smoothing)
    W_smoothed_freq = np.zeros_like(W_smoothed_time)
    for t in range(len(x)):
        W_smoothed_freq[t] = np.convolve(W_smoothed_time[t], freq_kernel, mode='same')
    
    return W_smoothed_freq

# Example: Create a signal (chirp signal)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)  # Time vector
x = chirp(t, f0=100, t1=1, f1=400, method='quadratic') + chirp(t, f0=350, t1=1, f1=50, method='quadratic')

# Compute SPWVD
spwvd_result = spwvd(x, fs, time_smoothing=20, freq_smoothing=20)

# Plot the SPWVD
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(spwvd_result), aspect='auto', cmap='jet', origin='lower', extent=[0, 1, 0, fs / 2])
plt.title('Smoothed Pseudo Wigner-Ville Distribution (SPWVD)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Magnitude')
plt.show()


# # WVD
# t = np.linspace(0, 1, 1000)
# signal = (3+t)*np.sin(2 * np.pi * 5 * t**2)
# fs = 0.001

# wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=t)
# tfr_wvd, t_wvd, f_wvd = wvd.run()  # tfr = time frequency representation

# print(tfr_wvd)
# # here t_wvd is the same as our ts, and f_wvd are the "normalized frequencies"
# # so we will not use them and construct our own.

# # SPWVD

# wvd(x,fs,"smoothedPseudo",NumFrequencyPoints=501,NumTimePoints=502)


# Hilbert
t = np.linspace(0, 1, 1000)
x = (3+t)*np.sin(2 * np.pi * 5 * t**2)
fs = 0.001

analytic_signal = hilbert(x)
amplitude_envelope = np.abs(analytic_signal)
Instantaneous_phase = np.unwrap(np.angle(analytic_signal))
Instantaneous_frequency = np.diff(Instantaneous_phase) / (2 * pi) *fs
print(Instantaneous_frequency)

plt.plot(t, x, label='Original Signal')
#plt.plot(t, analytic_signal, label='Analytic Signal')
plt.plot(t, amplitude_envelope, label='Amplitude Envelope')
plt.plot(t, Instantaneous_phase, label='Instantaneous Phase')
plt.plot(t[1:], Instantaneous_frequency, label='Instantaneous Frequency')
plt.legend()
plt.show()

# X = np.array([[5., 2., 8., 13., 0.],
#     [1., 0., 4., 7., 2.],
#     [3., 0., 1., 2., 1.],
#     [0., 4., 3., 2., 1.],
#     [1., 2., 0., 0., 6.]])

# print(X)

# # PCA
# Xavg = np.mean(X, axis=1)
# print(Xavg)
# #B = X - np.tile(Xavg)

# # Economy SVD
# U, S, VT = np.linalg.svd(X, full_matrices=False)