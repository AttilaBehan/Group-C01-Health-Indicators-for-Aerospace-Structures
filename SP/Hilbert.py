import numpy as np
#import tftb
from scipy.signal import hilbert, chirp
from scipy.signal.windows import gaussian
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

''' SORRY THIS SCRIPT IS RLLY MESSY, JUST TRYING STUFF OUT, DON'T USE THIS FOR ANYTHING'''


# Example: Create a signal (chirp signal)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)  # Time vector
x = chirp(t, f0=100, t1=1, f1=400, method='quadratic') + chirp(t, f0=350, t1=1, f1=50, method='quadratic')

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