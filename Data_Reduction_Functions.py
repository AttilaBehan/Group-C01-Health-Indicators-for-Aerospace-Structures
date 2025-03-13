import numpy as np
from scipy.signal import hilbert, chirp
import pandas as pd
import matplotlib.pyplot as plt

print('test')
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 5 * t)

analytic_signal = hilbert(x)
amplitude_envelope = np.abs(analytic_signal)
Instantaneous_phase = np.angle(analytic_signal)
Instantaneous_frequency = np.diff(np.unwrap(np.angle(analytic_signal)))
print(Instantaneous_frequency)

plt.plot(t, x, label='Original Signal')
#plt.plot(t, analytic_signal, label='Analytic Signal')
plt.plot(t, amplitude_envelope, label='Amplitude Envelope')
plt.plot(t, Instantaneous_phase, label='Instantaneous Phase')
#plt.plot(t, Instantaneous_frequency, label='Instantaneous Frequency')
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