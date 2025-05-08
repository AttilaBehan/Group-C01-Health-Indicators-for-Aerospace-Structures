import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cwt

# Define a Gaussian wavelet function
def gaussian_wavelet(length, width):
    """
    Creates a Gaussian wavelet (first derivative of Gaussian) for CWT.
    Similar to the 'gaus1' wavelet in MATLAB.
    
    Parameters:
    length : int
        Length of the wavelet
    width : float
        Width parameter (scale) of the wavelet
        
    Returns:
    wavelet : ndarray
        The Gaussian wavelet
    """
    x = np.linspace(-width, width, length)
    wavelet = -x * np.exp(-x**2 / 2)  # First derivative of Gaussian
    return wavelet

# Load data
df = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Assuming first column is time, second is signal
t = df.iloc[:, 0].values
signal = df.iloc[:, 1].values

# Define scales (widths for Gaussian)
widths = np.arange(1, 31)

# Perform CWT using Gaussian wavelet
# Note: We need to create a helper function that scipy's cwt can use
def gaussian_for_cwt(points, width):
    return gaussian_wavelet(points, width)

coef = cwt(signal, gaussian_for_cwt, widths)

# Plot the original signal
plt.figure(figsize=(15, 4))
plt.plot(t, signal)
plt.title("Input Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the scalogram using coolwarm (red to blue)
plt.figure(figsize=(15, 6))
plt.imshow(np.abs(coef), extent=[t[0], t[-1], widths[-1], widths[0]], 
           interpolation='bilinear', cmap='jet', aspect='auto', 
           vmin=0, vmax=np.abs(coef).max())
plt.title("Scalogram Using Gaussian Wavelet")
plt.xlabel("Time")
plt.ylabel("Width (Scale)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()