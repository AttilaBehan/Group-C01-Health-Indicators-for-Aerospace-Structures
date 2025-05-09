import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Assuming first column is time, second is signal
t = df.iloc[:, 0].values
signal = df.iloc[:, 1].values

# Define scales
scales = np.arange(1, 31)

# Perform CWT using Gaussian wavelet
coef, freqs = pywt.cwt(signal, scales, 'gaus1')

# Plot the original signal
plt.figure(figsize=(15, 4))
plt.plot(t, signal)
plt.title("Input Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the scalogram
plt.figure(figsize=(15, 6))
plt.imshow(np.abs(coef), extent=[t[0], t[-1], scales[-1], scales[0]],
           interpolation='bilinear', cmap='coolwarm', aspect='auto', vmin=0, vmax=np.abs(coef).max())
plt.title("CWT Scalogram using Gaussian Wavelet (gaus1)")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()
