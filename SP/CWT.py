import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cwt, ricker

# Load data
df = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Assuming first column is time, second is signal
t = df.iloc[:, 0].values
signal = df.iloc[:, 1].values

# Define scales (widths for Ricker)
widths = np.arange(1, 31)

# Perform CWT using Ricker (Mexican hat) wavelet
coef = cwt(signal, ricker, widths)

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
           interpolation='bilinear', cmap='coolwarm', aspect='auto', vmin=0, vmax=np.abs(coef).max())
plt.title("Scalogram Using SciPy (Ricker Wavelet)")
plt.xlabel("Time")
plt.ylabel("Width (Scale)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()
