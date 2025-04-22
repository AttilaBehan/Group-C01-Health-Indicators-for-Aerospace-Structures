import pywt
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy import stats
import pandas as pd
import numpy as np

chirp_data = True
test_data = True

# Load the CSV file
# Replace 'your_file.csv' with your actual file path
df = pd.read_csv('SP\\LowLevelFeaturesSample1.csv')

# Show available column names
print("Column names:", df.columns.tolist())

# Split each column into separate numpy arrays
column_arrays = {}
for column in df.columns:
    column_arrays[column] = df[column].to_numpy()

# Accessing arrays by column name
time_array = column_arrays['Time'] 
amplitude_array = column_arrays['Amplitude']  
rise_time_array = column_arrays['Rise-Time']
Energy_array = column_arrays['Energy']
Counts_array = column_arrays['Counts']
Duration_array = column_arrays['Duration']
RMS_array = column_arrays['RMS']

time_array_truncated = time_array[:1000]
amplitude_array_truncated = amplitude_array[:1000]

# Print the arrays
# print("Time Array:\n", time_array)
# print("Amplitude Array:\n", amplitude_array)



#coeffs, freqs = pywt.cwt(data, scales, wavelet)

def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)


def make_chirp(t, t0, a):
    frequency = (a * (t + t0)) ** 2
    chirp = np.sin(2 * np.pi * frequency * t)
    return chirp, frequency


if __name__ == "__main__":
    # generate signal
    time = np.linspace(0, 1, 2000)
    chirp1, frequency1 = make_chirp(time, 0.2, 9)
    chirp2, frequency2 = make_chirp(time, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(time, 0.5, 0.2)

    if chirp_data:
        # plot signal
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(time, chirp)
        axs[1].plot(time, frequency1, label='freq 1')
        axs[1].plot(time, frequency2, label='freq 2')
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Time (s)")
        axs[0].set_ylabel("Signal")
        axs[1].set_ylabel("True frequency (Hz)")
        plt.suptitle("Input signal")
        plt.legend()
        plt.show()

    if test_data:
        # plot signal
        plt.plot(time_array_truncated, amplitude_array_truncated)
        plt.xlabel('Time')
        plt.suptitle("Input signal")
        plt.legend()
        plt.show()

'''Apply the Continuous Wavelet Transform using a complex Morlet wavlet with a given center frequency and bandwidth (cmor1.5-1.0). 
    Then plot the scaleogram, which is the 2D plot of the signal strength vs. time and frequency.
    
    pywt.cwt returns 2D array of wavelet coefficients and 1D array of frequencies:
        wavelet coeff: rows = different scales or frequencies, columns = time instances on the signal.
            magnitude of coeff indicates strength of signal at each scale and time point
            coeff can be used to visualize how signal's freq content evolves over time
        frequencies: correspond to different scales used in CWT, inversely related to frequencies 
            (exact freq resolution depends on the wavelet type and scale range chosen), 
            array allows mapping each row into coeff matrix to a specific freq

    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    '''

# perform CWT
wavelet = "cmor1.5-1.0"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)  # Logarithmic spacing 
sampling_period = np.diff(time).mean()
cwtmatr, freqs = pywt.cwt(chirp, widths, wavelet, sampling_period=sampling_period) 
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

# plot fourier transform for comparison
yf = rfft(chirp)
xf = rfftfreq(len(chirp), sampling_period)
plt.semilogx(xf, np.abs(yf))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_title("Fourier Transform")
plt.tight_layout()
plt.show()

if test_data:
    # perform CWT
    wavelet = "cmor1.5-1.0"
    # logarithmic scale for scales, as suggested by Torrence & Compo:
    widths = np.geomspace(1, 1024, num=100)  # Logarithmic spacing 
    sampling_period = np.diff(time_array_truncated).mean()
    cwtmatr, freqs = pywt.cwt(amplitude_array_truncated, widths, wavelet, sampling_period=sampling_period) 
    # absolute take absolute value of complex result
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    # plot result using matplotlib's pcolormesh (image with annoted axes)
    fig, axs = plt.subplots(2, 1)
    pcm = axs[0].pcolormesh(time_array_truncated, freqs, cwtmatr)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
    fig.colorbar(pcm, ax=axs[0])

# perform CWT with different wavelets on same signal and plot results
def plot_wavelet(time, data, wavelet, title, ax):
    widths = np.geomspace(1, 1024, num=75)
    cwtmatr, freqs = pywt.cwt(
        data, widths, wavelet, sampling_period=np.diff(time).mean()
    )
    cwtmatr = np.abs(cwtmatr[:-1, :-1])
    pcm = ax.pcolormesh(time, freqs, cwtmatr)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax)
    return ax

wavelets = [f"cmor{x:.1f}-{y:.1f}" for x in [0.5, 1.5, 2.5] for y in [0.5, 1.0, 1.5]]
fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True)
for ax, wavelet in zip(axs.flatten(), wavelets):
    plot_wavelet(time, chirp, wavelet, wavelet, ax)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Scaleograms of the same signal with different wavelets")
plt.show()

# Feature extraction
coefficients = cwtmatr
frequencies = freqs

# Instantaneous frequency (frequency associated with the peak value of the CWT at each time instance)
#f_instantaneous = frequencies[np.argmax(np.abs(coefficients[time, :]))]

# Energy in frequency band (sum of squared magnitudes of coeff for specific range of scales/freqs)
freq_min = 15
freq_max = 35
scales_range = np.where((frequencies >= freq_min) & (frequencies <= freq_max))[0]
energy_band = np.sum(np.abs(coefficients[scales_range, :]) ** 2, axis=0)

plt.plot(time[1:], energy_band)
plt.title(f'Energy in {freq_min}-{freq_max} Hz band')
plt.xlabel('Time [s]')
plt.ylabel('Energy')
plt.show()

# Statistical features of coefficients to capture distribution of energy acreoss time and freq

mean_energy = np.mean(np.abs(coefficients))
variance_energy = np.var(np.abs(coefficients))
skewness_energy = stats.skew(np.abs(coefficients))
kurtosis_energy = stats.kurtosis(np.abs(coefficients))