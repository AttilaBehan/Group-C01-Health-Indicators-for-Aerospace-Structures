import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def perform_fft(input_csv, output_csv, *, cycle_duration=0.5):
    """
    Read `input_csv` with columns
        Time (in cycles), Amplitude, Rise-Time, Energy, Counts, Duration, RMS
    and write `output_csv` whose first column is Frequency (Hz) and the rest are
    single-sided FFT magnitudes of every signal column.

    The sampling frequency fs is **inferred**:
        1. Convert Time → seconds by multiplying with `cycle_duration`.
        2. Take the median of the non-zero Δt’s → dt.
        3. fs = 1 / dt.
    """
    # -------- Load & sanity-check --------------------------------------------------
    df = pd.read_csv(input_csv).dropna(how="all")
    if "Time" not in df.columns:
        raise ValueError("Missing 'Time' column needed for fs estimation.")

    # ------------------ sampling rate ----------------------------------------------
    t_sec = df["Time"].to_numpy(dtype=float) * cycle_duration
    diffs = np.diff(t_sec)
    diffs = diffs[diffs > 0]                 # ignore duplicates (Δt = 0)
    if diffs.size == 0:
        raise ValueError("All Time values are identical – can’t infer fs.")
    dt  = np.median(diffs)
    fs  = 1.0 / dt

    # -------- Frequency grid -------------------------------------------------------
    N      = len(t_sec)
    freqs  = fftfreq(N, d=1/fs)
    pos    = freqs >= 0
    freqs  = freqs[pos]

    # -------- Build output frame ---------------------------------------------------
    out = pd.DataFrame({"Frequency (Hz)": freqs})

    for col in (c for c in df.columns if c != "Time"):
        x   = df[col].to_numpy()
        Xf  = fft(x)
        Xf  = np.abs(Xf) / N          # magnitude spectrum, normalised
        Xf[1:N//2] *= 2               # single-sided scaling
        out[col] = Xf[pos]

    # -------- Save & return --------------------------------------------------------
    out.to_csv(output_csv, index=False)
    return out

# Example call:
# perform_fft("Sample1.csv", "Sample1FFT.csv")

output_csv_path = r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1FFT.csv'
results_df = perform_fft(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv', output_csv_path)


'''
---------------------
'''


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq

# def perform_fft(file_path):
#     """
#     Perform FFT on all columns except 'Time' from a CSV file.
#     Returns a dictionary with column names as keys and tuples of (frequencies, normalized FFT values).
#     Also plots the normalized FFT of the 'Amplitude' column.
#     """
#     # Parameters
#     sampling_period = 0.5  # seconds per cycle
#     fs = 1 / sampling_period  # samples per second (Hz)

#     # Load the CSV file
#     df = pd.read_csv(file_path)

#     # Drop 'Time' column if present
#     if 'Time' in df.columns:
#         df = df.drop(columns=['Time'])

#     results = {}

#     for column in df.columns:
#         data = df[column].values
#         N = len(data)

#         # Perform FFT
#         data_fft = fft(data)

#         # Normalize
#         data_fft = data_fft / N

#         # Frequency bins
#         freqs = fftfreq(N, d=sampling_period)

#         # Only keep positive frequencies
#         mask = freqs >= 0
#         freqs_pos = freqs[mask]
#         data_fft_pos = data_fft[mask]

#         results[column] = (freqs_pos, np.abs(data_fft_pos))



    # if 'RMS' in results:  #Time,Amplitude,Rise-Time,Energy,Counts,Duration,RMS
    #     freqs_plot, interest_plot = results['RMS']
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(freqs_plot, interest_plot)
    #     plt.title('Normalized FFT')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('RMS')
    #     plt.grid(True)
    #     plt.show()

#     return results


# results = perform_fft(r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv')
