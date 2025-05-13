import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def perform_fft(
        input_csv,              # path to raw file
        output_csv,             # where the FFT spectra go
        *,
        cycle_duration = 0.5,   # seconds per cycle
        cycle_block    = 500    # cycles per block (= 250 s here)
    ):
    """
    FFT for irregularly-sampled, cycle-counted data.

    1.  Divide the Time (cycles) axis into `cycle_block`-sized chunks.
    2.  Drop empty chunks (no rows in that 500-cycle span).
    3.  For each remaining chunk, take the median Δt (seconds).
    4.  Average those Δt’s → dt   → fs = 1/dt.
    5.  Do a single-sided FFT on every signal column using that fs.
    6.  Save a tidy CSV:  Frequency (Hz), Amplitude_spectrum, Rise-Time_spectrum, …

    The implicit assumption is that within each populated chunk the jitter is
    small enough that one representative dt is “good enough” for a classical FFT.
    If that isn’t true, see **Alternative approaches** below.
    """
    # ---------------- Load & pre-check -------------------------------------------
    df = pd.read_csv(input_csv).dropna(how="all")

    # Convert Time → seconds
    t_cycles = df["Time"].to_numpy(dtype=float)
    t_sec    = t_cycles * cycle_duration

    # ---------------- Block the timeline -----------------------------------------
    block_id = (t_cycles // cycle_block).astype(int)
    df["block"] = block_id
    df["t_sec"] = t_sec                       # keep for later use

    dt_blocks = []
    for bid, grp in df.groupby("block"):
        if len(grp) < 2:                      # 0-or-1 row → no Δt info
            continue
        dt = np.diff(np.sort(grp["t_sec"]))
        dt = dt[dt > 0]                       # ignore duplicates
        if dt.size:
            dt_blocks.append(np.median(dt))

    dt  = np.mean(dt_blocks)                  # average median Δt’s
    fs  = 1.0 / dt

    # ---------------- Frequency axis ---------------------------------------------
    N      = len(df)                          # all rows kept (including duplicates)
    freqs  = fftfreq(N, d=1/fs)
    pos    = freqs >= 0
    freqs  = freqs[pos]

    # ---------------- Build output spectrum --------------------------------------
    out = pd.DataFrame({"Frequency (Hz)": freqs})

    signal_cols = [c for c in df.columns if c not in ("Time", "block", "t_sec")]
    for col in signal_cols:
        x  = df[col].to_numpy()
        Xf = fft(x)
        Xf = np.abs(Xf) / N
        Xf[1:N//2] *= 2
        out[col] = Xf[pos]

    out.to_csv(output_csv, index=False)
    return out

# Example call
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
