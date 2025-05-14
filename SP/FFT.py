import pandas as pd
import numpy as np
from  scipy.fft import fft, fftfreq

def perform_fft(
        input_csv,
        output_csv,
        *,
        cycle_duration = 0.5,   # [s] length of one machine cycle
        cycle_block    = 500    # [cycles] cycles per bucket
    ):
    """
    FFT for irregularly-sampled AE data, producing *one* spectrum
    per cycle-block (“bucket”) and appending them all into one CSV.

    Output columns
    --------------
    bucket            : 0-based integer cycle-block number
    Frequency (Hz)    : positive-frequency axis for *that* block
    <signal columns>  : amplitude spectra, single-sided, Vrms-consistent
    """
    # ---------- Load data -------------------------------------------------------
    df = (pd.read_csv(input_csv)
            .dropna(how="all")        # drop all-NaN rows
         )
    if "Time" not in df.columns:
        raise ValueError("No 'Time' column in the input file.")

    # Time [cycle] → seconds, assign bucket number
    t_cycles  = df["Time"].astype(float).to_numpy()
    t_sec     = t_cycles * cycle_duration
    df["t_sec"] = t_sec
    df["bucket"] = (t_cycles // cycle_block).astype(int)

    # Which columns are actual signals?
    signal_cols = [c for c in df.columns
                   if c not in ("Time", "t_sec", "bucket")]

    spectra = []             # collect one DataFrame per bucket

    # ---------- Process each bucket -------------------------------------------
    for bid, grp in df.groupby("bucket", sort=True):
        if len(grp) < 2:              # need at least 2 points for Δt
            continue

        # Representative sample interval for *this* bucket
        dt = np.diff(np.sort(grp["t_sec"]))
        dt = dt[dt > 0]               # ignore duplicates
        if not dt.size:
            continue                  # skip all-duplicate buckets
        fs = 1.0 / np.median(dt)

        # Frequency axis (single-sided)
        N = len(grp)
        freqs = fftfreq(N, d=1/fs)
        pos   = freqs >= 0
        freqs = freqs[pos]

        # Build one tidy spectrum for this bucket
        spec = pd.DataFrame({
            "bucket": bid,
            "Frequency (Hz)": freqs
        })

        for col in signal_cols:
            x  = grp[col].to_numpy(dtype=float)
            Xf = fft(x)
            Xf = np.abs(Xf) / N
            if N % 2 == 0:            # even-length → Nyquist term exists
                Xf[1:N//2] *= 2
            else:                     # odd-length → no true Nyquist
                Xf[1:(N+1)//2] *= 2
            spec[col] = Xf[pos]

        spectra.append(spec)

    if not spectra:
        raise ValueError("No bucket contained enough data for an FFT.")

    # ---------- Concatenate all bucket spectra & save --------------------------
    out = pd.concat(spectra, ignore_index=True)
    out.to_csv(output_csv, index=False)
    return out


# ---------------- Example call -----------------------------------------
output_csv_path = r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1FFT.csv"
results_df = perform_fft(
    r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample1.csv",
    output_csv_path
)

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
