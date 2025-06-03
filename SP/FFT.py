import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os 

def perform_fft(cycle_length, input_dir, output_dir): 
    """
    Perform FFT on all columns except 'Time' from a CSV file.
    Returns a dictionary with column names as keys and tuples of (frequencies, normalized FFT values).
    Also plots the normalized FFT of the 'Amplitude' column.
    """
    #os.makedirs(output_dir, exist_ok=True)
    cycle_duration=0.5  # seconds per cycle
    #cycle_length -> cycles per block (= 250 s here)  
    os.makedirs(output_dir, exist_ok=True) 
    for root, dirs, samples in os.walk(input_dir):
        for sample in samples:
            file_path=os.path.join(root, sample) 
            print("Processing: ", sample)
            # Parameters
            df = pd.read_csv(file_path).dropna(how="all")
            if "Time (cycle)" not in df.columns: 
                raise ValueError("Missing 'Time (cycle)' column.")

            # Time [cycle] → seconds, assign bucket number
            t_cycles  = df["Time (cycle)"].astype(float).to_numpy()
            t_sec     = t_cycles * cycle_duration
            df["t_sec"] = t_sec
            df["bucket"] = (t_cycles // cycle_length).astype(int)

            # Which columns are actual signals?
            signal_cols = [c for c in df.columns
                        if c not in ("Time (cycle)", "t_sec", "bucket")] 

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
                
            # -------- Save & return --------------------------------------------------------
            filename = os.path.join(output_dir, sample)
            out = pd.concat(spectra, ignore_index=True) 
            out.to_csv(filename, index=False)
            
    return out

#results = perform_fft(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Low_Features_500_500_CSV") 
#print(results) 
