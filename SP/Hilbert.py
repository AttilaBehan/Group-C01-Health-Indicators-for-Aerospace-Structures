import pandas as pd
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

def perform_ht(
        file_path,
        output_csv_path,
        *,
        cycle_block = 500,      # cycles per bucket
        cycle_duration = 0.5,   # [s] …only needed for plots’ x-axis
        #plot_dir = None,
    ):
    """
    Hilbert envelopes *per 500-cycle bucket*.

    Output CSV columns
    ------------------
    bucket              : 0-based integer cycle-bucket number
    sample_index        : 0, 1, 2… within that bucket
    <sig>_envelope      : amplitude envelopes (one per signal column)
    """
    # ------------------------------------------------------------------
    # 1. Load & basic checks
    # ------------------------------------------------------------------
    df = (pd.read_csv(file_path)
            .dropna(how="all"))               # drop all-NaN rows
    if "Time" not in df.columns:
        raise ValueError("Need a 'Time' column (in cycles).")

    # ------------------------------------------------------------------
    # 2. Assign bucket numbers
    # ------------------------------------------------------------------
    t_cycles = df["Time"].astype(float).to_numpy()
    df["bucket"] = (t_cycles // cycle_block).astype(int)

    signal_cols = [c for c in df.columns if c not in ("Time", "bucket")]
    if not signal_cols:
        raise ValueError("No signal columns found.")

    spectra = []                              # collect each bucket’s envelope DF

    # ------------------------------------------------------------------
    # 3. Process each populated bucket
    # ------------------------------------------------------------------
    for bid, grp in df.groupby("bucket", sort=True):
        if len(grp) < 2:                      # need ≥2 samples for Hilbert
            continue

        # Order as recorded (keep any irregularity; we *assume* it’s tiny)
        grp = grp.reset_index(drop=True)
        sample_idx = np.arange(len(grp))

        out = pd.DataFrame({
            "bucket": bid,
            "sample_index": sample_idx
        })

        # Time axis only needed for optional plots
        t_sec = grp["Time"].to_numpy(dtype=float) * cycle_duration

        for col in signal_cols:
            sig = grp[col].astype(float).to_numpy()
            # NOTE: we do *not* resample – we pretend uniform spacing
            env = np.abs(hilbert(sig))
            out[f"{col}_envelope"] = env

            # # Optional plot
            # if plot_dir is not None:
            #     os.makedirs(plot_dir, exist_ok=True)
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(t_sec, sig,  label="Signal",  alpha=0.5)
            #     plt.plot(t_sec, env,  label="Envelope", lw=2)
            #     plt.title(f"Bucket {bid} – {col}")
            #     plt.xlabel("Time [s]")
            #     plt.ylabel("Amplitude")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(plot_dir,
            #                              f"bucket{bid}_{col}_env.png"))
            #     plt.close()

        spectra.append(out)

    if not spectra:
        raise ValueError("No bucket contained enough data for an envelope.")

    # ------------------------------------------------------------------
    # 4. Concatenate & save
    # ------------------------------------------------------------------
    envelope_df = pd.concat(spectra, ignore_index=True)
    envelope_df.to_csv(output_csv_path, index=False)
    return envelope_df



envelope_df = perform_ht(
    file_path=r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2.csv',
    output_csv_path=r'C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\Low_Features_500_500_CSV\Sample2HT.csv',
)
