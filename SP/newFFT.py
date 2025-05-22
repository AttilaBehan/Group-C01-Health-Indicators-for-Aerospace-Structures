"""
Window-by-window FFT for ReMAP AE data – wide CSV output, no plotting
---------------------------------------------------------------------

Run in VS Code with the ▶ “Run Python File” button.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────── EDIT THIS ONE LINE IF THE FOLDER MOVES ───────────────
DATA_DIR = Path(r"C:\Users\macpo\Desktop\TU Delft\Y2\Q3\project\dmc\ReMAP_Data")
# ----------------------------------------------------------------------

# AE variables expected in every file
VARS_AE     = ['Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']
WINDOW_SIZE = 500        # cycles per block
NFFT        = 512        # 2**9  → 257 single-sided bins
HANN        = np.hanning(WINDOW_SIZE)
F_BINS      = range(NFFT // 2 + 1)      # 0 … 256

# Acceptable header names for the cycle/time column
TIME_ALIASES = {
    "time", "cycle", "cycles", "cycle_number",
    "cyclenumber", "labelled_cycle", "labeled_cycle"
}

# ── helper functions ───────────────────────────────────────────────────
def find_time_column(df: pd.DataFrame) -> str:
    """Return the name of the column that holds cycle indices."""
    for col in df.columns:
        if col.strip().lower() in TIME_ALIASES:
            return col
    raise ValueError(
        f"No time/cycle column found. Available columns: {list(df.columns)}\n"
        f"Add the correct name to TIME_ALIASES."
    )

def uniform_series(df: pd.DataFrame, time_col: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build 1-sample-per-cycle vectors for every AE variable."""
    df[time_col] = df[time_col].astype(int)
    cycles = np.arange(df[time_col].min(), df[time_col].max() + 1)
    series = {}
    for var in VARS_AE:
        if var not in df.columns:
            print(f"⚠︎ '{var}' missing in {df.columns}. Skipping.")
            continue
        s = df.groupby(time_col)[var].mean().reindex(cycles)
        series[var] = s.interpolate("linear", limit_direction="both").to_numpy()
    return series, cycles

def block_fft(x: np.ndarray) -> np.ndarray:
    """Return single-sided FFT magnitude for one window."""
    x = (x - x.mean()) * HANN
    return np.abs(np.fft.rfft(x, n=NFFT) / WINDOW_SIZE)

def process_file(csv_path: Path) -> None:
    """Generate *_winFFT_wide.csv for one input file."""
    df = pd.read_csv(csv_path)
    time_col = find_time_column(df)
    series, cycles = uniform_series(df, time_col)

    n_windows = len(cycles) // WINDOW_SIZE
    rows = []

    for w in range(n_windows):
        i0, i1 = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
        centre_cycle = i0 + WINDOW_SIZE // 2
        row = {'time': centre_cycle}

        for var in VARS_AE:
            if var not in series:          # skipped earlier
                continue
            spec = block_fft(series[var][i0:i1])
            row.update({f'{var}_f{b}': spec[b] for b in F_BINS})
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = csv_path.with_stem(csv_path.stem + '_winFFT_wide')
    out_df.to_csv(out_path, index=False)
    print(f"✓ {csv_path.name} → {out_path.name}   ({len(out_df)} windows)")

# ── main – batch over all Sample*.csv ──────────────────────────────────
if __name__ == "__main__":
    for csv in sorted(DATA_DIR.glob('Sample*.csv')):
        process_file(csv)
    print("Done.")