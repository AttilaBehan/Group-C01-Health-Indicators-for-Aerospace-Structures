import pandas as pd
import numpy as np
''' The SPWVD code requires some additional processing which is done here'''

# Load AE data 
df = pd.read_csv("SP\\Sample1Interp.csv")

# Step 1: Group by cycle (Time column) and compute per-cycle features
grouped = df.groupby('Time')

# Define aggregations for useful features
agg_df = grouped.agg({
    'Amplitude': ['mean', 'std', 'max', 'min'],
    'Energy': ['sum', 'mean'],
    'Counts': ['sum'],
    'Duration': ['mean'],
    'RMS': ['mean'],
    'Rise-Time': ['mean'],
})

# Step 2: Flatten the MultiIndex column names
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df = agg_df.reset_index()

# Step 3: Add AE event count per cycle
agg_df['Num_Events'] = grouped.size().values

# Step 4: Handle NaNs (e.g., std when only one AE event per cycle)
agg_df.fillna(0, inplace=True)  # Or use a different strategy like interpolation

# Step 5: Apply rolling smoothing (e.g., 5-cycle window)
window_size = 5
smoothed_df = agg_df.copy()
rolling_cols = [col for col in agg_df.columns if col not in ['Time']]  # exclude 'Time'

for col in rolling_cols:
    smoothed_df[f'{col}_smoothed'] = agg_df[col].rolling(window=window_size, min_periods=1).mean()

# Step 6: Preview
print(smoothed_df.head())

# Optional: Save to CSV
smoothed_df.to_csv("Sample1_cycle_level_features_smoothed.csv", index=False)