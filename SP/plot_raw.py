import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for sample_number in range(1, 2):
    file_path = f"C:/Users/macpo/Desktop/TU Delft/Y2/Q3/project/Low_Features_500_500_CSV/Sample{sample_number}.csv"
    
    # Load data with optimized dtype
    df = pd.read_csv(file_path, dtype=np.float32)
    
    # Clean data more efficiently
    cols = ['Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']
    df = df.dropna(how='all', subset=cols)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(how='any', subset=cols)
    
    # Downsample data - adjust every_n value based on your data density
    every_n = 5  # Plot every 10th point (adjust this based on your data size)
    df_subsampled = df.iloc[::every_n]
    
    # Plot setup
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Faster plotting parameters
    plot_kwargs = {
        'marker': '.',      # Fastest marker type
        'linestyle': 'none',
        'markersize': 1,   # Smaller = faster
        'alpha': 0.3,       # Helps visualize density
        'color': 'green'    # Single color is faster
    }
    
    for idx, col in enumerate(cols):
        axes[idx].plot(
            df_subsampled['Time'],
            df_subsampled[col],
            **plot_kwargs
        )
        axes[idx].set_title(f'Sample {sample_number}: {col} vs Time')
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Crucial for memory management with many plots