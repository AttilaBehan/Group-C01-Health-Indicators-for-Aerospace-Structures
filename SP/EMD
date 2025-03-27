import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emd


def my_get_next_imf(x, zoom=None, sd_thresh=0.1, max_iters=10):
    proto_imf = x.copy()
    continue_sift = True
    niters = 0
    final_sd = 0

    if zoom is None:
        zoom = (0, x.shape[0])

    while continue_sift and niters < max_iters:
        niters += 1
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
        avg_env = (upper_env + lower_env) / 2
        stop, final_sd = emd.sift.stop_imf_sd(proto_imf - avg_env, proto_imf, sd=sd_thresh)
        proto_imf = proto_imf - avg_env
        if stop:
            continue_sift = False

    return proto_imf, niters, final_sd

def runEMD(df):
    time = df['Time'].to_numpy()
    signal_columns = df.columns[1:]
    imfs_dict = {}
    iterations_dict = {}
    sd_dict = {}
    
    for col in signal_columns:
        x = df[col].to_numpy()
        imf, niters, final_sd = my_get_next_imf(x)
        imfs_dict[col] = imf
        iterations_dict[col] = niters
        sd_dict[col] = final_sd
    
    return pd.DataFrame(imfs_dict), iterations_dict, sd_dict

def plot_results(df, imfs_df, iterations_dict, sd_dict):
    time = df['Time'].to_numpy()
    signal_columns = df.columns[1:]
    
    for col in signal_columns:
        fig, axes = plt.subplots(nrows=2, figsize=(10, 6))
        
        # Plot original signal
        axes[0].plot(time, df[col], label=f'Raw {col}', color='blue')
        axes[0].set_title(f'Original Signal: {col}')
        axes[0].legend()
        
        # Plot IMF
        axes[1].plot(time, imfs_df[col], label=f'IMF of {col}', color='red')
        axes[1].set_title(f'IMF Decomposition: {col}\nIterations: {iterations_dict[col]}, SD: {sd_dict[col]:.4f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# Load dataset
df = pd.read_csv(r"SP\LowLevelFeaturesSample1.csv")

# Run EMD
imfs_df, iterations_dict, sd_dict = runEMD(df)

# Save results
imfs_df.to_csv(r"SP\OutputEMD" , index=False)

# Plot results
plot_results(df, imfs_df, iterations_dict, sd_dict)
