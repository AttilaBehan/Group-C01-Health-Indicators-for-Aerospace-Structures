import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import resample
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
import math

def Tr(X):
    """
    Calculate trendability score for a set of HIs (Health Indicators)
    """
    m, n = X.shape
    min_trendability = np.inf
    print(m)
    for j in range(m):
        for k in range(j + 1, m):  # Avoid duplicate pairs & self-comparison
            vector1 = X[j]
            vector2 = X[k]

            rho = pearsonr(vector1, vector2)[0]

            min_trendability = min(min_trendability, abs(rho))
        print(j)
        print(rho)
        print(min_trendability)

    return min_trendability

# List of your folders
folders = [
    r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\EMD_Features_interpolated_500_500_CSV",
    #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\FFT_Features_interpolated500_500_CSV",
    #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\STFT_Features_interpolated_500_500_CSV"
]

# Dictionary to store trendability results
trendability_results = {}

# Target resample length
target_length = 40

for folder in folders:
    file_paths = glob.glob(os.path.join(folder, "*.csv"))
    file_paths.sort()

    all_his = []

    for file_path in file_paths:
        data = np.genfromtxt(file_path, delimiter=',')
        cleaned_data = data[~np.isnan(data).all(axis=1)]
        transpose = cleaned_data.T
        data_end = transpose[1:, 1:]  # Drop first row and column

        # Resample this file's data to target_length (along columns)
        resampled_data = resample(data_end, target_length, axis=1)

        all_his.append(resampled_data)
        print('done')

    print(f"Resampled all HIs to {target_length} columns.")

    # Stack into X (m HIs x n Samples)
    X = np.vstack(all_his)

    # Normalize once (row-wise)
    scaler = Normalizer()
    X = scaler.fit_transform(X)
    print('normalized')
    # Compute trendability
    trendability_score = Tr(X)
    print('TR1')
    trendability_results[folder] = trendability_score
    print('TR2')
    print(f"Folder: {os.path.basename(folder)}")
    print(f"Number of HIs processed: {X.shape[0]}")
    print(f"Trendability score: {trendability_score:.4f}")
    print("---")

# Summary
print("\nTrendability Results Summary:")
for folder, score in trendability_results.items():
    print(f"{os.path.basename(folder)}: {score:.4f}")
