import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import resample
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
import math
from itertools import combinations

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

# def Pr(X):
#     """
#     Calculate prognosability score for a set of HIs

#     Parameters:
#         - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
#     Returns:
#         - prognosability (float): Prognosability score for given set of HIs
#     """
#     # Compute M as the number of HIs, and Nfeatures as the number of timesteps
#     M = len(X)
#     Nfeatures = X.shape[1]

#     # Initialize top and bottom of fraction in prognosability formula to zero
#     top = np.zeros((M, Nfeatures))
#     bottom = np.zeros((M, Nfeatures))

#     # Iterate over each HI
#     for j in range(M):

#         # Set row in top to the final HI value for current HI
#         top[j, :] = X[j, -1]

#         # Compute absolute difference between initial and final values for current HI
#         bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

#     # Compute prognosability score with formula
#     prognosability = np.exp(-np.std(top) / np.mean(bottom))

#     return prognosability

# def Mo(X):
#     """
#     Calculate monotonicity score for a set of HIs

#     Parameters:
#         - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
#     Returns:
#         - monotonicity (float): Monotonicity score for given set of HIs
#     """
#     # Initialize sum of individual monotonicities to 0
#     sum_monotonicities = 0

#     # Iterate over all HIs
#     for i in range(len(X)):

#         # Calculate the monotonicity of each HI with the Mo_single function, add to the sum
#         monotonicity_i = Mo_single(X[i, :])
#         sum_monotonicities += monotonicity_i

#     # Compute monotonicity score by normalizing over number of HIs
#     monotonicity = sum_monotonicities / np.shape(X)[0]

#     return monotonicity

# def fitness(X, Mo_a=1.0, Tr_b=1.0, Pr_c=1.0):
#     """
#     Calculate fitness score for a set of HIs

#     Parameters:
#         - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
#         - Mo_a (float): Weight of monotonicity score in the fitness function, with default value 1
#         - Tr_b (float): Weight of trendability score in the fitness function, with default value 1
#         - Pr_c (float): Weight of prognosability score in the fitness function, with default value 1
#     Returns:
#         - ftn (float): Fitness score for given set of HIs
#         - monotonicity (float): Monotonicity score for given set of HIs
#         - trendability (float): Trendability score for given set of HIs
#         - prognosability (float): Prognosability score for given set of HIs
#         - error (float): Error value for given set of HIs, defined as the sum of weights (default value 3) / fitness
#     """
#     # Compute the 3 prognostic criteria scores
#     monotonicity = Mo(X)
#     trendability = Tr(X)
#     prognosability = Pr(X)

#     # Compute fitness score as sum of scores multiplied by their respective weights
#     ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability

#     # Compute the error value, defined as the sum of the weights (default value 3) divided by the fitness score
#     error = (Mo_a + Tr_b + Pr_c) / ftn
#     #print("Error: ", error)

#     return ftn, monotonicity, trendability, prognosability, error

# List of your folders
folders = [
    r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\EMD_Features_interpolated_500_500_CSV",
    #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\FFT_Features_interpolated500_500_CSV",
    #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\STFT_Features_interpolated_500_500_CSV"
]

# Dictionary to store trendability results
trendability_results = {}

# Target resample length
target_length = 10

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
