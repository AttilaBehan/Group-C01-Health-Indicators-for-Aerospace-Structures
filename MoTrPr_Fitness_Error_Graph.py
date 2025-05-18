import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os
import pandas as pd

def Tr(X):
    m, n = X.shape
    nonconstant_rows = [i for i in range(m) if np.std(X[i]) > 1e-12]
    if len(nonconstant_rows) < 2:
        print("Not enough non-constant rows for correlation.")
        return 0.0
    X_filtered = X[nonconstant_rows]

    correlations = []
    for j in range(len(X_filtered)):
        for k in range(j + 1, len(X_filtered)):
            rho, _ = pearsonr(X_filtered[j], X_filtered[k])
            if not np.isnan(rho):
                correlations.append(abs(rho))
    if len(correlations) == 0:
        return 0.0
    return np.mean(correlations)

def Pr(X):
    M, N = X.shape
    valid_rows = [i for i in range(M) if np.std(X[i]) > 1e-12]
    if len(valid_rows) == 0:
        print("No valid rows for prognosability calculation.")
        return 0.0
    X_filtered = X[valid_rows]

    top = X_filtered[:, -1]
    bottom = np.abs(X_filtered[:, 0] - X_filtered[:, -1])
    mean_bottom = np.mean(bottom)
    if mean_bottom < 1e-12:
        mean_bottom = 1e-8

    ratio = np.std(top) / mean_bottom
    ratio = min(ratio, 10)  # clip max ratio

    prognosability = np.exp(-ratio)
    return prognosability

def Mo_single(x):
    diffs = np.diff(x)
    if len(diffs) == 0:
        return 0.0
    signs = np.sign(diffs)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    monotonicity = 1 - sign_changes / (len(signs) - 1) if len(signs) > 1 else 1.0
    if np.all(diffs == 0):
        monotonicity = 1.0
    return monotonicity

def Mo(X):
    sum_monotonicities = 0
    for i in range(len(X)):
        monotonicity_i = Mo_single(X[i, :])
        sum_monotonicities += monotonicity_i
    monotonicity = sum_monotonicities / np.shape(X)[0]
    return monotonicity

def fitness(X, Mo_a=1.0, Tr_b=1.0, Pr_c=1.0):
    monotonicity_score = Mo(X)
    trendability_score = Tr(X)
    prognosability_score = Pr(X)

    ftn = Mo_a * monotonicity_score + Tr_b * trendability_score + Pr_c * prognosability_score
    error = (Mo_a + Tr_b + Pr_c) / ftn if ftn != 0 else float('inf')

    return ftn, monotonicity_score, trendability_score, prognosability_score, error

def run_on_folders_aggregate(folders):
    monotonicity_dict = {}
    prognosability_dict = {}
    trendability_dict = {}

    for directory in folders:
        print(f"Processing folder: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".csv"):
                    filepath = os.path.join(root, file)
                    print(f"Processing file: {filepath}")
                    df = pd.read_csv(filepath).dropna()
                    X = df.to_numpy()

                    ftn, monotonicity_score, trendability_score, prognosability_score, error = fitness(X)

                    # Extract feature name prefix (adjust this as needed)
                    feature_name = os.path.splitext(file)[0].split('_')[0]

                    monotonicity_dict.setdefault(feature_name, []).append(monotonicity_score)
                    prognosability_dict.setdefault(feature_name, []).append(prognosability_score)
                    trendability_dict.setdefault(feature_name, []).append(trendability_score)

    # Aggregate scores by averaging over all files belonging to the same feature
    features = sorted(monotonicity_dict.keys())
    Mo_scores = np.array([np.mean(monotonicity_dict[f]) for f in features])
    Pr_scores = np.array([np.mean(prognosability_dict[f]) for f in features])
    Tr_scores = np.array([np.mean(trendability_dict[f]) for f in features])

    return features, Mo_scores, Pr_scores, Tr_scores

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    plt.figure(figsize=(12, 6))
    plt.bar(features, Mo, label="Monotonicity")
    plt.bar(features, Pr, bottom=Mo, label="Prognosability")
    plt.bar(features, Tr, bottom=Mo+Pr, label="Trendability")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.title('Prognostic Criteria Scores per Feature')
    plt.tight_layout()

    if dir and name:
        plt.savefig(os.path.join(dir, f"{name}_criteria_chart.png"))
        print(f"Saved chart as {os.path.join(dir, f'{name}_criteria_chart.png')}")
    else:
        plt.show()
    plt.close()

# Your folders here
folders = [
    r"C:\Users\apaun\OneDrive\Desktop\TAS\EMD_Features_interpolated_500_500_CSV",
    r"C:\Users\apaun\OneDrive\Desktop\TAS\FFT_Features_interpolated500_500_CSV",
    r"C:\Users\apaun\OneDrive\Desktop\TAS\STFT_Features_interpolated_500_500_CSV"
]

features, Mo_scores, Pr_scores, Tr_scores = run_on_folders_aggregate(folders)

criteria_chart(features, Mo_scores, Pr_scores, Tr_scores)