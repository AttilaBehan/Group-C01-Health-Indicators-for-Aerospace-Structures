import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os
import pandas as pd
import time

def Tr(X):
    m, n = X.shape
    nonconstant_rows = [i for i in range(m) if np.std(X[i]) > 1e-12]
    if len(nonconstant_rows) < 2:
        return 0.0
    X_filtered = X[nonconstant_rows]
    correlations = []
    for j in range(len(X_filtered)):
        for k in range(j + 1, len(X_filtered)):
            rho, _ = pearsonr(X_filtered[j], X_filtered[k])
            if not np.isnan(rho):
                correlations.append(abs(rho))
    return np.mean(correlations) if correlations else 0.0

def Pr(X):
    M, N = X.shape
    valid_rows = [i for i in range(M) if np.std(X[i]) > 1e-12]
    if not valid_rows:
        return 0.0
    X_filtered = X[valid_rows]
    top = X_filtered[:, -1]
    bottom = np.abs(X_filtered[:, 0] - X_filtered[:, -1])
    mean_bottom = np.mean(bottom) if np.mean(bottom) >= 1e-12 else 1e-8
    ratio = np.std(top) / mean_bottom
    return np.exp(-min(ratio, 10))

def Mo_single(x):
    diffs = np.diff(x)
    if len(diffs) == 0: return 0.0
    signs = np.sign(diffs)
    changes = np.sum(signs[1:] != signs[:-1])
    return 1 - changes / (len(signs) - 1) if len(signs) > 1 else 1.0

def Mo(X):
    return np.mean([Mo_single(row) for row in X]) if len(X) > 0 else 0.0

def fitness(X, Mo_a=1.0, Tr_b=1.0, Pr_c=1.0):
    m = Mo(X)
    t = Tr(X)
    p = Pr(X)
    f = Mo_a * m + Tr_b * t + Pr_c * p
    error = (Mo_a + Tr_b + Pr_c) / f if f != 0 else float('inf')
    return f, m, t, p, error

def run_on_folders_all_files(folders):
    file_ids = []
    Mo_scores = []
    Pr_scores = []
    Tr_scores = []

    all_files = []
    for directory in folders:
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if file.lower().endswith(".csv"):
                    filepath = os.path.join(root, file)
                    all_files.append((filepath, os.path.splitext(file)[0]))

    total = len(all_files)
    print(f"\n🟢 Starting processing of {total} files...\n")

    for idx, (filepath, file_id) in enumerate(all_files, start=1):
        print(f"🔄 [{idx}/{total}] Processing: {file_id}")
        try:
            df = pd.read_csv(filepath).dropna()
            X = df.to_numpy()
            _, m, t, p, _ = fitness(X)
            print(f"   ➤ Monotonicity:    {m:.4f}")
            print(f"   ➤ Prognosability:  {p:.4f}")
            print(f"   ➤ Trendability:    {t:.4f}")
        except Exception as e:
            print(f"   ❌ Failed to process {file_id}: {e}")
            m, p, t = 0.0, 0.0, 0.0

        file_ids.append(file_id)
        Mo_scores.append(m)
        Pr_scores.append(p)
        Tr_scores.append(t)

    print("\n✅ All files processed.\n")
    return file_ids, np.array(Mo_scores), np.array(Pr_scores), np.array(Tr_scores)

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    plt.figure(figsize=(18, 6))
    plt.bar(features, Mo, label="Monotonicity")
    plt.bar(features, Pr, bottom=Mo, label="Prognosability")
    plt.bar(features, Tr, bottom=Mo + Pr, label="Trendability")
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.legend()
    plt.xlabel('Feature (File Name)')
    plt.ylabel('Score')
    plt.title('Prognostic Criteria Scores per Feature File')
    plt.tight_layout()

    if dir and name:
        plt.savefig(os.path.join(dir, f"{name}_criteria_chart.png"))
    else:
        plt.show()
    plt.close()

# === Main Execution ===
folders = [
    r"C:\Users\apaun\OneDrive\Desktop\TAS\EMD_Features_interpolated_500_500_CSV",
    r"C:\Users\apaun\OneDrive\Desktop\TAS\FFT_Features_interpolated500_500_CSV",
    r"C:\Users\apaun\OneDrive\Desktop\TAS\STFT_Features_interpolated_500_500_CSV"
]

start_time = time.time()

features, Mo_scores, Pr_scores, Tr_scores = run_on_folders_all_files(folders)
criteria_chart(features, Mo_scores, Pr_scores, Tr_scores)

end_time = time.time()
print(f"⏱️ Total runtime: {end_time - start_time:.2f} seconds")
#blue = Mo, orange = Pr, green = Tr
