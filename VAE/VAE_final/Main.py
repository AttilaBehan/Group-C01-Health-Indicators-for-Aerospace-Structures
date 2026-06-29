"""
VAE Health-Indicator pipeline -- full leave-one-out (LOOCV) run.

WHAT THIS DOES (no interactive prompts):
  * Leave-one-out over the 12 panels. For each fold:
      - train the LSTM-VAE on the 10 training panels,
      - generate the Health Indicator (HI) for the held-out TEST panel,
      - score that test HI ONLY against the training HIs (the test panel is never
        used to choose hyperparameters -> an honest, held-out estimate).
  * Saves HIs, per-panel fitness and plots to ./vae_results/.

TWO MODES (set MODE below):
  * "fixed"    -> use one documented hyperparameter set for every fold.
                  Fast and fully reproducible. Recommended for a first full run.
  * "optimize" -> run a leakage-free Bayesian search per fold first, then train
                  the final model per fold with the best params. Much slower.

HOW TO RUN:
    # from anywhere (paths are resolved relative to this file):
    python VAE/VAE_final/Main.py
    # or from inside the folder:
    cd VAE/VAE_final && python Main.py

Outputs (in VAE/VAE_final/vae_results/):
    test_HIs.csv         12 x target_rows  -- the held-out HI of each panel
    test_fitness.csv     per-panel honest held-out fitness (Mo / Tr / Pr)
    test_HIs_all.png     all 12 held-out HIs on one axis
    test_HIs_grid.png    per-panel HI (blue) over its training HIs (grey)
    hyperparameters-opt-samples.csv   (only in MODE="optimize")
"""

import os
import sys
import re
import ast
import glob
import random

# Resolve imports + data relative to THIS file, so the script runs from any CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
from skopt.space import Real, Integer

from Model_architecture import VAE_Seed
from File_handling import VAE_merge_data_per_timestep, resample_dataframe
from Train import VAE_train
from Prog_crit import test_fitness
from Bayesian_optimization import VAE_optimize_hyperparameters

# ----------------------------------------------------------------------------
# CONFIG  -- edit these
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join(HERE, "VAE_AE_DATA")      # one CSV per panel
RESULTS_DIR = os.path.join(HERE, "vae_results")   # all outputs go here

MODE = "fixed"        # "fixed" (fast, reproducible) or "optimize" (slow search)
QUICK_RUN = True     # True -> tiny/fast smoke test (NOT for reporting)

# Features fed to the VAE (must exist as columns in every VAE_AE_DATA CSV).
EXPECTED_COLS = ["Counts_Variance", "Energy_P10", "Duration_Variance"]
BATCH_SIZE = 40

# Hyperparameters used in MODE="fixed" (and as a documented default).
DEFAULT_HP = dict(
    hidden_1=72,
    learning_rate=0.005,
    epochs=300,
    hidden_2=32,
    reloss_coeff=0.05,
    klloss_coeff=1.6,
    moloss_coeff=2.5,
)

# Bayesian search settings (MODE="optimize").
# NOTE: scikit-optimize uses 10 random initial points by default, so N_CALLS must
# be comfortably > 10 for any GP-guided steps to actually happen.
N_CALLS_PER_SAMPLE = 25
SEARCH_SPACE = [
    Integer(30, 120, name="hidden_1"),
    Real(1e-4, 1e-2, "log-uniform", name="learning_rate"),
    Integer(50, 600, name="epochs"),
    Integer(6, 64, name="hidden_2"),
    Real(0.02, 0.5, name="reloss_coeff"),
    Real(0.8, 2.0, name="klloss_coeff"),
    Real(0.8, 3.5, name="moloss_coeff"),
]

if QUICK_RUN:
    TARGET_ROWS = 300
    DEFAULT_HP["epochs"] = 60
    N_CALLS_PER_SAMPLE = 12
else:
    TARGET_ROWS = 1200

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def set_global_seeds():
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)


def sample_number(path):
    """Numeric sort key so SampleX.csv files sort 1,2,...,12 (not 1,10,11,...)."""
    digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
    return int(digits) if digits else 0


def get_sample_paths():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=sample_number)
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. Expected SampleX.csv feature files."
        )
    return paths


def prepare_fold(all_paths, i, expected_cols, target_rows, num_features):
    """Build the (train, val, test) tensors for leave-one-out fold i.

    The StandardScaler is fit on the TRAINING panels only and then applied to the
    validation and test panels (no leakage through normalisation).
    """
    n = len(all_paths)
    val_idx = (i + 5) % n
    test_path = all_paths[i]
    val_path = all_paths[val_idx]
    train_paths = [p for j, p in enumerate(all_paths) if j != i and j != val_idx]

    train_data, scaler = VAE_merge_data_per_timestep(train_paths, expected_cols, target_rows)

    df_test = resample_dataframe(pd.read_csv(test_path)[expected_cols], target_rows)
    df_val = resample_dataframe(pd.read_csv(val_path)[expected_cols], target_rows)
    test_data = scaler.transform(df_test.values)
    val_data = scaler.transform(df_val.values)

    # Reshape flat (n*target_rows, features) -> (n, target_rows, features)
    train_data = train_data.reshape(-1, target_rows, num_features)
    test_data = test_data.reshape(1, target_rows, num_features)
    val_data = val_data.reshape(1, target_rows, num_features)
    return train_data, val_data, test_data, val_idx


def load_hp_from_csv(path, n_expected):
    """Parse the per-fold hyperparameters saved by the optimizer."""
    df = pd.read_csv(path)
    hps = []
    for _, row in df.iterrows():
        clean = re.sub(r"np\.(?:int64|float64)\((-?\d+\.?\d*(?:e-?\d+)?)\)", r"\1", str(row["params"]))
        p = ast.literal_eval(clean)
        hps.append(dict(
            hidden_1=int(p[0]), learning_rate=float(p[1]), epochs=int(p[2]),
            hidden_2=int(p[3]), reloss_coeff=float(p[4]), klloss_coeff=float(p[5]),
            moloss_coeff=float(p[6]),
        ))
    if len(hps) != n_expected:
        raise ValueError(f"Expected {n_expected} hyperparameter rows, got {len(hps)} in {path}")
    return hps


def plot_results(test_his, train_his_per_fold, target_rows, results_dir):
    n = test_his.shape[0]
    x = np.linspace(0, 100, target_rows)

    # (1) All held-out HIs on one axis.
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(x, test_his[i], label=f"Sample{i + 1}")
    plt.xlabel("Lifetime [%]")
    plt.ylabel("Health Indicator")
    plt.title("Held-out test health indicators (all panels)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_HIs_all.png"), dpi=200)
    plt.close()

    # (2) Per-panel grid: held-out HI (blue) over its training HIs (grey).
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i in range(n):
        ax = axes[i]
        for tr_hi in train_his_per_fold[i]:
            ax.plot(x, tr_hi, color="gray", alpha=0.4, linewidth=0.6)
        ax.plot(x, test_his[i], color="tab:blue", linewidth=1.5)
        ax.set_title(f"Test Sample{i + 1}")
        ax.grid(True, alpha=0.3)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.supxlabel("Lifetime [%]")
    fig.supylabel("Health Indicator")
    fig.suptitle("Per-panel held-out HI (blue) over training HIs (grey)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "test_HIs_grid.png"), dpi=200)
    plt.close(fig)


def run_final_loocv(all_paths, per_fold_hp, expected_cols, target_rows,
                    num_features, batch_size, results_dir):
    """Train the final model for every fold and collect honest held-out results."""
    os.makedirs(results_dir, exist_ok=True)
    n = len(all_paths)
    test_his = np.full((n, target_rows), np.nan)
    train_his_per_fold = []
    fitness_rows = []

    for i in range(n):
        hp = per_fold_hp[i]
        print(f"\n===== Fold {i + 1}/{n}: TEST = Sample{i + 1} =====")
        set_global_seeds()
        train_data, val_data, test_data, val_idx = prepare_fold(
            all_paths, i, expected_cols, target_rows, num_features
        )
        hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(
            train_data, val_data, test_data,
            hp["hidden_1"], batch_size, hp["learning_rate"], hp["epochs"],
            hp["reloss_coeff"], hp["klloss_coeff"], hp["moloss_coeff"],
            hp["hidden_2"], target_rows, num_features,
        )

        hi_train = np.asarray(hi_train).reshape(-1, target_rows)
        hi_test = np.asarray(hi_test).reshape(-1)
        test_his[i] = hi_test
        train_his_per_fold.append(hi_train)

        # Honest held-out fitness: test HI vs the TRAIN HIs only.
        ftn, mo, tr, pr = test_fitness(hi_test, hi_train)
        fitness_rows.append([f"Sample{i + 1}", ftn, mo, tr, pr])
        print(f"Held-out fitness Sample{i + 1}: fitness={ftn:.3f} "
              f"(Mo={mo:.3f}, Tr={tr:.3f}, Pr={pr:.3f})")

    # Save artifacts.
    pd.DataFrame(test_his).to_csv(os.path.join(results_dir, "test_HIs.csv"), index=False)
    df_fit = pd.DataFrame(
        fitness_rows,
        columns=["panel", "fitness", "monotonicity", "trendability", "prognosability"],
    )
    df_fit.to_csv(os.path.join(results_dir, "test_fitness.csv"), index=False)

    print("\n================ SUMMARY (held-out) ================")
    print(df_fit.to_string(index=False))
    print(f"\nMean held-out fitness: {df_fit['fitness'].mean():.3f} "
          f"(Mo={df_fit['monotonicity'].mean():.3f}, "
          f"Tr={df_fit['trendability'].mean():.3f}, "
          f"Pr={df_fit['prognosability'].mean():.3f})")

    plot_results(test_his, train_his_per_fold, target_rows, results_dir)
    print(f"\nSaved results + plots to: {results_dir}")
    return df_fit


def main():
    set_global_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_paths = get_sample_paths()
    num_features = len(EXPECTED_COLS)

    print(f"Mode: {MODE} | quick_run: {QUICK_RUN} | target_rows: {TARGET_ROWS}")
    print(f"Panels: {len(all_paths)} | features: {EXPECTED_COLS}")

    if MODE == "optimize":
        print("\n--- Stage 1/2: leakage-free Bayesian hyperparameter search (slow) ---")
        VAE_optimize_hyperparameters(
            RESULTS_DIR, EXPECTED_COLS, all_paths, N_CALLS_PER_SAMPLE,
            TARGET_ROWS, SEARCH_SPACE, BATCH_SIZE, num_features,
        )
        hp_csv = os.path.join(RESULTS_DIR, "hyperparameters-opt-samples.csv")
        per_fold_hp = load_hp_from_csv(hp_csv, len(all_paths))
    elif MODE == "fixed":
        per_fold_hp = [dict(DEFAULT_HP) for _ in all_paths]
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (use 'fixed' or 'optimize')")

    print("\n--- Final LOOCV training + honest held-out evaluation ---")
    run_final_loocv(
        all_paths, per_fold_hp, EXPECTED_COLS, TARGET_ROWS,
        num_features, BATCH_SIZE, RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
