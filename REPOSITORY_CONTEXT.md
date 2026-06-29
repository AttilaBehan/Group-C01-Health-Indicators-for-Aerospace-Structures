# Repository Context — Health Indicators for Aerospace Structures

> Standing context for this repo. Read this first; it removes the need to re-explain
> the project on every prompt. (TU Delft AE2224-I "Test, Analysis and Simulation"
> theme project, Group C01.)

---

## 1. What this project is about

**Goal:** Structural Health Monitoring (SHM) of composite aerospace structures using
**Acoustic Emission (AE)** data. The structures (stiffened composite panels representing
an aircraft wing) are subjected to **run-to-failure fatigue loading**, and AE sensors
record the response over the panel's lifetime.

The core deliverable is a **Health Indicator (HI)**: a 1-D curve over the structure's
lifetime that should rise (or fall) monotonically as damage accumulates, so it can be
used for **diagnostics** (how damaged is it now) and **prognostics** (remaining useful
life). Because there is **no ground-truth HI label**, the HI is learned with
**unsupervised / semi-supervised deep learning** and judged by how well it satisfies
three **prognostic criteria** (monotonicity, trendability, prognosability).

There are **two modelling tracks**:
1. **VAE track (primary)** — a Variational Autoencoder constructs the HI from AE features.
2. **DCEC track (secondary)** — a Deep Convolutional Embedded Clustering model does
   **damage-mode classification** on the raw AE low-level features.

**The two key pipeline stages to keep in mind:**
1. **Signal processing + feature selection** — transform raw AE signals into time /
   frequency / time-frequency representations, extract statistical features, and score
   them so only the *most prognostic* features survive.
2. **VAE** — feed the selected features into a Variational Autoencoder that outputs the HI.

---

## 2. The data

- **Source:** ReMAP project public dataset (run-to-failure fatigue of composite panels).
- **12 samples** (panels), `Sample1` … `Sample12`. Work is done **leave-one-out across
  the 12 samples** (one test panel, one validation panel, the remaining ~10 train).
- **Raw input:** `Signals_LW500Int500Cycle.mat` (~200 MB, git-LFS). MATLAB struct
  `Signals` containing per-sample `Data`.
- **Raw AE low-level features** (7 columns), produced when the `.mat` is unpacked to CSV:
  `Time (cycle)`, **`Amplitude`, `Rise-Time`, `Energy`, `Counts`, `Duration`, `RMS`**.
  These six non-time columns are the AE "low-level features" everything is built on.
- **Time axis** is in fatigue **cycles**, bucketed into windows of `cycle_length = 500`
  (also `wavelength = 500`). The code is currently hard-wired to 500/500.

> **git-LFS:** `Signals_LW500Int500Cycle.mat`, everything under `DCEC/`, and
> `Output/Low_Features_500_500_CSV/**` are tracked via LFS (see `.gitattributes`).

---

## 3. End-to-end pipeline (the mental model)

```
Signals_LW500Int500Cycle.mat
        │  (1) load_mat  → unpack to per-sample CSV of 6 AE low-level features
        ▼
Output/Low_Features_500_500_CSV/SampleXX.csv          (raw low-level features)
        │  (2) signal processing: transform each feature into another domain
        │       FFT (freq) · STFT/CWT/SPWVD (time-freq) · EMD/Hilbert (time)
        ▼
Output/<METHOD>_500_500_CSV/                          (transformed signals)
        │  (3) statistical feature extraction (Mean, Std, Skew, Kurtosis, RMS, …)
        ▼
Output/<METHOD>_Features_500_500_CSV/
        │  (4) interpolate to fill missing cycles → uniform time grid
        ▼
Extracted_Features/<METHOD>_..._500_500_CSV/          (clean per-sample feature tables)
        │  (5) fitness scoring: rank every feature by prognostic criteria
        ▼
fitness_scores.csv  +  Feature_Score_Graphs/          (which features are useful)
        │  (6) hand-pick the top features as VAE inputs
        ▼
VAE/VAE_final/VAE_AE_DATA/SampleX.csv                 (selected features per sample)
        │  (7) VAE training (leave-one-out) → Health Indicator per sample
        ▼
Health Indicator curves + fitness scores (Mo / Tr / Pr)
```

The DCEC track is a **separate branch** off step (1): it consumes the raw 6-feature
ReMAP CSVs directly (`DCEC/ReMAP_Data/SampleX.csv`) and clusters AE events into damage
modes — it does **not** go through the VAE.

---

## 4. Repository layout

| Path | Role |
|------|------|
| `main.py` | **Interactive CLI menu** orchestrating steps 1–6 of the signal-processing/feature pipeline. Run this for everything except the VAE and DCEC. |
| `Data_Processing/` | `.mat → CSV` extraction, statistical feature extraction, cycle interpolation. |
| `Signal_Processing/` | The six transforms: `FFT`, `STFT`, `CWT`, `SPWVD` (+`Data_processing_SPWVD`), `EMD`, `Hilbert`, plus `plot_raw`. |
| `Fitness_Scoring.py` | Prognostic-criteria scoring (Mo/Tr/Pr) + plotting + `fitness_scores.csv` writing. **Used by `main.py`.** |
| `Output/` | Intermediate transformed signals & raw extracted features (per method). |
| `Extracted_Features/` | Cleaned, interpolated per-method feature tables (the "good" outputs). |
| `Feature_Score_Graphs/` | Saved bar charts of per-feature fitness scores. |
| `fitness_scores.csv`, `Fitness_scores_before_AS.txt` | Tabulated feature fitness results (AS = adaptive standardization). |
| `Morteza_Statistical_Features/` | Reference feature set (per sample) from the tutor/paper. |
| `VAE/VAE_final/` | **The VAE pipeline** (see §6). `VAE_AE_DATA/` holds the selected-feature CSVs fed to the VAE. |
| `DCEC/` | **The DCEC damage-classification pipeline** (see §7) incl. trained models, ReMAP data, and train/val/test outputs. |
| `SP/`, `SHM-GroupA7-main/`, `Data Processing/` (space) | Mostly empty / legacy / reference scaffolding. Not part of the active pipeline. |

---

## 5. Signal processing & feature extraction (stages 1–5)

Driven by `main.py`'s menu (choices `1`–`6`):

1. **Extract CSV from `.mat`** — `Low_Features_Extract_CSV.load_mat`. Produces the 7-column
   raw CSVs.
2. **Time-domain features** — `Feature_Extraction.extract_time_statistical_features`
   computes **19 time-domain features** per low-level feature per cycle-window
   (Mean, Std, Root amplitude, RMS, RSS, Peak, Skewness, Kurtosis, Crest/Clearance/Shape/Impulse
   factors, Max-Min diff, central moments 3–6, FM4, Median).
3. **Signal Processing** — pick one transform:
   - **FFT** → frequency domain (then 14 frequency features P1–P14).
   - **STFT / CWT / SPWVD** → time-frequency domain (then 4 features: Mean, Std, Skew, Kurtosis).
   - **EMD / Hilbert** → time domain (reuse the 19 time-domain features).
   - SPWVD has a pre-step (`Data_processing_SPWVD`: smoothing/downsampling/windowing) and
     its own transform step before feature extraction.
4. **Feature Extraction** — `Feature_Extraction.extract_*_statistical_features` turns each
   transformed signal into a per-cycle feature table.
5. **Fill missing cycles** — `interp_data.get_missing_cycles` linearly interpolates onto a
   uniform 500-cycle grid (needed because AE events are sparse/irregular in time).

The three feature-domain helpers live in `Data_Processing/Feature_Extraction.py`:
`time_domain_features` (19), `frequency_domain_features` (14), `time_frequency_domain_features` (4).

### Fitness scoring (stage 5/6) — `Fitness_Scoring.py`
A feature is "good" if, viewed as a candidate HI across the 12 samples, it scores high on
the three **prognostic criteria**:
- **Monotonicity (`Mo`)** — does it move consistently in one direction over time.
- **Trendability (`Tr`)** — minimum absolute Pearson correlation between samples' curves
  (consistent shape across panels).
- **Prognosability (`Pr`)** — how tightly the end-of-life values cluster relative to the
  total excursion: `exp(-std(final) / mean|initial−final|)`.
- **Fitness** = `Mo + Tr + Pr` (each weight defaults to 1). **Error** = `3 / fitness`
  (this is the quantity the VAE optimizer minimizes).

`main.py` choice `6` reshapes each method's extracted features, scores them, plots a
stacked bar chart, and appends results to `fitness_scores.csv`. **This is the feature-selection
step** — the highest-scoring features become VAE inputs.

---

## 6. The VAE track — `VAE/VAE_final/` (HI construction)

This is the heart of the project. It learns a Health Indicator from a **small set of
hand-selected, high-fitness AE features**.

**Inputs:** `VAE/VAE_final/VAE_AE_DATA/SampleX.csv` — one CSV per panel, columns are the
selected features. Example selected set currently in `Main.py`:
`['Counts_Variance', 'Energy_P10', 'Duration_Variance']` (the column headers in the
`VAE_AE_DATA` CSVs are a wider candidate set; `expected_cols` picks the subset actually used).

**Module map:**
| File | Responsibility |
|------|----------------|
| `Main.py` | Entry point. Leave-one-out split, runs training and/or hyperparameter optimization. Has several `if __name__=="__main__"` blocks toggled by `train_once` / `optimizing` / `code_og` flags. |
| `Model_architecture.py` | `VAE(tf.keras.Model)` — **LSTM-based** sequence VAE. Encoder: 2× LSTM → Dense(2·latent) giving mean+logvar; reparameterize; Decoder: Dense → RepeatVector → 2× LSTM → TimeDistributed Dense. `VAE_Seed.vae_seed = 42`. |
| `Train.py` | `VAE_train` (full LOO training loop w/ early stopping), `train_step` (gradient-clipped), and **`compute_health_indicator`** = `exp(-k · mean_features((x − x_recon)²))` per timestep (HI from reconstruction error). |
| `Loss_function.py` | `vae_loss` = `reloss_coeff·recon + klloss_coeff·KL + moloss_coeff·monotonicity_penalty`. The monotonicity penalty `relu(-Δhealth)` directly pushes the HI to be non-decreasing. |
| `Prog_crit.py` / `Prognostic_criteria.py` | Mo/Tr/Pr/fitness for HIs (two near-identical copies; `Main`/`Train` import from `Prog_crit`). Also `scale_exact` (resample HI to fixed length) and test-time `*_single` variants. |
| `Bayesian_optimization.py` | `scikit-optimize` (`gp_minimize`) hyperparameter search, minimizing fitness **error**. |
| `Optuna_optimization.py` | Alternative Optuna-based search. `Main.py` asks `1` (Bayesian) or `2` (Optuna). |
| `File_handling.py` | `VAE_merge_data_per_timestep` (load + resample each sample to `target_rows`, stack, `StandardScaler`), `resample_dataframe` (np.interp to `target_rows`). |
| `Plot_function.py` | HI curve plotting. |
| `Using_hyperparameters.py`, `old_stuff.py` | Apply stored best hyperparameters / legacy code. |
| `hyperparameters-opt-samples*.csv` | Saved best hyperparameter sets per test panel. |

**Hyperparameters being optimized** (`space` in `Main.py`): `hidden_1` (LSTM units 30–120),
`learning_rate`, `epochs`, `hidden_2` (latent dim), and the three loss coefficients
`reloss_coeff`, `klloss_coeff`, `moloss_coeff`.

**Key conventions / gotchas:**
- All data is **resampled to `target_rows`** (commonly 1200) so every panel's HI is the
  same length; time is expressed as **Lifetime (%)** 0–100 on plots.
- Leave-one-out: test = one panel, validation = `(i+5) % 12`, train = the rest.
- `num_features` must equal `len(expected_cols)`; mismatches are a common bug source.
- Feature-wise standardization is done on **train** data; applying the train scaler to
  test/val is currently **commented out** in several places (deliberate, noted in code).
- TensorFlow/Keras, seed 42 everywhere for reproducibility.

---

## 7. The DCEC track — `DCEC/` (damage-mode classification)

Separate from the HI work. **PyTorch**, runs on CUDA. Goal: cluster AE events into
**damage modes** (matrix cracking, delamination, fibre breakage, etc.) without labels.

- `DCEC_Model.py` — `CAE` (1-D convolutional autoencoder) + clustering layer (KMeans-init,
  KL-divergence cluster refinement = classic DCEC). Input = the 6 raw AE low-level features
  (`Amplitude, Rise-Time, Energy, Counts, Duration, RMS`), `StandardScaler`-normalized.
- `DCEC_Plotting.py`, `DCEC_Identification_Plotting.py` — visualization / cluster labelling.
- `Cluster_Label_Mappings.json` — maps cluster IDs → damage-mode names.
- `ReMAP_Data/SampleX.csv` — raw 7-column AE data (12 samples).
- `DCEC_Models/Train_<ids>/` — saved `autoencoder.pth` + `clustering.pth` per LOO fold
  (folder name lists the training-sample IDs).
- `DCEC_Training_Output/`, `DCEC_Validation_Output/`, `DCEC_Testing_Output/` — per-fold
  cluster assignments (`Train_SampleX_ValY_TestZ.csv` naming encodes the split).

---

## 8. How to run things

- **Signal processing / feature extraction / fitness scoring:** run `python main.py` from
  the repo root and follow the numbered menu. Paths are derived from `cycle_length=500`,
  `wavelength=500` (hard-coded). `main.py` recurses (calls `main()` again) so it loops the menu.
- **VAE:** run `python Main.py` **from inside `VAE/VAE_final/`** (paths like `"VAE_AE_DATA"`
  are relative to the CWD). Toggle `train_once` / `optimizing` flags at the top of the
  `__main__` blocks.
- **DCEC:** run scripts in `DCEC/` (needs a CUDA GPU; `device = "cuda"` is hard-coded).

**Dependencies (no requirements.txt yet):** `numpy`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`, `tensorflow` (VAE), `torch` (DCEC), `mat73` (read v7.3 `.mat`),
`scikit-optimize`/`skopt`, `optuna`, `tqdm`, `PyWavelets` (CWT, `pywt`), `emd` (EMD).

---

## 9. Glossary

- **AE** — Acoustic Emission (the SHM sensing modality used here).
- **HI** — Health Indicator: 1-D degradation curve over lifetime; the project's main output.
- **Low-level features** — the 6 raw AE descriptors: Amplitude, Rise-Time, Energy, Counts,
  Duration, RMS.
- **Prognostic criteria** — Monotonicity (Mo), Trendability (Tr), Prognosability (Pr);
  their sum is the **fitness** of an HI/feature.
- **VAE** — Variational Autoencoder (LSTM sequence VAE) that produces the HI.
- **DCEC** — Deep Convolutional Embedded Clustering for damage-mode classification.
- **LOO / LOOCV** — Leave-One-Out cross-validation across the 12 panels.
- **AS** — Adaptive Standardization (feature normalization used when generating HI labels).
- **ReMAP** — the EU project that produced the dataset.
- **cycle / cycle_length** — fatigue cycles; data is windowed in blocks of 500 cycles.

---

## 10. Known rough edges (so you don't trip on them)

- Everything is hard-wired to `500/500`; changing cycle length is not supported end-to-end.
- `Main.py`'s `code_og` block ("CHECK THIS OUT LATER") is **deprecated/non-functional**:
  it references an undefined `val_data_scaled` and unpacks `VAE_train`'s 6 return values
  into 2. It's guarded by `code_og = False`. Recommend deleting it rather than reviving it.
- `train_optimized_VAE` (Train.py) is **unused** experimental code; its surface bugs were
  fixed but the per-fold `hi_full` insertion logic is still untested. The live path is
  `optimizing=True → VAE_optimize_hyperparameters` (Bayesian) / `optimize_hyperparameters_optuna`.
- **Methodological items left as-is on purpose (not silently changed):**
  - `Feature_Extraction.frequency_domain_features` builds a *synthetic* frequency axis
    (`F = arange(1000, …, 1000)`) instead of the real `Frequency (Hz)` column, and is
    coupled to that via `/1000` and `*1000` scaling. Fixing it would change every FFT
    feature value, so it needs a deliberate decision + re-validation, not a blind edit.
  - `DCEC.load_data` fits a **separate `StandardScaler` per sample** (train *and* test),
    so features aren't normalized on a shared basis across a fold. Changing it would
    invalidate the already-trained models in `DCEC/DCEC_Models/`.

### Bug fixes applied (2026-06-30)
- **VAE HI sensitivity `k`** — training passed `target_rows` (~1200) positionally into the
  `k` slot of `compute_health_indicator`, so training used `exp(-1200·err)≈0` while eval
  used `k=1.0`. All call sites now pass `target_rows`/`num_features` by keyword (`k=1.0`).
  This re-activates the monotonicity loss, which was previously inert.
- **VAE batch_size ignored** — the epoch loop fed the whole tensor to one `train_step`
  (full-batch GD); `batch_size` is a tuned hyperparameter. It now iterates `train_dataset`
  (with `drop_remainder=False`).
- **`VAE_train` arg order** fixed in the two legacy `Main.py` calls (`hidden_2, target_rows,
  num_features`).
- **`Prognostic_criteria.py`** is now a thin re-export of `Prog_crit.py` (single source of
  truth; the optimiser objective and the reported fitness can no longer desync).
- Minor: dead duplicate branch in `vae_loss` removed; invalid `np.tile(..., (1,-1))` dead
  branch removed; `main.py` menu de-recursed into a loop; missing `CWT` column added to the
  fitness-scores CSV header; DCEC now falls back to CPU and only runs `loocv_run()` under
  `if __name__ == "__main__"`.
