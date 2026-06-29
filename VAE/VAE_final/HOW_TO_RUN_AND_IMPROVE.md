# VAE Health-Indicator model — how to run & how to improve

## How to run a full run

The VAE step assumes the per-panel feature CSVs already exist in
`VAE/VAE_final/VAE_AE_DATA/` (they do — `Sample1.csv … Sample12.csv`). If you need
to regenerate features from the raw `.mat`, run the signal-processing pipeline
(`python main.py` at the repo root) first.

```bash
# from the repo root (paths inside Main.py are resolved relative to the file):
python VAE/VAE_final/Main.py
# or:
cd VAE/VAE_final && python Main.py
```

All knobs live in the **CONFIG** block at the top of `Main.py`:

| setting | meaning | default |
|---|---|---|
| `MODE` | `"fixed"` = one documented hyperparameter set for every fold (fast, reproducible). `"optimize"` = leakage-free Bayesian search per fold, then final training (slow). | `"fixed"` |
| `QUICK_RUN` | `True` = small/fast smoke test (`target_rows=300`, few epochs). Not for reporting. | `False` |
| `EXPECTED_COLS` | the features fed to the VAE | `Counts_Variance, Energy_P10, Duration_Variance` |
| `DEFAULT_HP` | hyperparameters used in `fixed` mode | see file |
| `N_CALLS_PER_SAMPLE` | search budget per fold (`optimize` mode). Must be **> 10** for any GP-guided steps. | `25` |
| `TARGET_ROWS` | resampled lifetime length | `1200` |

**Outputs** land in `VAE/VAE_final/vae_results/`:
- `test_HIs.csv` — 12 × `target_rows`, the held-out HI of each panel
- `test_fitness.csv` — per-panel **honest, held-out** fitness (Mo / Tr / Pr)
- `test_HIs_all.png` — all 12 held-out HIs on one axis
- `test_HIs_grid.png` — each panel's held-out HI (blue) over its training HIs (grey)
- `hyperparameters-opt-samples.csv` — only in `optimize` mode

**First run advice:** start with `MODE="fixed"`, `QUICK_RUN=True` to confirm it runs
end-to-end in ~1 min, then set `QUICK_RUN=False` for the real run. Move to
`MODE="optimize"` only once the fixed run looks sensible (it is much slower:
12 folds × `N_CALLS_PER_SAMPLE` trainings).

## What changed (integrity fixes baked into this run)

1. **No test leakage.** Hyperparameters / model selection now use **train+val only**;
   the held-out test panel is scored *after* selection via `test_fitness`. Previously
   the objective minimised `fitness(vstack(train, test, val))`, i.e. it tuned on the
   test panel.
2. **Honest reporting.** One reproducible results folder is written by the code. The
   old hand-named `hyperparameters-opt-samples_*.csv` snapshots are **not** used.
3. **Real training.** The monotonicity loss and `batch_size` now actually affect
   training (two earlier bugs made the HI sensitivity `k≈1200` during training and
   ignored `batch_size`).
4. **Windows-safe / portable.** No emoji in `print()`, CUDA falls back to CPU, paths
   resolved relative to the script.

---

## How to improve the model

Ordered roughly by expected payoff per unit effort.

### 1. Fix the remaining leakage in *feature* selection
The 3 input features were chosen by scoring features across **all 12 panels**. That
is selection-on-the-test-panel one level up. Within each LOOCV fold, re-select
features using only that fold's training panels (or at least report that selection
used the full set). Until then, even the de-leaked HI fitness is mildly optimistic.

### 2. Make the search real (or drop the pretence)
With `n_calls = 10` and scikit-optimize's 10 random init points, the "Bayesian"
search did **zero** guided steps and evaluated the *same* 10 candidates for every
panel. Either raise `N_CALLS_PER_SAMPLE` to 40–60 so the GP actually guides the
search, or call it a random search honestly. Tune **one shared** hyperparameter set
on a held-out validation split rather than a different set per panel — per-panel
hyperparameters with a tiny budget mostly fit noise.

### 3. Reconsider the HI definition
`HI = exp(-mean reconstruction error)` is pushed monotonic only by a soft penalty
(`relu(-Δhealth)`), so the curve's direction is imposed, not measured. Options:
- weight the monotonicity term per-fold and report sensitivity to it;
- add a **smoothness**/total-variation penalty so HIs are less jagged;
- try the literature target directly (DeepSAD / DTC-VAE from the assignment refs)
  which constrain the latent trend instead of the reconstruction error.

### 4. Strengthen validation
- LOOCV gives 12 point estimates with high variance. Run **multiple seeds** and report
  mean ± std of held-out fitness, not a single number.
- The validation panel is `(i+5) % 12` — a fixed offset. Rotate it or use a small
  inner validation loop so val choice doesn't bias early stopping.

### 5. Model / data robustness
- **Standardisation:** features are standardised on train and applied to test — good.
  But each panel is independently resampled to 1200 points, which **erases absolute
  lifetime**. If remaining-useful-life matters, keep a time/҂cycle channel.
- **Sequence length:** a single 1200-step LSTM is heavy and hard to train. Consider a
  1-D conv encoder (like the DCEC CAE already in the repo) or chunked sequences.
- **Capacity vs. data:** with only ~10 training sequences per fold, large `hidden_1`
  overfits. Keep the network small; rely on the latent bottleneck (`hidden_2`).
- **Early stopping** currently monitors a val loss that mixes recon+KL+monotonicity;
  monitor the **held-out HI fitness** instead, which is what you actually care about.

### 6. Reproducibility hygiene
- Delete the three hand-named hyperparameter CSVs; commit only code-generated results.
- Pin package versions (`requirements.txt`) — TF/Keras changes alter results.
- Log the exact config + seed next to each results folder.

### 7. Quantify, don't eyeball
Report held-out Mo/Tr/Pr per panel **with** a baseline (e.g. the best single raw
feature's fitness, and a random-HI control). A VAE HI is only worth it if it beats the
best hand-crafted feature — show that gap.
