# LSTM Fall Prediction

A multi-step LSTM that predicts whether a first fall will occur in the next `HORIZON` clinical visits, given `LOOKBACK` visits of patient history. The pipeline is implemented as a set of Jupyter notebooks plus one analysis utility. Two cohorts are supported: **ParkWest** (primary training and testing) and **PPMI** (external validation).

---

## Folder contents

| File                              | Purpose |
|-----------------------------------|---------|
| `ParkVest_LSTM.ipynb`             | Main training notebook (ParkWest). Runs end-to-end: data load → feature discovery → splits → imputation/scaling → training → evaluation → figures. |
| `PPMI_LSTM.ipynb`                 | External-validation notebook (PPMI). Applies the ParkWest-trained pipeline to the PPMI cohort, using the cross-cohort feature mapping. |
| `Literature_feat_reduce.ipynb`    | Builds the reduced literature feature sets (N = 30 → 20 → 15 → 10) used by the literature-baseline LSTM runs. |
| `plotresult.ipynb`                | Aggregates the per-run result tables and produces the comparison figures and tables used in the thesis Results chapter. |
| `analyze_lstm_features.py`        | Command-line utility that summarises which features each run actually used (presence, missingness, derived deltas) and produces a short report. |
| `ReadME.md`                       | This file. |

---

## Run order

For a full sweep of experiments, the intended order is:

### 1. *(optional)* `Literature_feat_reduce.ipynb`

Re-build the reduced literature sets if you have changed the base literature feature list. The reduced sets are stored under `Results/LSTM/literature_feature_sets/`.

### 2. `ParkVest_LSTM.ipynb`

Train and evaluate one configuration on ParkWest. Change `FEATURE_SET`, `FEATURE_COUNT`, and the horizon/lookback configuration to sweep the experimental grid.

### 3. `PPMI_LSTM.ipynb`

Apply the same configuration to PPMI for external validation. Re-uses the training-fold standardisation transferred from ParkWest where possible.

### 4. `plotresult.ipynb`

Collect the per-run output folders under `Results/LSTM/` and produce the comparison plots / tables across feature sets, feature counts, horizons, and lookback windows.

### 5. *(optional)* `analyze_lstm_features.py`

```bash
python Programs/LSTM/analyze_lstm_features.py
```

Produces a short report on the features actually used in each run folder.

---

## Inputs

### ParkWest

```text
Data/ParkWest/Raw/ParkVest_ClinicalData_with_Metadata.xlsx
```

| Convention              | Value |
|-------------------------|-------|
| Patient ID column       | `BL_CASE` |
| Visit-date columns      | `V{v}_DateVisit` (default `V4..V21`) |
| Visit feature columns   | `V{v}_<FEATURE>` (e.g. `V7_PIGD`, `V12_HAD4`) |

### PPMI

```text
Data/PPMI/Raw/<LONI exports>.csv
```

See the [top-level ReadME](../../ReadME.md) for the expected file list.

### Fall-event definition (per visit)

| Cohort   | Definition |
|----------|------------|
| ParkWest | `UPO13 >= 1` or `UPO14 >= 3` |
| PPMI     | `NP2WALK >= 1` or `NP2FREZ >= 3` |

> Event-defining columns are explicitly excluded from the feature set to avoid label leakage.

---

## Configuration knobs

Set in the first cell of each training notebook.

### Data and target

| Knob | Default | Description |
|------|---------|-------------|
| `DATA_PATH` | — | Path to the ParkWest `.xlsx` / PPMI CSV root. |
| `VISIT_START` / `VISIT_END` | `4 / 21` | Visit-number range to include. |
| `HORIZON` | `3` | Number of future visits the model predicts the first-fall probability for. |
| `LOOKBACK` | `3` | Number of past visits fed to the LSTM as input sequence (experiments also use `5`). |
| `MIN_HISTORY_VISITS` | `3` | Minimum number of past visits required for a patient to be included. |
| `STOP_AFTER_FIRST_EVENT` | `True` | Drop visits after the first observed event when constructing the training sequences. |

### Features

| Knob | Description |
|------|-------------|
| `FEATURE_SET` | One of `literature_baseline`, `set_a`, `set_b`, `univariate_ranked`. |
| `FEATURE_COUNT` | `30`, `20`, `15`, or `10`. |
| `BASE_FEATURES` | Explicit list of base features to use. |
| `DELTA_FEATURE_BASES` | Base features to also include as visit-to-visit deltas (`delta = current − previous`). |

### Model

| Knob | Default |
|------|---------|
| `HIDDEN_SIZE` | `64` |
| `NUM_LAYERS` | `1` |
| `DROPOUT` | `0.1` |
| `BATCH_SIZE` | `16` |
| `LR` | `1e-3` |
| `EPOCHS` | `50` |
| `PATIENCE` | `5` (early-stopping patience) |

### Output

| Knob | Description |
|------|-------------|
| `RUN_NAME` | Sub-folder name for this run's outputs (defaults to a timestamp + configuration tag). |

---

## Outputs

Each training run writes to:

```text
Results/LSTM/<FEATURE_SET>/results/<RUN_NAME>/
```

### Typical files

| File                                 | Content |
|--------------------------------------|---------|
| `visit_dates_repaired.csv`           | Cleaned visit-date table |
| `patient_event_summary.csv`          | Per-patient event time and censoring |
| `visit_level_long_raw.csv`           | Long-format pre-imputation data |
| `selected_features.csv`              | Features used (base + delta) |
| `train_patients.csv` / `val` / `test`| Patient ID splits |
| `*_processed_long.csv`               | Imputed + scaled long format |
| `sample_preview.csv`                 | First batch sample for sanity check |
| `training_history.csv`               | Per-epoch train / val loss + AUC |
| `training_history.png`               | Loss / AUC curves |
| `test_predictions.csv`               | Per-sample test set predictions |
| `test_predictions_thresholded.csv`   | Binarised at the Youden-optimal threshold |
| `metrics_summary_t+1.csv` *(and `+2`, `+3`)* | AUC, sensitivity, specificity, accuracy (with bootstrap CIs if enabled) |
| `best_lstm_model.pt`                 | Trained model weights |

PPMI external-validation runs write the same set of files under `Results/LSTM/<FEATURE_SET>_PPMI/`.

> All writes use mode `'w'` (overwrite). Re-running a configuration with the same `RUN_NAME` replaces its output, never appends.

---

## Adapting to a new dataset

The two main extension points are:

### 1. Data loading

The notebooks expect a wide-format Excel / CSV with one row per patient and visit-numbered columns. To plug in a new cohort:

- Update `DATA_PATH` and `VISIT_START` / `VISIT_END`.
- Update the column conventions in the *"Load data"* cell:
  - patient ID column,
  - visit-date column pattern,
  - visit-feature column pattern.
- Update the fall-event definition cell to use the new cohort's event-defining columns.

For larger cohorts that need a custom long-format build, mirror `Common/ppmi_data_loader.py` and call it from a new notebook section.

### 2. Features

Either point `BASE_FEATURES` at a column list valid for the new cohort, or extend the feature-set chooser cell to load a different pre-defined set (e.g. a new literature baseline).

> If the cohort uses MDS-UPDRS rather than the original UPDRS, the Cox cross-cohort mapping file ([`Programs/Cox/Common/cross_cohort_mapping.py`](../Cox/Common/cross_cohort_mapping.py)) already contains the column translations and can be reused to define the LSTM feature columns.

---

## Reproducibility

- A fixed seed is set for `numpy`, `torch`, and `random` in the first cells.
- The patient ID splits are deterministic given the seed; the resulting CSV files in the run folder make them auditable.
- Imputation and scaling are fit on the training fold only and applied to validation / test, so there is no leakage.
- Exact bit-for-bit reproducibility across machines is **not** guaranteed due to cuDNN non-determinism and floating-point ordering in batched training; AUCs typically agree to within ~0.005 across runs.

---

## Dependencies

| Package                  | Purpose                                  |
|--------------------------|------------------------------------------|
| `torch`                  | LSTM model and training loop             |
| `pandas`, `numpy`, `scipy` | data wrangling                         |
| `scikit-learn`           | `StandardScaler`, AUC, etc.              |
| `matplotlib`, `seaborn`  | figures                                  |
| `jupyter`, `ipykernel`   | run the notebooks                        |
| `openpyxl`               | read ParkWest `.xlsx`                    |

See the [top-level ReadME](../../ReadME.md) for environment setup.

---

## Notes

- **Leakage-safe.** Event-defining columns are excluded from the feature space, future visits are never visible to the model at a given prediction point, and imputation / scaling are fit on the training fold only.

- **Time semantics.** `HORIZON = 3` with 6-month visit spacing corresponds to a prediction window of approximately **18 months**. `LOOKBACK = 3` likewise corresponds to approximately **18 months** of clinical history per input sequence; `LOOKBACK = 5` gives ~30 months.

- **Bootstrap CIs.** The test-set evaluation uses bootstrap resampling (default 1000 replicates) for 95 % confidence intervals on AUC, sensitivity, and specificity. Bootstrap can be disabled for fast iteration via `BOOTSTRAP_N = 0`.

- **Threshold choice.** Sensitivity / specificity in the result tables are reported at the Youden-optimal threshold computed on the **validation** set, not on the test set, to avoid threshold-tuning bias.