# PARK-AI / Alan_Sondre

Master's thesis project: prognostic feature importance and temporal fall prediction in Parkinson's disease, using the **ParkWest** cohort (UiS / SUS) and the **PPMI** cohort for external validation.

| | |
|---|---|
| **Authors**  | Sondre Lyngstad and Alan Lici |
| **Program**  | MSc Computer Science / Data Science, University of Stavanger |
| **Semester** | Spring 2026 |

---

## What the project does

Two parallel analyses on longitudinal PD clinical data:

1. **Cox feature importance.** Three independent feature-selection pipelines (literature-based, domain-stratified, and top-8 univariate) are run on ParkWest, then externally validated on PPMI. Outputs are hazard ratios, C-index, proportional-hazards diagnostics, Kaplan–Meier risk-quartile curves, and cross-cohort comparison tables.

2. **LSTM fall prediction.** A multi-step LSTM forecasts whether the next *H* clinical visits will contain a first-fall event, using *L* visits of patient history. Four feature sets (literature baseline and three Cox-derived) and several feature counts and horizon/lookback configurations are compared.

The fall event is defined consistently across both cohorts as:

| Cohort   | Event definition |
|----------|---------------------------------|
| ParkWest | `UPO13 >= 1` or `UPO14 >= 3`    |
| PPMI     | `NP2WALK >= 1` or `NP2FREZ >= 3` |

(both correspond to MDS-UPDRS Part II walking / freezing thresholds.)

---

## Folder structure

```text
Alan_Sondre/
├── Programs/                 # source code
│   ├── Cox/                  # Cox proportional-hazards pipelines
│   │   ├── Common/             # shared modules (data loaders, feature maps,
│   │   │                       #   screening, models, diagnostics)
│   │   ├── Litterature/        # ParkWest literature pipeline
│   │   ├── Univariate/         # ParkWest top-8 univariate pipeline
│   │   ├── Domain/             # ParkWest domain-stratified pipeline (multi-stage)
│   │   ├── Litterature_PPMI/   # PPMI external validation (literature)
│   │   ├── Univariate_PPMI/    # PPMI external validation (top-8)
│   │   ├── Domain_PPMI/        # PPMI domain-stratified pipeline (end-to-end)
│   │   └── ReadME.md           # how to run the Cox pipelines
│   │
│   └── LSTM/                 # LSTM temporal prediction
│       ├── ParkVest_LSTM.ipynb       # main training notebook (ParkWest)
│       ├── PPMI_LSTM.ipynb           # external validation notebook (PPMI)
│       ├── Literature_feat_reduce.ipynb  # reduced literature feature sets
│       ├── plotresult.ipynb          # aggregated plotting of LSTM results
│       ├── analyze_lstm_features.py  # feature-list analysis utility
│       └── ReadME.md                 # how to run the LSTM notebooks
│
├── Data/                     # raw and preprocessed datasets (gitignored)
│   ├── ParkWest/
│   │   ├── Raw/                # ParkVest_ClinicalData_with_Metadata.xlsx
│   │   └── Preprocessed/Cox/   # long-format CSVs from the Cox pipelines
│   └── PPMI/
│       ├── Raw/                # LONI CSV exports
│       └── Preprocessed/Cox/   # long-format CSVs from the Cox pipelines
│
├── Results/                  # model output (mostly gitignored CSV, tracked PNG/TXT)
│   ├── Cox/
│   │   ├── Litterature_PW/     # ParkWest literature pipeline outputs
│   │   ├── Domain_PW/          # ParkWest domain pipeline outputs
│   │   ├── Top8_PW/            # ParkWest top-8 pipeline outputs
│   │   ├── Litterature_PPMI/   # PPMI literature replication outputs
│   │   ├── Univariate_PPMI/    # PPMI top-8 replication outputs
│   │   └── Domain_PPMI/        # PPMI domain pipeline outputs
│   └── LSTM/                 # LSTM training runs and validation outputs
│
└── ReadME.md                 # this file
```

---

## Data access

The clinical datasets are confidential and are **not distributed with this repository**. They must be obtained directly from the data owners.

### ParkWest

Stavanger University Hospital, ParkWest study. The expected file is:

```text
Data/ParkWest/Raw/ParkVest_ClinicalData_with_Metadata.xlsx
```

### PPMI

Free academic access on request at [ppmi-info.org](https://www.ppmi-info.org). Place the LONI CSV exports under `Data/PPMI/Raw/`. The pipelines read e.g.:

- `Participant_Status_13Jan2026.csv`
- `MDS-UPDRS_Part_I_13Jan2026.csv`
- `MDS-UPDRS_Part_II_13Jan2026.csv`
- `MDS-UPDRS_Part_III_13Jan2026.csv`
- `MDS-UPDRS_Part_IV__Motor_Complications_13Jan2026.csv`
- `Modified_Schwab___England_Activities_of_Daily_Living_13Jan2026.csv`
- `SCOPA-AUT_13Jan2026.csv`
- `Epworth_Sleepiness_Scale_13Jan2026.csv`
- `Geriatric_Depression_Scale__Short_Version__13Jan2026.csv`
- `Determination_of_Freezing_and_Falls_13Jan2026.csv`

(plus a few more — see [`Programs/Cox/Common/ppmi_data_loader.py`](Programs/Cox/Common/ppmi_data_loader.py) for the exact list and column expectations).

> No patient-level data are committed to the repository.

---

## Environment and dependencies

Python **3.11+** recommended. Core packages:

| Package                    | Purpose                                         |
|----------------------------|-------------------------------------------------|
| `pandas`, `numpy`, `scipy` | data wrangling                                  |
| `scikit-learn`             | scaling, AUC, splits                            |
| `lifelines`                | Cox time-varying models, Kaplan–Meier           |
| `statsmodels`              | VIF computation                                 |
| `matplotlib`, `seaborn`    | figures                                         |
| `torch`                    | LSTM                                            |
| `openpyxl`                 | read ParkWest `.xlsx`                           |
| `jupyter`, `ipykernel`     | run the LSTM notebooks                          |

### Suggested setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If `requirements.txt` is missing, install the packages above directly.

---

## How to run

See the two sub-READMEs for details:

- [`Programs/Cox/ReadME.md`](Programs/Cox/ReadME.md) — how to run the six Cox configurations
- [`Programs/LSTM/ReadME.md`](Programs/LSTM/ReadME.md) — how to run the LSTM notebooks

All scripts and notebooks use paths relative to `Alan_Sondre/` via `pathlib.Path`; you can run them from any working directory.

---

## Reproducibility notes

- All output writes overwrite (no append modes). Re-running a pipeline produces the same files, not duplicated rows.
- Cox univariate and multivariable fits use `lifelines` defaults with a fixed penalizer (`0.01`) for numerical stability; results are deterministic given the input data.
- LSTM training is seeded, but exact reproducibility across machines is not guaranteed due to non-deterministic cuDNN kernels and floating-point ordering in batched training.

---

## Contact

For questions about the code or thesis, contact the authors at the University of Stavanger.