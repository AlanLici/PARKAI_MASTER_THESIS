"""
run_pipeline.py
---------------
Literature-Based Feature Selection — Cox Pipeline

This pipeline uses a predefined set of features selected from the fall-risk
literature in Parkinson's disease.  It applies the same data quality filters
as the other pipelines, runs univariate screening on the literature features,
selects the top non-redundant features, and fits a multivariable Cox model.

Features
--------
30 features from published fall-risk studies, covering:
    Motor (PIGD, UPO items, SEON, HYON)
    Cognition (STROOPW, STROOPC, MMSE, IQCODE, MCI)
    Mood (madrs10, madrs_tot, npi8hi)
    Sleep (pdss02, pdss_tot, epwo6)
    Autonomic (URINDYS, CONSTIPAT, PressStand_Dia)
    Fatigue (fss2, fss_sum, fss5)
    Quality of Life (sf36pf, sf36pcs, PHYFUN)
    Demographics (age_at_visit, disease_duration_years — computed)

Stages
------
1. Univariate screening   — screen each literature feature individually
2. Feature selection       — p < 0.05, correlation dedup, top N
3. VIF screening           — remove features with VIF >= 6.0
4. Multivariable Cox       — fit final model, evaluate, forest plot
5. Diagnostics             — Schoenfeld residuals (PH assumption test)

Data quality filters (shared with other pipelines)
    - V9 guard, 58% threshold, median fill, LOCF

Output folder
    Results/Cox/Litterature_PW/

Usage
    python Programs/Cox/Litterature/run_pipeline.py
"""

import sys
import time
import importlib.util
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports — reuse shared functions from models/
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Litterature/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

_spec_uni = importlib.util.spec_from_file_location(
    "univariate_screening",
    COMMON_DIR / "univariate_cox_model_screening.py",
)
uni = importlib.util.module_from_spec(_spec_uni)
_spec_uni.loader.exec_module(uni)

_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

from cox_diagnostics import run_schoenfeld_test


# Shared functions
load_and_filter           = uni.load_and_filter
compute_event_time        = uni.compute_event_time
run_univariate_screening  = uni.run_univariate_screening
select_top_features       = uni.select_top_features
build_long_multi_feature  = uni.build_long_multi_feature
build_long_single_feature = uni.build_long_single_feature
get_feature_schedule      = uni.get_feature_schedule
apply_v9_guard            = uni.apply_v9_guard
iterative_vif_selection   = multi.iterative_vif_selection
save_vif_results          = multi.save_vif_results
standardize_features      = multi.standardize_features
fit_cox_model             = multi.fit_cox_model
build_results_table       = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest               = multi.plot_forest

# ---------------------------------------------------------------------------
# Literature-based feature list
# ---------------------------------------------------------------------------
LITERATURE_FEATURES = [
    # Motor
    "PIGD",
    "UPO_part2_tot",
    "UPO_part3_tot",
    "UPO27",
    "UPO15",
    "UPO_part4_tot",
    "SEON",
    "HYON",
    # Cognition
    "STROOPW",
    "MMSE_tot",
    "IQCODE_mean",
    "STROOPC",
    "MCI_15SD_MDScriteria",
    # Mood / Psychiatric
    "madrs10",
    "madrs_tot",
    "npi8hi",
    # Sleep
    "pdss02",
    "pdss_tot",
    "epwo6",
    # Autonomic
    "URINDYS",
    "CONSTIPAT",
    "PressStand_Dia",
    # Fatigue
    "fss2",
    "fss_sum",
    "fss5",
    # Quality of Life
    "sf36pf",
    "sf36pcs",
    "PHYFUN",
    # Demographics (computed time-varying)
    "age_at_visit",
    "disease_duration_years",
]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TOP_N_FEATURES   = 8
CORR_THRESHOLD   = 0.70
VIF_THRESHOLD    = 6.0

# Results folder (Cox tables, plots, reports, diagnostics)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Litterature_PW"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessed long-format data (the dataset the Cox model is trained on)
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "ParkWest" / "Preprocessed" / "Cox"
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Compute time-varying demographic features
# ---------------------------------------------------------------------------

def add_computed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add visit-level columns for age_at_visit and disease_duration_years.

    These are not in the raw data but can be derived from:
        age_at_visit          = BL age + (visit_date - BL_date) in years
        disease_duration_years = Time_PDfirst_BL + (visit_date - BL_date) in years

    Creates columns like V4_age_at_visit, V5_age_at_visit, etc.
    """
    df = df.copy()

    # Resolve baseline age — try common column names
    bl_age = None
    for col_name in ["BL_Alder", "BL_Age", "BL_alder"]:
        if col_name in df.columns:
            bl_age = pd.to_numeric(df[col_name], errors="coerce")
            print(f"[Computed] Using {col_name} for baseline age.")
            break

    # Resolve disease duration at baseline
    bl_duration = None
    if "Time_PDfirst_BL" in df.columns:
        bl_duration = pd.to_numeric(df["Time_PDfirst_BL"], errors="coerce")
        print("[Computed] Using Time_PDfirst_BL for baseline disease duration.")

    bl_dates = uni._parse_date(df["BL_DateVisit"])

    for visit in uni.MODEL_VISIT_PREFIXES:
        date_col = f"{visit}_DateVisit"
        if date_col not in df.columns:
            continue

        visit_dates = uni._parse_date(df[date_col])
        years_since_bl = (visit_dates - bl_dates).dt.days / 365.25

        if bl_age is not None:
            df[f"{visit}_age_at_visit"] = bl_age + years_since_bl

        if bl_duration is not None:
            df[f"{visit}_disease_duration_years"] = bl_duration + years_since_bl

    n_age = sum(1 for v in uni.MODEL_VISIT_PREFIXES
                if f"{v}_age_at_visit" in df.columns)
    n_dur = sum(1 for v in uni.MODEL_VISIT_PREFIXES
                if f"{v}_disease_duration_years" in df.columns)
    print(f"[Computed] Created age_at_visit for {n_age} visits, "
          f"disease_duration_years for {n_dur} visits.")

    return df


# ---------------------------------------------------------------------------
# Check which literature features are available
# ---------------------------------------------------------------------------

def validate_features(df: pd.DataFrame,
                      features: list[str]) -> tuple[list[str], list[str]]:
    """Check which features exist as visit-level columns in the data.

    Returns (available, missing) lists.
    """
    available = []
    missing = []

    for feat in features:
        schedule = get_feature_schedule(df, feat)
        if schedule:
            available.append(feat)
        else:
            missing.append(feat)

    return available, missing


# ---------------------------------------------------------------------------
# Selection report
# ---------------------------------------------------------------------------

def write_selection_report(screening: pd.DataFrame,
                           initial_features: list[str],
                           missing_features: list[str],
                           v9_excluded: list[str],
                           selected: list[str],
                           output_path: Path) -> None:
    sep = "=" * 72
    lines = [
        sep,
        "LITERATURE-BASED FEATURES -- SELECTION REPORT",
        sep, "",
        f"Initial literature features     : {len(initial_features)}",
        f"Missing from data               : {len(missing_features)}",
        f"Excluded by V9 guard            : {len(v9_excluded)}",
        f"Available for screening         : {len(screening)}",
        "",
    ]

    if missing_features:
        lines.append("  Missing features:")
        for f in missing_features:
            lines.append(f"    - {f}")
        lines.append("")

    if v9_excluded:
        lines.append("  V9 guard excluded:")
        for f in v9_excluded:
            lines.append(f"    - {f}")
        lines.append("")

    # Univariate results summary
    converged = screening[screening["converged"] == True]
    n_sig = len(converged[converged["p"] < 0.05])
    lines.append(f"  Converged                     : {len(converged)}")
    lines.append(f"  Significant (p < 0.05)        : {n_sig}")
    lines.append(f"  Selected for multivariable    : {len(selected)}")
    lines.append("")

    lines += [sep, "UNIVARIATE RESULTS (all literature features)", sep, ""]
    for _, row in screening.iterrows():
        if not row.get("converged", False):
            lines.append(f"  {row['Feature']:<30s}  did not converge")
            continue
        p = row["p"]
        hr = row["HR"]
        sig = "*" if p < 0.05 else " "
        p_str = f"p={p:.4e}" if pd.notna(p) else "p=n/a"
        hr_str = f"HR={hr:.3f}" if pd.notna(hr) else "HR=n/a"
        lines.append(f"  {row['Feature']:<30s}  {hr_str}  {p_str}  {sig}")
    lines.append("")

    lines += [sep, "SELECTED FOR MULTIVARIABLE MODEL", sep, ""]
    for i, feat in enumerate(selected, 1):
        row = screening[screening["Feature"] == feat]
        if not row.empty:
            row = row.iloc[0]
            p_str = f"p={row['p']:.4e}" if pd.notna(row["p"]) else "p=n/a"
            hr_str = f"HR={row['HR']:.3f}" if pd.notna(row["HR"]) else "HR=n/a"
            lines.append(f"  {i:2d}. {feat:<30s}  {hr_str}  {p_str}")
        else:
            lines.append(f"  {i:2d}. {feat}")
    lines += ["", sep]

    report = "\n".join(lines)
    print("\n" + report)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[Output] Selection report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("LITERATURE-BASED FEATURE SELECTION -- COX PIPELINE")
    print("=" * 72)

    # ── Load & preprocess ─────────────────────────────────────────────────
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.")

    # ── Compute time-varying demographic features ─────────────────────────
    df = add_computed_features(df)

    # ── Validate which features exist ─────────────────────────────────────
    print(f"\n[Features] Checking {len(LITERATURE_FEATURES)} literature features...")
    available, missing = validate_features(df, LITERATURE_FEATURES)
    print(f"  Available: {len(available)}")
    if missing:
        print(f"  Missing:   {len(missing)} -> {missing}")

    # ── Apply V9 guard ────────────────────────────────────────────────────
    screened = apply_v9_guard(df, available)
    v9_excluded = [f for f in available if f not in screened]
    print(f"  After V9 guard: {len(screened)} features")

    if not screened:
        raise ValueError("No literature features available after V9 guard.")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: UNIVARIATE SCREENING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"STAGE 1: UNIVARIATE SCREENING ({len(screened)} features)")
    print(f"{'='*72}")

    t0 = time.time()
    screening = run_univariate_screening(df, screened)
    print(f"[Screening] Completed in {time.time() - t0:.1f}s.")

    # Save full screening results
    screening_path = OUTPUT_DIR / "univariate_results.csv"
    screening.to_csv(screening_path, index=False)
    print(f"[Output] Univariate results saved to {screening_path}")

    converged = screening[screening["converged"] == True]
    n_sig = len(converged[converged["p"] < 0.05])
    print(f"  {len(converged)} converged, {n_sig} significant (p < 0.05).")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: FEATURE SELECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 2: FEATURE SELECTION")
    print(f"{'='*72}")

    selected = select_top_features(
        screening, df,
        p_cutoff=0.05,
        max_features=TOP_N_FEATURES,
        corr_threshold=CORR_THRESHOLD,
    )

    if not selected:
        raise ValueError("No significant features among the literature set.")

    # ── Write selection report ────────────────────────────────────────────
    report_path = OUTPUT_DIR / "feature_selection_report.txt"
    write_selection_report(
        screening, LITERATURE_FEATURES, missing, v9_excluded,
        selected, report_path,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: BUILD DATA & VIF SCREENING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 3: BUILD LONG FORMAT & VIF SCREENING")
    print(f"{'='*72}")

    long_df = build_long_multi_feature(df, selected)

    if long_df.empty or long_df["event"].sum() < 5:
        raise ValueError(
            f"Too few events ({int(long_df['event'].sum())}) after building long format."
        )

    n_patients = long_df["id"].nunique()
    n_events = int(long_df["event"].sum())
    n_intervals = len(long_df)
    print(f"[Long format] Patients: {n_patients} | Events: {n_events} | "
          f"Intervals: {n_intervals}")

    # Save long-format data (before standardisation)
    long_path = PREPROCESSED_DATA_DIR / "PW_litterature_long_format_data.csv"

    long_df.to_csv(long_path, index=False)
    print(f"[Output] Long-format data saved to {long_path}")

    # Invert PIGD if present
    if "PIGD" in long_df.columns:
        long_df["PIGD"] = 4 - long_df["PIGD"]
        print("[PIGD] Inverted PIGD (4 - PIGD) so higher = worse.")

    # VIF screening
    final_features, vif_df = iterative_vif_selection(
        long_df, selected, vif_threshold=VIF_THRESHOLD
    )

    if not final_features:
        raise ValueError("No features remain after VIF screening.")

    epv = n_events / len(final_features)
    print(f"[Final] {len(final_features)} features after VIF. "
          f"EPV = {epv:.1f}  ({'LOW' if epv < 10 else 'ok'})")

    # Save VIF + correlation
    corr_matrix = long_df[final_features].corr()
    save_vif_results(vif_df, corr_matrix, OUTPUT_DIR)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: MULTIVARIABLE COX MODEL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 4: MULTIVARIABLE COX MODEL")
    print(f"{'='*72}")

    long_df, _ = standardize_features(long_df, final_features)
    ctv = fit_cox_model(long_df, final_features)

    results = build_results_table(ctv, sort_by="p")

    print("\n" + "=" * 72)
    print("RESULTS TABLE")
    print("=" * 72)
    print(results.to_string(index=False))
    print("=" * 72)

    results_path = OUTPUT_DIR / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[Output] Results saved to {results_path}")

    perf_path = OUTPUT_DIR / "multivariable_model_performance.txt"
    evaluate_model_performance(ctv, long_df, final_features, perf_path)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 5: SCHOENFELD DIAGNOSTICS")
    print(f"{'='*72}")

    run_schoenfeld_test(long_df, ctv, final_features, OUTPUT_DIR)

    plot_path = OUTPUT_DIR / "multivariable_forest_plot.png"
    plot_forest(results, plot_path)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*72}")
    print(f"[Done] Total time: {elapsed:.0f}s")
    print(f"[Output] All results saved to {OUTPUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
