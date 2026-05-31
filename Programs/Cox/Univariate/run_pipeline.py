"""
run_pipeline.py
---------------
Top-8 Univariate Cox Pipeline — Domain-Free Feature Selection

This pipeline discovers ALL dynamic features in the ParkWest dataset,
screens each with a univariate Cox time-varying model, selects the top 8
(significant, non-redundant), and fits a single multivariable Cox model.

Stages
------
1. Univariate screening  — auto-discover features, fit one Cox model each
2. Feature selection      — p < 0.05, correlation dedup (|r| > 0.70), top 8
3. VIF screening          — remove features with VIF >= 6.0
4. Multivariable Cox      — fit final model, evaluate, forest plot
5. Diagnostics            — Schoenfeld residuals (PH assumption test)

Data quality filters (shared with the other Cox pipelines)
    - V9 guard         : exclude features whose first measurement is after V9
    - 58% threshold    : patient must have data at >= 58% of attended visits
    - Median fill      : leading NaN filled with cross-sectional median
    - LOCF             : last observation carried forward for mid-sequence gaps

Output folder
    Results/Cox/Top8_PW/
        univariate_results.csv
        feature_selection_report.txt
        vif_table.csv
        correlation_matrix.csv
        correlation_heatmap.png
        multivariable_cox_results.csv
        multivariable_forest_plot.png
        multivariable_model_performance.txt
        schoenfeld_test.csv
        schoenfeld_residuals_plot.png

Usage
    python Programs/Cox/Univariate/run_pipeline.py
"""

import sys
import time
import importlib.util
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and shared imports
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Univariate/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

# Import univariate screening module
_spec_uni = importlib.util.spec_from_file_location(
    "univariate_screening",
    COMMON_DIR / "univariate_cox_model_screening.py",
)
uni = importlib.util.module_from_spec(_spec_uni)
_spec_uni.loader.exec_module(uni)

# Import multivariable module
_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

# Import cox diagnostics (Schoenfeld test)
from cox_diagnostics import run_schoenfeld_test

# Shared functions
load_and_filter           = uni.load_and_filter
compute_event_time        = uni.compute_event_time
discover_dynamic_features = uni.discover_dynamic_features
run_univariate_screening  = uni.run_univariate_screening
select_top_features       = uni.select_top_features
build_long_multi_feature  = uni.build_long_multi_feature
iterative_vif_selection   = multi.iterative_vif_selection
save_vif_results          = multi.save_vif_results
standardize_features      = multi.standardize_features
fit_cox_model             = multi.fit_cox_model
build_results_table       = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest               = multi.plot_forest

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TOP_N_FEATURES   = 8
CORR_THRESHOLD   = 0.70
VIF_THRESHOLD    = 6.0

# Results folder (Cox tables, plots, reports, diagnostics)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Top8_PW"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessed long-format data (the dataset the Cox model is trained on)
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "ParkWest" / "Preprocessed" / "Cox"
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature selection report
# ---------------------------------------------------------------------------

def write_selection_report(screening: pd.DataFrame,
                           selected: list[str],
                           skipped: list[tuple],
                           output_path: Path) -> None:
    sep = "=" * 72
    lines = [
        sep,
        "TOP-8 UNIVARIATE — FEATURE SELECTION REPORT",
        sep, "",
        f"Selection criteria",
        f"  Significant        : p < 0.05 in univariate Cox model",
        f"  Correlation dedup  : |r| > {CORR_THRESHOLD} at V4 -> skip",
        f"  Max features       : {TOP_N_FEATURES}",
        "",
    ]

    # Summarise screening
    converged = screening[screening["converged"] == True]
    n_sig = len(converged[converged["p"] < 0.05])
    lines.append(f"  Total features screened  : {len(screening)}")
    lines.append(f"  Converged                : {len(converged)}")
    lines.append(f"  Significant (p < 0.05)   : {n_sig}")
    lines.append("")

    if skipped:
        lines.append(f"  Skipped (correlated with already-selected):")
        for feat, corr_with, r in skipped:
            lines.append(f"    {feat:<30s}  |r|={abs(r):.2f}  with {corr_with}")
        lines.append("")

    lines += [sep, "SELECTED FEATURES", sep, ""]
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
    print("TOP-8 UNIVARIATE COX PIPELINE")
    print("=" * 72)

    # ── Load & preprocess ─────────────────────────────────────────────────
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: UNIVARIATE SCREENING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 1: UNIVARIATE SCREENING")
    print(f"{'='*72}")

    all_features = discover_dynamic_features(df)
    print(f"  {len(all_features)} dynamic features discovered.")

    t0 = time.time()
    screening = run_univariate_screening(df, all_features)
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
        raise ValueError("No features selected from univariate screening.")

    # Capture skipped features for the report (re-run selection logic)
    sig = screening[
        (screening["p"] < 0.05) & (screening["converged"] == True)
    ].sort_values("p")
    skipped = []
    seen = set()
    for _, row in sig.iterrows():
        feat = row["Feature"]
        if feat in selected:
            seen.add(feat)
            continue
        if feat not in seen and len(seen) < TOP_N_FEATURES:
            # This feature was skipped — find which selected feature it correlated with
            for sel_feat in selected:
                v4_cand = f"V4_{feat}" if f"V4_{feat}" in df.columns else None
                v4_sel = f"V4_{sel_feat}" if f"V4_{sel_feat}" in df.columns else None
                if v4_cand and v4_sel:
                    mask = df[v4_cand].notna() & df[v4_sel].notna()
                    if mask.sum() >= 20:
                        r = df.loc[mask, v4_cand].astype(float).corr(
                            df.loc[mask, v4_sel].astype(float))
                        if abs(r) > CORR_THRESHOLD:
                            skipped.append((feat, sel_feat, r))
                            break

    # Write selection report
    report_path = OUTPUT_DIR / "feature_selection_report.txt"
    write_selection_report(screening, selected, skipped, report_path)

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

    # Save long-format data
    long_path = PREPROCESSED_DATA_DIR / "PW_top8_long_format_data.csv"
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

    # Standardise
    long_df, _ = standardize_features(long_df, final_features)

    # Fit Cox model
    ctv = fit_cox_model(long_df, final_features)

    # Results table
    results = build_results_table(ctv, sort_by="p")

    print("\n" + "=" * 72)
    print("RESULTS TABLE")
    print("=" * 72)
    print(results.to_string(index=False))
    print("=" * 72)

    # Save results
    results_path = OUTPUT_DIR / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[Output] Results saved to {results_path}")

    # Model performance
    perf_path = OUTPUT_DIR / "multivariable_model_performance.txt"
    evaluate_model_performance(ctv, long_df, final_features, perf_path)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STAGE 5: SCHOENFELD DIAGNOSTICS")
    print(f"{'='*72}")

    run_schoenfeld_test(long_df, ctv, final_features, OUTPUT_DIR)

    # Forest plot
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
