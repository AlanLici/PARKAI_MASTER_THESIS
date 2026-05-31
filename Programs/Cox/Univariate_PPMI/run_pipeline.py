"""
Programs/Cox/Univariate_PPMI/run_pipeline.py
--------------------------------------------
PPMI external validation of the ParkWest top-8 univariate Cox feature set.

VARIANT A (validation fit): the ParkWest-selected feature set is translated
to PPMI columns via the cross-cohort feature map; a single multivariable Cox
time-varying model is then fit on PPMI WITHOUT re-running the univariate
screen, correlation deduplication, or VIF refinement on PPMI.

Source (ParkWest features):
    Results/Cox/Top8_PW/multivariable_cox_results.csv

Output: Results/Cox/Univariate_PPMI/
    feature_mapping.txt             cross-cohort mapping decisions
    long_format_data.csv            PPMI counting-process data
    multivariable_cox_results.csv   Cox model HRs
    multivariable_forest_plot.png
    multivariable_model_performance.txt
    vif_table.csv                   informational (no refinement performed)
    correlation_matrix.csv          + correlation_heatmap.png
    schoenfeld_test.csv             proportional-hazards test
    schoenfeld_residuals_plot.png
"""

import sys
import time
import importlib.util
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Univariate_PPMI/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Univariate_PPMI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessed long-format data (the dataset the PPMI Cox model is trained on)
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "PPMI" / "Preprocessed" / "Cox"
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input: ParkWest top-8 pipeline's selected features (Variant A validation)
PARKWEST_TOP8_CSV = PROJECT_ROOT / "Results" / "Cox" / "Top8_PW" / "multivariable_cox_results.csv"

from ppmi_data_loader import load_ppmi, build_ppmi_long_format
from ppmi_feature_map  import apply_feature_inversions
from cox_diagnostics   import run_schoenfeld_test

_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

standardize_features       = multi.standardize_features
fit_cox_model              = multi.fit_cox_model
build_results_table        = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest                = multi.plot_forest
save_vif_results           = multi.save_vif_results
compute_vif                = multi.compute_vif


# ---------------------------------------------------------------------------
# Cross-cohort feature map for the ParkWest top-8 final Cox features.
# ---------------------------------------------------------------------------
# Source: Results/Cox/Top8_PW/multivariable_cox_results.csv
#   PIGD_nevner, PIGD, UPO11, UPO_part2_ON_tot, UPO_part1_tot,
#   SEON, UPO_part3_tot, UPO10
#
# Each entry:
#   ParkWest_feature -> {
#       "ppmi"         : PPMI column name OR list of items to sum OR None to drop
#       "derived"      : True if computed from underlying items
#       "justification": one-line rationale
#       "status"       : "confirmed" | "derived" | "dropped"
#   }
#
# Notes:
#   - UPO10 / UPO11 verified via the ParkWest Metadata sheet
#     (UPO10 = "UPO dressing ADL", UPO11 = "UPO hygiene ADL").
#   - UPO_part2_ON_tot in ParkWest already includes the event items
#     UPO13 (falling) and UPO14 (freezing); the natural PPMI analogue
#     NP2PTOT also includes its event items NP2WALK and NP2FREZ. The
#     counting-process construction reads features at interval start
#     and events at interval end, so there is no temporal contamination.
PARKWEST_TO_PPMI: dict[str, dict] = {
    "PIGD_nevner": {
        "ppmi": None,
        "derived": False,
        "justification": ("ParkWest-internal 'PIGD denominator' "
                          "(count of non-NaN PIGD items); no PPMI equivalent."),
        "status": "dropped",
    },
    "PIGD": {
        "ppmi": "PIGD",
        "derived": False,
        "justification": ("PIGD composite computed in both cohorts "
                          "(different items, same conceptual score)."),
        "status": "confirmed",
    },
    "UPO11": {
        "ppmi": "NP2HYGN",
        "derived": False,
        "justification": ("ParkWest UPO11 = 'UPO hygiene ADL' (metadata). "
                          "PPMI NP2HYGN = hygiene (MDS-UPDRS Part II)."),
        "status": "confirmed",
    },
    "UPO_part2_ON_tot": {
        "ppmi": "NP2PTOT",
        "derived": False,
        "justification": ("UPDRS Part II total (ON state, ParkWest) = NP2PTOT "
                          "(Part II total, PPMI). Both include the event items."),
        "status": "confirmed",
    },
    "UPO_part1_tot": {
        "ppmi": ["NP1RTOT", "NP1PTOT"],
        "derived": True,
        "justification": ("MDS-UPDRS Part I is split between rater (NP1RTOT) and "
                          "patient (NP1PTOT) sections; summed to form a full "
                          "Part I total comparable to ParkWest UPO_part1_tot."),
        "status": "derived",
    },
    "SEON": {
        "ppmi": "MSEADLG",
        "derived": False,
        "justification": ("Schwab & England ADL (ON state). MSEADLG is the PPMI "
                          "Modified S&E score; inverted via INVERT_FEATURES."),
        "status": "confirmed",
    },
    "UPO_part3_tot": {
        "ppmi": "NP3TOT",
        "derived": False,
        "justification": ("UPDRS Part III total = MDS-UPDRS Part III total "
                          "(motor examination)."),
        "status": "confirmed",
    },
    "UPO10": {
        "ppmi": "NP2DRES",
        "derived": False,
        "justification": ("ParkWest UPO10 = 'UPO dressing ADL' (metadata). "
                          "PPMI NP2DRES = dressing (MDS-UPDRS Part II)."),
        "status": "confirmed",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def derive_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PPMI derived columns (e.g. NP1RTOT + NP1PTOT) and rewrite the
    `ppmi` entry of the mapping to the new column name."""
    for pw_feat, spec in PARKWEST_TO_PPMI.items():
        if not spec.get("derived"):
            continue
        items = spec["ppmi"]
        if not isinstance(items, list):
            continue
        present = [c for c in items if c in df.columns]
        if not present:
            print(f"  [Derive] {pw_feat}: none of {items} found in PPMI data")
            spec["ppmi"] = None
            spec["status"] = "dropped"
            continue
        new_col = "_plus_".join(items)
        df[new_col] = df[present].sum(axis=1, skipna=False)
        n_valid = int(df[new_col].notna().sum())
        spec["ppmi"] = new_col
        print(f"  [Derive] {pw_feat}: sum({present}) -> '{new_col}' "
              f"({n_valid} non-null rows)")
    return df


def get_ppmi_feature_list() -> tuple[list[str], list[tuple]]:
    """Return the list of PPMI columns to fit and a per-feature mapping log."""
    feats, log = [], []
    for pw_feat, spec in PARKWEST_TO_PPMI.items():
        if spec["ppmi"] is None or spec["status"] == "dropped":
            log.append((pw_feat, None, spec["justification"], "dropped"))
            continue
        feats.append(spec["ppmi"])
        log.append((pw_feat, spec["ppmi"], spec["justification"], spec["status"]))
    return feats, log


def write_mapping_report(log: list[tuple], output_path: Path) -> None:
    sep = "=" * 72
    lines = [
        sep,
        "PPMI TOP-8 VALIDATION  -  CROSS-COHORT FEATURE MAPPING",
        sep, "",
        "Variant A: ParkWest top-8 features are translated to PPMI columns,",
        "  then a multivariable Cox time-varying model is fit on PPMI without",
        "  re-running univariate selection, correlation dedup, or VIF refinement.",
        "",
        f"Source (ParkWest): top8_univariate/results/multivariable_cox_results.csv",
        "",
        sep,
        "PER-FEATURE DECISIONS",
        sep, "",
    ]
    for pw, ppmi, justification, status in log:
        ppmi_str = ppmi if ppmi else "(dropped, no PPMI equivalent)"
        lines.append(f"  [{status:^9s}]  {pw:<22s}  ->  {ppmi_str}")
        lines.append(f"             {justification}")
        lines.append("")
    n_fit  = len([t for t in log if t[3] != "dropped"])
    n_drop = len([t for t in log if t[3] == "dropped"])
    lines += [
        sep,
        f"FITTED PPMI FEATURES : {n_fit}",
        f"DROPPED              : {n_drop}",
        sep,
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Output] Mapping report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    print("=" * 72)
    print("PPMI TOP-8 EXTERNAL VALIDATION  (Variant A: translated ParkWest features)")
    print("=" * 72)

    # ── 1. Sanity-check the ParkWest top-8 list ─────────────────────────
    if not PARKWEST_TOP8_CSV.exists():
        raise FileNotFoundError(f"ParkWest top-8 results not found: {PARKWEST_TOP8_CSV}")
    pw_top8 = pd.read_csv(PARKWEST_TOP8_CSV)
    pw_features = pw_top8["Feature"].tolist()
    actual   = set(pw_features)
    expected = set(PARKWEST_TO_PPMI.keys())
    if actual != expected:
        print("\n  [Warn] ParkWest top-8 features differ from this script's mapping table:")
        if actual - expected:
            print(f"         in CSV but not mapped: {sorted(actual - expected)}")
        if expected - actual:
            print(f"         mapped but not in CSV: {sorted(expected - actual)}")
        print("         (continuing with the intersection — review the mapping table)")
    else:
        print(f"\n  [OK] All {len(pw_features)} ParkWest features have a mapping entry.")

    # ── 2. Load PPMI data ───────────────────────────────────────────────
    print("\n[Load] Loading PPMI dataset...")
    df = load_ppmi(verbose=True)

    # ── 3. Compute derived features ─────────────────────────────────────
    print("\n[Derive] Computing PPMI derived columns (sums of underlying items)...")
    df = derive_combined_features(df)

    # ── 4. Build PPMI counting-process long format ──────────────────────
    ppmi_feats, log = get_ppmi_feature_list()
    write_mapping_report(log, OUTPUT_DIR / "feature_mapping.txt")

    print(f"\n[Long format] Building counting-process table for "
          f"{len(ppmi_feats)} features: {ppmi_feats}")
    long_df = build_ppmi_long_format(df, ppmi_feats)
    if long_df.empty or long_df["event"].sum() < 5:
        raise RuntimeError(
            f"PPMI long format empty or too few events "
            f"(events={int(long_df['event'].sum()) if not long_df.empty else 0}). "
            f"Cannot fit Cox model.")
    n_patients  = long_df["id"].nunique()
    n_events    = int(long_df["event"].sum())
    n_intervals = len(long_df)
    print(f"  Patients: {n_patients} | Events: {n_events} | Intervals: {n_intervals}")
    long_df.to_csv(PREPROCESSED_DATA_DIR / "PPMI_top8_long_format_data.csv", index=False)

    # ── 5. Feature inversion (e.g. MSEADLG higher = better) ─────────────
    long_df, inverted = apply_feature_inversions(long_df, ppmi_feats)

    # Keep only features actually present in the long format
    present_features = [f for f in ppmi_feats if f in long_df.columns]
    missing_features = [f for f in ppmi_feats if f not in long_df.columns]
    if missing_features:
        print(f"  [Warn] Mapped features missing from long format "
              f"(dropped by missingness / coverage filters): {missing_features}")
    if not present_features:
        raise RuntimeError("No features survived the long-format build.")
    print(f"  Final fitted feature count: {len(present_features)}")

    # ── 6. Complete-case filter ─────────────────────────────────────────
    # build_ppmi_long_format only drops rows where ALL features are NaN
    # (how="all"); the Cox fit requires every fitted row to be non-NaN in
    # every feature. Drop rows with any NaN in the joint feature set.
    before = len(long_df)
    long_df = long_df.dropna(subset=present_features, how="any").copy()
    if len(long_df) < before:
        n_pat_cc = long_df["id"].nunique()
        n_ev_cc  = int(long_df["event"].sum())
        print(f"  [Complete-case] Dropped {before - len(long_df)} rows with NaN "
              f"in any fitted feature; {len(long_df)} intervals remain "
              f"(patients={n_pat_cc}, events={n_ev_cc}).")
    if long_df.empty or long_df["event"].sum() < 5:
        raise RuntimeError(
            f"After complete-case filtering, too few events to fit Cox model "
            f"(events={int(long_df['event'].sum()) if not long_df.empty else 0}).")

    # ── 7. Standardise + multivariable Cox ──────────────────────────────
    long_df, _ = standardize_features(long_df, present_features)
    ctv = fit_cox_model(long_df, present_features)

    results = build_results_table(ctv, sort_by="p")
    results_path = OUTPUT_DIR / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print("\n" + "=" * 72)
    print("RESULTS TABLE")
    print("=" * 72)
    print(results.to_string(index=False))
    print(f"\n[Output] Results saved to {results_path}")

    # ── 8. Model performance ────────────────────────────────────────────
    perf_path = OUTPUT_DIR / "multivariable_model_performance.txt"
    evaluate_model_performance(ctv, long_df, present_features, perf_path)

    # ── 9. Informational VIF + correlation (no refinement) ──────────────
    vif_df = compute_vif(long_df, present_features) if len(present_features) > 1 else \
             pd.DataFrame({"Feature": present_features, "VIF": [1.0]*len(present_features)})
    corr_matrix = long_df[present_features].corr()
    save_vif_results(vif_df, corr_matrix, OUTPUT_DIR)

    # ── 10. Forest plot + Schoenfeld diagnostics ────────────────────────
    plot_forest(results, OUTPUT_DIR / "multivariable_forest_plot.png")
    run_schoenfeld_test(long_df, ctv, present_features, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"[Output] All results in: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
