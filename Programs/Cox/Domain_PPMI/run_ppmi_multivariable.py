"""
run_ppmi_multivariable.py
--------------------------
Multivariable Cox time-varying model per domain on PPMI data.
Mirrors Domain/run_domain_multivariable.py exactly.

Pipeline per domain
    1. Load univariate results from Results/Cox/Domain_PPMI/<domain>/univariate_results.csv
    2. Select significant features (p < 0.05, converged), cap at MAX_FEATURES
    3. Build long-format data for those features
    4. Apply feature inversions (higher=better -> negate)
    5. Iterative VIF screening (remove highest VIF until all < VIF_THRESHOLD)
    6. Standardise features (z-score)
    7. Fit multivariable Cox time-varying model
    8. Schoenfeld residual test
    9. Save results

Combined output
    Results/Cox/Domain_PPMI/domain_multivariable_summary.csv
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
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Domain_PPMI/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PPMI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from ppmi_feature_map  import DOMAINS, apply_feature_inversions
from ppmi_data_loader  import load_ppmi, build_ppmi_long_format
from cox_diagnostics   import run_schoenfeld_test

# Multivariable functions from shared model
_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

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
P_CUTOFF      = 0.05
VIF_THRESHOLD = 6.0
MAX_FEATURES  = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_univariate_results(domain: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / domain / "univariate_results.csv"
    if not path.exists():
        print(f"  [SKIP] No univariate results at {path}")
        return None
    return pd.read_csv(path)


def select_candidates(uni_results: pd.DataFrame) -> list[str]:
    """Select candidate features with quasi-separation filter.

    Mirrors ParkWest: converged, p < cutoff, |coef| <= 3.0 (exclude
    near-constant features that produce artifact extreme HRs).
    """
    MAX_COEF = 3.0
    sig = uni_results[
        (uni_results["converged"] == True) &
        (uni_results["p"] < P_CUTOFF) &
        (uni_results["coef"].abs() <= MAX_COEF)
    ].sort_values("p")

    n_excluded = len(uni_results[
        (uni_results["converged"] == True) &
        (uni_results["p"] < P_CUTOFF) &
        (uni_results["coef"].abs() > MAX_COEF)
    ])
    if n_excluded > 0:
        print(f"  [Selection] Excluded {n_excluded} features with |coef| > {MAX_COEF} "
              f"(quasi-separation artifacts)")

    return sig["Feature"].tolist()[:MAX_FEATURES]


# ---------------------------------------------------------------------------
# Per-domain pipeline
# ---------------------------------------------------------------------------

def run_domain_multivariable(domain: str, df: pd.DataFrame) -> dict | None:

    print(f"\n{'='*72}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*72}")

    domain_dir = OUTPUT_DIR / domain
    domain_dir.mkdir(exist_ok=True)

    uni_results = load_univariate_results(domain)
    if uni_results is None:
        return None

    candidates = select_candidates(uni_results)
    if not candidates:
        print(f"  [SKIP] No features with p < {P_CUTOFF}")
        return {"domain": domain, "status": "skipped_no_significant",
                "n_candidates": 0, "n_after_vif": 0,
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "-",
                "top_HR": None, "top_p": None}

    print(f"  Candidates (p < {P_CUTOFF}): {candidates}")

    # Build long format
    long_df = build_ppmi_long_format(df, candidates)
    if long_df.empty or long_df["event"].sum() < 5:
        print(f"  [SKIP] Too few events "
              f"({int(long_df['event'].sum()) if not long_df.empty else 0})")
        return {"domain": domain, "status": "skipped_too_few_events",
                "n_candidates": len(candidates), "n_after_vif": 0,
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "-",
                "top_HR": None, "top_p": None}

    # Drop rows where any candidate feature is NaN (LOCF can't fill first obs)
    feat_cols_present = [f for f in candidates if f in long_df.columns]
    long_df = long_df.dropna(subset=feat_cols_present, how="any")

    if long_df.empty or long_df["event"].sum() < 5:
        print(f"  [SKIP] Too few events after dropping NaN rows.")
        return {"domain": domain, "status": "skipped_too_few_events_after_dropna",
                "n_candidates": len(candidates), "n_after_vif": 0,
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "-",
                "top_HR": None, "top_p": None}

    n_patients = long_df["id"].nunique()
    n_events   = int(long_df["event"].sum())
    print(f"  Patients: {n_patients} | Events: {n_events} | "
          f"Intervals: {len(long_df)}")

    # Feature inversions
    long_df, inverted_feats = apply_feature_inversions(long_df, candidates)

    # VIF screening
    final_features, vif_df = iterative_vif_selection(
        long_df, candidates, vif_threshold=VIF_THRESHOLD
    )
    if not final_features:
        print("  [SKIP] No features remain after VIF screening.")
        return {"domain": domain, "status": "skipped_vif_removed_all",
                "n_candidates": len(candidates), "n_after_vif": 0,
                "n_patients": n_patients, "n_events": n_events, "epv": None,
                "final_features": "", "top_feature": "-",
                "top_HR": None, "top_p": None}

    corr_matrix = long_df[final_features].corr()
    save_vif_results(vif_df, corr_matrix, domain_dir)

    epv = n_events / len(final_features)
    print(f"  EPV: {epv:.1f}  ({'LOW' if epv < 10 else 'ok'})")

    # Standardise
    long_df, _ = standardize_features(long_df, final_features)

    # Fit
    try:
        ctv = fit_cox_model(long_df, final_features)
    except Exception as e:
        print(f"  [ERROR] Cox model failed: {e}")
        return {"domain": domain, "status": f"error: {e}",
                "n_candidates": len(candidates),
                "n_after_vif": len(final_features),
                "n_patients": n_patients, "n_events": n_events,
                "epv": round(epv, 1),
                "final_features": ", ".join(final_features),
                "top_feature": "-", "top_HR": None, "top_p": None}

    # Results table
    results = build_results_table(ctv, sort_by="p")
    results.insert(0, "domain", domain)
    results["inverted"] = results["Feature"].isin(inverted_feats)
    results_path = domain_dir / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"  [Saved] {results_path}")

    # Performance
    perf_path = domain_dir / "multivariable_model_performance.txt"
    perf = evaluate_model_performance(ctv, long_df, final_features, perf_path)

    # Schoenfeld test
    run_schoenfeld_test(long_df, ctv, final_features, domain_dir)

    # Forest plot
    plot_path = domain_dir / "multivariable_forest_plot.png"
    plot_forest(results.drop(columns=["domain"]), plot_path)

    # Summary row
    sig_in_multi = results[results["p"] < P_CUTOFF] if "p" in results.columns else pd.DataFrame()
    best = results.iloc[0] if len(results) > 0 else None

    return {
        "domain":         domain,
        "status":         "ok",
        "n_candidates":   len(candidates),
        "n_after_vif":    len(final_features),
        "n_patients":     n_patients,
        "n_events":       n_events,
        "epv":            round(epv, 1),
        "final_features": ", ".join(final_features),
        "n_sig_in_multi": len(sig_in_multi),
        "top_feature":    best["Feature"] if best is not None else "-",
        "top_HR":         round(best["HR"], 3) if best is not None and pd.notna(best.get("HR")) else None,
        "top_p":          round(best["p"],  4) if best is not None and pd.notna(best.get("p"))  else None,
        "c_index":        round(perf["c_index"], 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("PPMI - DOMAIN-STRATIFIED MULTIVARIABLE COX MODELLING")
    print("=" * 72)

    df = load_ppmi()

    summary_rows = []
    for domain in DOMAINS:
        row = run_domain_multivariable(domain, df)
        if row is not None:
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "domain_multivariable_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Output] Domain summary -> {summary_path}")

    elapsed = time.time() - total_start
    print(f"\n{'='*72}")
    print("DOMAIN MULTIVARIABLE SUMMARY")
    print(f"{'='*72}")
    display_cols = ["domain","n_candidates","n_after_vif","n_events",
                    "epv","n_sig_in_multi","top_feature","top_HR","top_p",
                    "c_index","status"]
    show = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[show].to_string(index=False))
    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
