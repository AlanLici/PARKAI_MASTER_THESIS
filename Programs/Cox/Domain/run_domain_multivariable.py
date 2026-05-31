"""
run_domain_multivariable.py
---------------------------
For each clinical domain subgroup, fits a multivariable Cox time-varying model
using the significant features from the univariate screening.

Pipeline per domain
    1. Read univariate results from Results/Cox/Domain_PW/<domain>/univariate_results.csv
    2. Select significant features (p < 0.05, converged)
    3. Build long-format data for those features (with LOCF)
    4. Iterative VIF screening (remove highest VIF until all < 6)
    5. Standardise features (z-score)
    6. Fit multivariable Cox time-varying model
    7. Save results to Results/Cox/Domain_PW/<domain>/

Folder structure (per domain)
    Results/Cox/Domain_PW/<domain>/
        univariate_results.csv          <- already exists from previous step
        vif_table.csv                   <- new
        correlation_matrix.csv          <- new
        correlation_heatmap.png         <- new
        multivariable_cox_results.csv   <- new
        multivariable_forest_plot.png   <- new
        multivariable_model_performance.txt  <- new

Combined outputs
    Results/Cox/Domain_PW/domain_multivariable_summary.csv
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
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Domain/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

from domain_feature_lists import DOMAINS, apply_feature_inversions
from cox_diagnostics import run_schoenfeld_test

# -- Univariate script (for data loading, event computation, long-format builder)
_spec_uni = importlib.util.spec_from_file_location(
    "univariate_screening",
    COMMON_DIR / "univariate_cox_model_screening.py",
)
uni = importlib.util.module_from_spec(_spec_uni)
_spec_uni.loader.exec_module(uni)

# -- Multivariable script (for VIF, standardisation, Cox fitting, plotting)
_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

# Pull the functions we need
load_and_filter        = uni.load_and_filter
compute_event_time     = uni.compute_event_time
build_long_multi_feature = uni.build_long_multi_feature

iterative_vif_selection  = multi.iterative_vif_selection
save_vif_results         = multi.save_vif_results
standardize_features     = multi.standardize_features
fit_cox_model            = multi.fit_cox_model
build_results_table      = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest              = multi.plot_forest

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PW"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
P_CUTOFF    = 0.05   # univariate p-value threshold for entering domain model
VIF_THRESHOLD = 6.0
MAX_FEATURES  = 8    # cap per domain to protect EPV
POOL_SIZE     = 20   # how many univariate candidates to keep in the replacement pool

compute_vif = multi.compute_vif


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_univariate_results(domain: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / domain / "univariate_results.csv"
    if not path.exists():
        print(f"  [SKIP] No univariate results found at {path}")
        return None
    return pd.read_csv(path)


def select_candidates_with_pool(uni_results: pd.DataFrame):
    """Return (initial top-8, replacement pool) from all converged features."""
    converged = uni_results[uni_results["converged"] == True].sort_values("p")
    all_ranked = converged["Feature"].tolist()
    initial = all_ranked[:MAX_FEATURES]
    pool = all_ranked[MAX_FEATURES:POOL_SIZE]
    return initial, pool


def vif_with_replacement(
    df_wide: pd.DataFrame, initial: list[str], pool: list[str],
    domain_dir: Path,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """VIF screening that backfills removed features from the replacement pool."""
    current = list(initial)
    pool = list(pool)
    log_lines = ["VIF REPLACEMENT LOG", "=" * 50, "",
                 f"Initial ({len(current)}): {current}",
                 f"Pool ({len(pool)}): {pool}", ""]
    iteration = 0

    while len(current) > 1:
        long_df = build_long_multi_feature(df_wide, current)
        if long_df.empty or long_df["event"].sum() < 5:
            log_lines.append(f"[iter {iteration}] Too few events — stopping.")
            break

        if "PIGD" in long_df.columns and "PIGD" in current:
            long_df["PIGD"] = 4 - long_df["PIGD"]

        long_df, _ = apply_feature_inversions(long_df, current)

        vif_df = compute_vif(long_df, current)
        max_vif = vif_df["VIF"].iloc[0]
        worst = vif_df["Feature"].iloc[0]

        vif_str = ", ".join(f"{r['Feature']}={r['VIF']:.2f}"
                            for _, r in vif_df.iterrows())
        log_lines.append(f"[iter {iteration}] VIF: {vif_str}")

        if max_vif < VIF_THRESHOLD:
            log_lines.append(f"[iter {iteration}] All VIF < {VIF_THRESHOLD}. Done.")
            break

        current.remove(worst)
        replacement = None
        while pool:
            cand = pool.pop(0)
            if cand not in current:
                replacement = cand
                current.append(replacement)
                break

        if replacement:
            log_lines.append(
                f"[iter {iteration}] Removed {worst} (VIF={max_vif:.2f}), "
                f"replaced with {replacement}")
        else:
            log_lines.append(
                f"[iter {iteration}] Removed {worst} (VIF={max_vif:.2f}), "
                f"no replacement available")
        iteration += 1

    # Final rebuild
    long_df = build_long_multi_feature(df_wide, current)
    if "PIGD" in long_df.columns and "PIGD" in current:
        long_df["PIGD"] = 4 - long_df["PIGD"]
    long_df, inverted = apply_feature_inversions(long_df, current)

    final_vif = compute_vif(long_df, current) if len(current) > 1 else pd.DataFrame()

    log_lines.append(f"\nFinal features ({len(current)}): {current}")
    log_path = domain_dir / "vif_replacement_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"  [VIF] Replacement log -> {log_path}")

    return current, final_vif, inverted


def run_domain_multivariable(domain: str,
                             df: pd.DataFrame) -> dict | None:
    """
    Full pipeline for one domain. Returns a summary dict or None if skipped.
    """
    print(f"\n{'='*72}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*72}")

    domain_dir = OUTPUT_DIR / domain
    domain_dir.mkdir(exist_ok=True)

    # ── 1. Load univariate results ────────────────────────────────────────
    uni_results = load_univariate_results(domain)
    if uni_results is None:
        return None

    # ── 2. Select candidates with replacement pool ─────────────────────────
    initial, pool = select_candidates_with_pool(uni_results)
    if not initial:
        print(f"  [SKIP] No converged features in this domain.")
        return {"domain": domain, "status": "skipped_no_converged",
                "n_candidates": 0, "n_after_vif": 0,
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "—",
                "top_HR": None, "top_p": None}

    print(f"  Top {len(initial)} features: {initial}")
    if pool:
        print(f"  Replacement pool ({len(pool)}): {pool[:5]}{'...' if len(pool) > 5 else ''}")

    # ── 3-5. VIF with replacement (builds long format, applies inversions) ─
    final_features, vif_df, inverted_feats = vif_with_replacement(
        df, initial, pool, domain_dir)

    if not final_features:
        print(f"  [SKIP] No features remain after VIF screening.")
        return {"domain": domain, "status": "skipped_vif_removed_all",
                "n_candidates": len(initial), "n_after_vif": 0,
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "—",
                "top_HR": None, "top_p": None}

    # Rebuild long format with final features for model fitting
    long_df = build_long_multi_feature(df, final_features)
    if long_df.empty or long_df["event"].sum() < 5:
        print(f"  [SKIP] Too few events after VIF replacement.")
        return {"domain": domain, "status": "skipped_too_few_events",
                "n_candidates": len(initial), "n_after_vif": len(final_features),
                "n_patients": 0, "n_events": 0, "epv": None,
                "final_features": "", "top_feature": "—",
                "top_HR": None, "top_p": None}

    if "PIGD" in long_df.columns and "PIGD" in final_features:
        long_df["PIGD"] = 4 - long_df["PIGD"]
    long_df, inverted_feats = apply_feature_inversions(long_df, final_features)

    n_patients = long_df["id"].nunique()
    n_events = int(long_df["event"].sum())

    # Save VIF + correlation matrix
    corr_matrix = long_df[final_features].corr()
    save_vif_results(vif_df, corr_matrix, domain_dir)

    epv = n_events / len(final_features)
    print(f"  Patients: {n_patients} | Events: {n_events} | "
          f"Features: {len(final_features)} | EPV: {epv:.1f}")

    # ── Save long-format data (before standardisation) for inspection ────
    long_path = domain_dir / "long_format_data.csv"
    long_df.to_csv(long_path, index=False)
    print(f"  [Output] Long-format data saved to {long_path}")

    # ── 6. Standardise ────────────────────────────────────────────────────
    long_df, _ = standardize_features(long_df, final_features)

    # ── 7. Fit Cox model ──────────────────────────────────────────────────
    try:
        ctv = fit_cox_model(long_df, final_features)
    except Exception as e:
        print(f"  [ERROR] Cox model failed: {e}")
        return {"domain": domain, "status": f"error: {e}",
                "n_candidates": len(initial),
                "n_after_vif": len(final_features),
                "n_patients": n_patients, "n_events": n_events,
                "epv": round(epv, 1),
                "final_features": ", ".join(final_features),
                "top_feature": "—", "top_HR": None, "top_p": None}

    # ── 8. Results table ──────────────────────────────────────────────────
    results = build_results_table(ctv, sort_by="p")
    results.insert(0, "domain", domain)

    # Add inverted flag
    results["inverted"] = results["Feature"].isin(inverted_feats)

    results_path = domain_dir / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"  [Saved] {results_path}")

    # ── 9. Model performance ──────────────────────────────────────────────
    perf_path = domain_dir / "multivariable_model_performance.txt"
    perf = evaluate_model_performance(ctv, long_df, final_features, perf_path)

    # ── 9b. Schoenfeld test (PH assumption) ───────────────────────────────
    run_schoenfeld_test(long_df, ctv, final_features, domain_dir)

    # ── 10. Forest plot ───────────────────────────────────────────────────
    plot_path = domain_dir / "multivariable_forest_plot.png"
    plot_forest(results.drop(columns=["domain"]), plot_path)

    # ── Summary row ───────────────────────────────────────────────────────
    sig_in_multi = results[results["p"] < P_CUTOFF] if "p" in results.columns else pd.DataFrame()
    best = results.iloc[0] if len(results) > 0 else None

    return {
        "domain":           domain,
        "status":           "ok",
        "n_candidates":     len(initial),
        "n_after_vif":      len(final_features),
        "n_patients":       n_patients,
        "n_events":         n_events,
        "epv":              round(epv, 1),
        "final_features":   ", ".join(final_features),
        "n_sig_in_multi":   len(sig_in_multi),
        "top_feature":      best["Feature"] if best is not None else "—",
        "top_HR":           round(best["HR"], 3) if best is not None and pd.notna(best.get("HR")) else None,
        "top_p":            round(best["p"], 4)  if best is not None and pd.notna(best.get("p"))  else None,
        "c_index":          round(perf["c_index"], 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("DOMAIN-STRATIFIED MULTIVARIABLE COX MODELLING")
    print("=" * 72)

    # ── Load and preprocess data once ────────────────────────────────────
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.\n")

    # ── Run multivariable model per domain ────────────────────────────────
    summary_rows = []

    for domain in DOMAINS:
        row = run_domain_multivariable(domain, df)
        if row is not None:
            summary_rows.append(row)

    # ── Save combined summary ─────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "domain_multivariable_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Output] Domain summary -> {summary_path}")

    # ── Print final overview ──────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*72}")
    print("DOMAIN MULTIVARIABLE SUMMARY")
    print(f"{'='*72}")

    display_cols = ["domain", "n_candidates", "n_after_vif", "n_events",
                    "epv", "n_sig_in_multi", "top_feature", "top_HR",
                    "top_p", "c_index", "status"]
    show = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[show].to_string(index=False))

    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
