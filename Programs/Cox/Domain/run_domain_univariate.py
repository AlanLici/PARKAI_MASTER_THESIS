"""
run_domain_univariate.py
------------------------
Runs univariate Cox screening separately for each clinical domain subgroup.

Reuses all core functions from Common/univariate_cox_model_screening.py.
Data is loaded and preprocessed once; domains are screened in turn.

Output (per domain)
    Results/Cox/Domain_PW/<domain>/univariate_results.csv

Output (combined)
    Results/Cox/Domain_PW/all_domains_univariate.csv
    Results/Cox/Domain_PW/domain_screening_summary.csv
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

from domain_feature_lists import DOMAINS

_spec = importlib.util.spec_from_file_location(
    "univariate_screening",
    COMMON_DIR / "univariate_cox_model_screening.py",
)
uni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uni)

# Functions pulled from the screening module
load_and_filter        = uni.load_and_filter
compute_event_time     = uni.compute_event_time
discover_dynamic_features = uni.discover_dynamic_features
run_univariate_screening  = uni.run_univariate_screening

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PW"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def intersect_with_dynamic(domain_features: dict[str, bool],
                           dynamic_features: list[str]) -> list[str]:
    """Return domain feature names that exist as dynamic features in the data."""
    dynamic_set = set(dynamic_features)
    return [f for f in domain_features if f in dynamic_set]


def run_domain(domain_name: str,
               features: list[str],
               is_summary: dict[str, bool],
               df: pd.DataFrame) -> pd.DataFrame | None:
    """Run univariate screening for one domain and return the results table."""

    print(f"\n{'='*72}")
    print(f"DOMAIN: {domain_name}  ({len(features)} features to screen)")
    print(f"{'='*72}")

    if not features:
        print("  No dynamic features found for this domain — skipping.")
        return None

    t0 = time.time()
    results = run_univariate_screening(df, features)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Annotate
    results.insert(0, "domain", domain_name)
    results["is_summary_score"] = results["Feature"].map(is_summary).fillna(False)

    # Print quick summary
    converged = results[results["converged"] == True]
    n_sig = (converged["p"] < 0.05).sum()
    print(f"  {len(converged)} converged | {n_sig} significant (p < 0.05)")
    if n_sig > 0:
        top = converged[converged["p"] < 0.05][["Feature", "HR", "p"]].head(5)
        print(top.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("DOMAIN-STRATIFIED UNIVARIATE COX SCREENING")
    print("=" * 72)

    # ── Load data once ────────────────────────────────────────────────────
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.\n")

    # ── Discover all dynamic features (appears in >= 2 visits) ───────────
    dynamic_features = discover_dynamic_features(df)
    dynamic_set = set(dynamic_features)

    # ── Report coverage per domain before running ─────────────────────────
    print(f"\n{'-'*72}")
    print("DOMAIN COVERAGE (features defined vs found as dynamic in data)")
    print(f"{'-'*72}")
    for domain, feat_dict in DOMAINS.items():
        available = [f for f in feat_dict if f in dynamic_set]
        summary_avail = [f for f in available if feat_dict[f]]
        print(f"  {domain:<35s}  "
              f"{len(available):3d}/{len(feat_dict)} found  "
              f"({len(summary_avail)} summary scores)")
    print(f"{'-'*72}\n")

    # ── Run screening per domain ──────────────────────────────────────────
    all_results: list[pd.DataFrame] = []
    domain_summary_rows: list[dict] = []

    for domain_name, feat_dict in DOMAINS.items():
        features = intersect_with_dynamic(feat_dict, dynamic_features)
        results = run_domain(domain_name, features, feat_dict, df)

        if results is None or results.empty:
            domain_summary_rows.append({
                "domain":           domain_name,
                "n_features_screened": 0,
                "n_converged":      0,
                "n_significant":    0,
                "top_feature":      "—",
                "top_HR":           None,
                "top_p":            None,
            })
            continue

        # Save per-domain CSV
        domain_dir = OUTPUT_DIR / domain_name
        domain_dir.mkdir(exist_ok=True)
        out_path = domain_dir / "univariate_results.csv"
        results.to_csv(out_path, index=False)
        print(f"  [Saved] {out_path}")

        all_results.append(results)

        # Domain summary row
        conv = results[results["converged"] == True]
        sig  = conv[conv["p"] < 0.05]
        if len(sig) > 0:
            best = sig.iloc[0]
            top_feat = best["Feature"]
            top_hr   = round(best["HR"], 3) if pd.notna(best["HR"]) else None
            top_p    = round(best["p"], 4)  if pd.notna(best["p"])  else None
        else:
            top_feat, top_hr, top_p = "none", None, None

        domain_summary_rows.append({
            "domain":              domain_name,
            "n_features_screened": len(results),
            "n_converged":         len(conv),
            "n_significant":       len(sig),
            "top_feature":         top_feat,
            "top_HR":              top_hr,
            "top_p":               top_p,
        })

    # ── Save combined results ─────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_domains_univariate.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[Output] Combined results -> {combined_path}")

    domain_summary = pd.DataFrame(domain_summary_rows)
    summary_path = OUTPUT_DIR / "domain_screening_summary.csv"
    domain_summary.to_csv(summary_path, index=False)
    print(f"[Output] Domain summary   -> {summary_path}")

    # ── Print final overview ──────────────────────────────────────────────
    elapsed_total = time.time() - total_start
    print(f"\n{'='*72}")
    print("DOMAIN SCREENING SUMMARY")
    print(f"{'='*72}")
    print(domain_summary.to_string(index=False))
    print(f"\n[Done] Total time: {elapsed_total:.0f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
