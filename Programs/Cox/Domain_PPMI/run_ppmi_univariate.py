"""
run_ppmi_univariate.py
----------------------
Univariate Cox time-varying screening for each clinical domain on the PPMI
dataset.  Mirrors Domain/run_domain_univariate.py exactly, but uses the PPMI
data loader and PPMI feature map instead of ParkWest.

Output (per domain)
    Results/Cox/Domain_PPMI/<domain>/univariate_results.csv

Output (combined)
    Results/Cox/Domain_PPMI/all_domains_univariate.csv
    Results/Cox/Domain_PPMI/domain_screening_summary.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Domain_PPMI/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

OUTPUT_DIR  = PROJECT_ROOT / "Results" / "Cox" / "Domain_PPMI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import PPMI-specific modules
# ---------------------------------------------------------------------------
from ppmi_feature_map import DOMAINS
from ppmi_data_loader  import (
    load_ppmi,
    build_ppmi_long_format,
    discover_dynamic_features,
)


# ---------------------------------------------------------------------------
# PPMI-specific univariate screening
# ---------------------------------------------------------------------------

def run_univariate_screening_ppmi(df: pd.DataFrame,
                                   features: list[str]) -> pd.DataFrame:
    """
    One-feature-at-a-time Cox time-varying screening on PPMI data.
    Builds long format per feature, standardises, fits Cox, collects results.
    """
    rows = []
    n_total = len(features)
    t0 = time.time()

    for i, feat in enumerate(features, 1):
        if i % 10 == 0 or i == 1 or i == n_total:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n_total - i) if i > 0 else 0
            print(f"  [{i}/{n_total}] '{feat}' ... "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

        long_df = build_ppmi_long_format(df, [feat])

        if long_df.empty or long_df["event"].sum() < 5:
            rows.append({"Feature": feat, "coef": np.nan, "HR": np.nan,
                         "HR_lower_95": np.nan, "HR_upper_95": np.nan,
                         "SE": np.nan, "z": np.nan, "p": np.nan,
                         "n_patients": 0, "n_events": 0, "n_intervals": 0,
                         "converged": False, "note": "too few events"})
            continue

        # Standardise
        scaler = StandardScaler()
        long_df[[feat]] = scaler.fit_transform(long_df[[feat]])

        if long_df[feat].std() < 1e-10:
            rows.append({"Feature": feat, "coef": np.nan, "HR": np.nan,
                         "HR_lower_95": np.nan, "HR_upper_95": np.nan,
                         "SE": np.nan, "z": np.nan, "p": np.nan,
                         "n_patients": long_df["id"].nunique(),
                         "n_events": int(long_df["event"].sum()),
                         "n_intervals": len(long_df),
                         "converged": False, "note": "zero variance"})
            continue

        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(long_df[["id","start","stop","event",feat]],
                    id_col="id", event_col="event",
                    start_col="start", stop_col="stop",
                    show_progress=False)
            s    = ctv.summary
            coef = s["coef"].iloc[0]
            hr   = np.exp(coef)
            se   = s["se(coef)"].iloc[0]
            z    = s["z"].iloc[0]
            p    = s["p"].iloc[0]
            if "exp(coef) lower 95%" in s.columns:
                hr_lo = s["exp(coef) lower 95%"].iloc[0]
                hr_hi = s["exp(coef) upper 95%"].iloc[0]
            elif "coef lower 95%" in s.columns:
                hr_lo = np.exp(s["coef lower 95%"].iloc[0])
                hr_hi = np.exp(s["coef upper 95%"].iloc[0])
            else:
                hr_lo = np.exp(coef - 1.96 * se)
                hr_hi = np.exp(coef + 1.96 * se)
            converged = True
        except Exception:
            coef = hr = hr_lo = hr_hi = se = z = p = np.nan
            converged = False

        rows.append({"Feature": feat, "coef": coef, "HR": hr,
                     "HR_lower_95": hr_lo, "HR_upper_95": hr_hi,
                     "SE": se, "z": z, "p": p,
                     "n_patients": long_df["id"].nunique(),
                     "n_events": int(long_df["event"].sum()),
                     "n_intervals": len(long_df),
                     "converged": converged,
                     "note": "" if converged else "failed to converge"})

    out = pd.DataFrame(rows).sort_values("p", na_position="last").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Domain runner
# ---------------------------------------------------------------------------

def run_domain(domain_name: str,
               features: list[str],
               df: pd.DataFrame) -> pd.DataFrame | None:

    print(f"\n{'='*72}")
    print(f"DOMAIN: {domain_name}  ({len(features)} features to screen)")
    print(f"{'='*72}")

    if not features:
        print("  No features available for this domain - skipping.")
        return None

    t0 = time.time()
    results = run_univariate_screening_ppmi(df, features)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    results.insert(0, "domain", domain_name)

    conv  = results[results["converged"] == True]
    n_sig = (conv["p"] < 0.05).sum()
    print(f"  {len(conv)} converged | {n_sig} significant (p < 0.05)")
    if n_sig > 0:
        top = conv[conv["p"] < 0.05][["Feature","HR","p"]].head(5)
        print(top.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("PPMI — DOMAIN-STRATIFIED UNIVARIATE COX SCREENING")
    print("=" * 72)

    # ── Load PPMI data ────────────────────────────────────────────────────
    df = load_ppmi()

    # ── Discover which features actually have longitudinal data ───────────
    all_domain_feats = [f for feats in DOMAINS.values() for f in feats]
    dynamic = discover_dynamic_features(df, all_domain_feats, min_visits=2)
    dynamic_set = set(dynamic)

    # ── Coverage report ───────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print("DOMAIN COVERAGE")
    print(f"{'-'*72}")
    for domain, feats in DOMAINS.items():
        avail = [f for f in feats if f in dynamic_set]
        print(f"  {domain:<35s}  {len(avail):3d}/{len(feats)} found")
    print(f"{'-'*72}\n")

    # ── Screen per domain ─────────────────────────────────────────────────
    all_results: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for domain_name, feats in DOMAINS.items():
        features = [f for f in feats if f in dynamic_set]
        results  = run_domain(domain_name, features, df)

        if results is None or results.empty:
            summary_rows.append({
                "domain": domain_name, "n_features_screened": 0,
                "n_converged": 0, "n_significant": 0,
                "top_feature": "-", "top_HR": None, "top_p": None,
            })
            continue

        # Save per-domain
        domain_dir = OUTPUT_DIR / domain_name
        domain_dir.mkdir(exist_ok=True)
        out_path = domain_dir / "univariate_results.csv"
        results.to_csv(out_path, index=False)
        print(f"  [Saved] {out_path}")
        all_results.append(results)

        conv = results[results["converged"] == True]
        sig  = conv[conv["p"] < 0.05]
        if len(sig) > 0:
            best = sig.iloc[0]
            top_feat = best["Feature"]
            top_hr   = round(best["HR"], 3) if pd.notna(best["HR"]) else None
            top_p    = round(best["p"],  4) if pd.notna(best["p"])  else None
        else:
            top_feat, top_hr, top_p = "none", None, None

        summary_rows.append({
            "domain":              domain_name,
            "n_features_screened": len(results),
            "n_converged":         len(conv),
            "n_significant":       len(sig),
            "top_feature":         top_feat,
            "top_HR":              top_hr,
            "top_p":               top_p,
        })

    # ── Save combined ─────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "all_domains_univariate.csv", index=False)
        print(f"\n[Output] Combined -> {OUTPUT_DIR / 'all_domains_univariate.csv'}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "domain_screening_summary.csv", index=False)

    elapsed = time.time() - total_start
    print(f"\n{'='*72}")
    print("DOMAIN SCREENING SUMMARY")
    print(f"{'='*72}")
    print(summary_df.to_string(index=False))
    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
