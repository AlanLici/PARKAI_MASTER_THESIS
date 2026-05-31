"""
run_final_combined_model.py
---------------------------
Builds the final combined multivariable Cox model by taking the best feature
from each domain's multivariable model, then running VIF screening and fitting
a single cross-domain model.

Selection logic (per domain)
    1. If the domain has >= 1 significant feature (p < 0.05) in the multivariable
       model  →  take the most significant one.
    2. If no significant features but domain C-index >= C_INDEX_MIN  →  take the
       top feature by p-value (noted as "marginal" in the report).
    3. If domain C-index < C_INDEX_MIN  →  domain is excluded (no discriminative
       power in multivariable setting).

After selection
    VIF screening → standardise → multivariable Cox → save results.

Output folder
    Results/Cox/Domain_PW/final_combined_model/
        feature_selection_report.txt
        vif_table.csv
        correlation_matrix.csv
        correlation_heatmap.png
        multivariable_cox_results.csv
        multivariable_forest_plot.png
        multivariable_model_performance.txt
"""

import sys
import time
import importlib.util
import warnings
from pathlib import Path

import numpy as np
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

load_and_filter           = uni.load_and_filter
compute_event_time        = uni.compute_event_time
build_long_multi_feature  = uni.build_long_multi_feature
iterative_vif_selection   = multi.iterative_vif_selection
save_vif_results          = multi.save_vif_results
standardize_features      = multi.standardize_features
fit_cox_model             = multi.fit_cox_model
build_results_table       = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest               = multi.plot_forest

RESULTS_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PW"
OUTPUT_DIR  = RESULTS_DIR / "final_combined_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessed long-format data for the final combined Cox model
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "ParkWest" / "Preprocessed" / "Cox"
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
P_CUTOFF      = 0.05   # threshold for "significant" in domain multivariable model
C_INDEX_MIN   = 0.75   # minimum domain C-index to include marginal features
VIF_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Feature selection from domain results
# ---------------------------------------------------------------------------

def select_feature_from_domain(domain: str,
                                summary_row: pd.Series) -> dict:
    """
    Select the best representative feature for a domain.

    Returns a dict with keys: feature, p, HR, status, reason
    """
    domain_dir = RESULTS_DIR / domain
    results_path = domain_dir / "multivariable_cox_results.csv"

    if not results_path.exists():
        return {"domain": domain, "feature": None,
                "status": "excluded", "reason": "no multivariable results file"}

    results = pd.read_csv(results_path)
    # Drop domain column if present
    if "domain" in results.columns:
        results = results.drop(columns=["domain"])

    if results.empty:
        return {"domain": domain, "feature": None,
                "status": "excluded", "reason": "empty results table"}

    results = results.sort_values("p").reset_index(drop=True)
    top_row  = results.iloc[0]
    top_feat = top_row["Feature"]
    top_p    = top_row["p"]
    top_hr   = top_row.get("HR", np.nan)

    c_index  = summary_row.get("c_index", np.nan)
    n_events = summary_row.get("n_events", 0)

    # ── Decision logic ──────────────────────────────────────────────────
    if top_p < P_CUTOFF:
        return {"domain": domain, "feature": top_feat,
                "p": round(top_p, 4), "HR": round(top_hr, 3),
                "c_index": round(c_index, 3) if pd.notna(c_index) else None,
                "n_events": int(n_events),
                "status": "included_significant",
                "reason": f"p={top_p:.4f} < {P_CUTOFF} in domain multivariable model"}

    if pd.notna(c_index) and c_index >= C_INDEX_MIN:
        return {"domain": domain, "feature": top_feat,
                "p": round(top_p, 4), "HR": round(top_hr, 3),
                "c_index": round(c_index, 3),
                "n_events": int(n_events),
                "status": "included_marginal",
                "reason": (f"no significant features, but C-index={c_index:.3f} "
                           f">= {C_INDEX_MIN} threshold; best feature p={top_p:.4f}")}

    return {"domain": domain, "feature": None,
            "p": round(top_p, 4), "HR": round(top_hr, 3),
            "c_index": round(c_index, 3) if pd.notna(c_index) else None,
            "n_events": int(n_events),
            "status": "excluded",
            "reason": (f"no significant features and C-index={c_index:.3f} "
                       f"< {C_INDEX_MIN} threshold")}


def write_selection_report(selections: list[dict],
                           final_features: list[str],
                           output_path: Path) -> None:
    sep = "=" * 72
    lines = [
        sep,
        "FINAL COMBINED MODEL — FEATURE SELECTION REPORT",
        sep, "",
        f"Selection criteria",
        f"  Included (significant) : feature with p < {P_CUTOFF} in domain multivariable model",
        f"  Included (marginal)    : best feature when domain C-index >= {C_INDEX_MIN}",
        f"  Excluded               : domain C-index < {C_INDEX_MIN} (no useful signal)",
        "", sep, "PER-DOMAIN DECISIONS", sep, "",
    ]

    for sel in selections:
        status_tag = {
            "included_significant": "[INCLUDED - significant]",
            "included_marginal":    "[INCLUDED - marginal]",
            "excluded":             "[EXCLUDED]",
        }.get(sel["status"], f"[{sel['status']}]")

        lines.append(f"  {sel['domain']:<30s}  {status_tag}")
        lines.append(f"    Feature : {sel.get('feature') or '—'}")
        lines.append(f"    Reason  : {sel.get('reason', '')}")
        hr_str = f"{sel['HR']:.3f}" if sel.get("HR") and pd.notna(sel["HR"]) else "—"
        p_str  = f"{sel['p']:.4f}"  if sel.get("p")  and pd.notna(sel["p"])  else "—"
        ci_str = f"{sel['c_index']:.3f}" if sel.get("c_index") and pd.notna(sel["c_index"]) else "—"
        lines.append(f"    HR={hr_str}  p={p_str}  Domain C-index={ci_str}")
        lines.append("")

    lines += [
        sep,
        "FINAL FEATURE SET (entering combined model)",
        sep, "",
    ]
    for i, f in enumerate(final_features, 1):
        lines.append(f"  {i:2d}. {f}")
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
    print("FINAL COMBINED CROSS-DOMAIN COX MODEL")
    print("=" * 72)

    # ── Load domain multivariable summary ─────────────────────────────────
    summary_path = RESULTS_DIR / "domain_multivariable_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run run_domain_multivariable.py first.\n"
            f"Expected: {summary_path}"
        )
    summary_df = pd.read_csv(summary_path).set_index("domain")

    # ── Select best feature from each domain ─────────────────────────────
    print("\n[Selection] Choosing best feature from each domain...\n")
    selections = []
    for domain in DOMAINS:
        if domain not in summary_df.index:
            selections.append({"domain": domain, "feature": None,
                                "status": "excluded", "reason": "domain not in summary"})
            continue
        sel = select_feature_from_domain(domain, summary_df.loc[domain])
        selections.append(sel)
        tag = sel["status"].replace("_", " ").upper()
        feat_str = sel.get("feature") or "—"
        print(f"  {domain:<30s}  {tag:<28s}  {feat_str}")

    # ── Collect final feature set ──────────────────────────────────────────
    candidate_features = [s["feature"] for s in selections if s["feature"] is not None]
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for f in candidate_features:
        if f not in seen:
            seen.add(f)
            unique_candidates.append(f)
    candidate_features = unique_candidates

    print(f"\n[Selection] {len(candidate_features)} candidate features selected: "
          f"{candidate_features}")

    if not candidate_features:
        raise ValueError("No features selected from any domain.")

    # ── Load and preprocess data ──────────────────────────────────────────
    print("\n[Data] Loading and preprocessing...")
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.")

    # ── Build long-format data ────────────────────────────────────────────
    long_df = build_long_multi_feature(df, candidate_features)

    if long_df.empty or long_df["event"].sum() < 5:
        raise ValueError(
            f"Too few events ({int(long_df['event'].sum())}) after building long format."
        )

    n_patients  = long_df["id"].nunique()
    n_events    = int(long_df["event"].sum())
    n_intervals = len(long_df)
    print(f"[Long format] Patients: {n_patients} | Events: {n_events} | "
          f"Intervals: {n_intervals}")

    # ── Invert PIGD if present ────────────────────────────────────────────
    if "PIGD" in long_df.columns:
        long_df["PIGD"] = 4 - long_df["PIGD"]
        print("[PIGD] Inverted PIGD (4 - PIGD) so higher = worse.")

    # ── Apply scale inversions (higher=better -> higher=worse) ───────────
    long_df, inverted_feats = apply_feature_inversions(long_df, candidate_features)

    # ── VIF screening ─────────────────────────────────────────────────────
    final_features, vif_df = iterative_vif_selection(
        long_df, candidate_features, vif_threshold=VIF_THRESHOLD
    )

    if not final_features:
        raise ValueError("No features remain after VIF screening.")

    epv = n_events / len(final_features)
    print(f"[Final] {len(final_features)} features after VIF screening. "
          f"EPV = {epv:.1f}  ({'LOW' if epv < 10 else 'ok'})")

    # ── Save VIF + correlation ────────────────────────────────────────────
    corr_matrix = long_df[final_features].corr()
    save_vif_results(vif_df, corr_matrix, OUTPUT_DIR)

    # ── Write selection report ────────────────────────────────────────────
    report_path = OUTPUT_DIR / "feature_selection_report.txt"
    write_selection_report(selections, final_features, report_path)

    # ── Save long-format data (before standardisation) to central Data folder ────
    long_path = PREPROCESSED_DATA_DIR / "PW_domain_long_format_data.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[Output] Long-format data saved to {long_path}")

    # ── Standardise ───────────────────────────────────────────────────────
    long_df, _ = standardize_features(long_df, final_features)

    # ── Fit Cox model ─────────────────────────────────────────────────────
    ctv = fit_cox_model(long_df, final_features)

    # ── Results table ─────────────────────────────────────────────────────
    results = build_results_table(ctv, sort_by="p")

    # Annotate with domain and inversion flag
    feat_to_domain = {s["feature"]: s["domain"]
                      for s in selections if s["feature"] is not None}
    results.insert(1, "domain", results["Feature"].map(feat_to_domain).fillna("—"))
    results["inverted"] = results["Feature"].isin(inverted_feats)

    print("\n" + "=" * 72)
    print("FINAL COMBINED MODEL — RESULTS")
    print("=" * 72)
    print(results.to_string(index=False))
    print("=" * 72)

    results_path = OUTPUT_DIR / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[Output] Results saved to {results_path}")

    # ── Model performance ─────────────────────────────────────────────────
    perf_path = OUTPUT_DIR / "multivariable_model_performance.txt"
    evaluate_model_performance(ctv, long_df, final_features, perf_path)

    # ── Schoenfeld test (PH assumption) ───────────────────────────────────
    run_schoenfeld_test(long_df, ctv, final_features, OUTPUT_DIR)

    # ── Forest plot ───────────────────────────────────────────────────────
    plot_path = OUTPUT_DIR / "multivariable_forest_plot.png"
    plot_forest(results.drop(columns=["domain"], errors="ignore"), plot_path)

    elapsed = time.time() - total_start
    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"[Output] All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
