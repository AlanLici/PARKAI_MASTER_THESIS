"""
run_ppmi_final_model.py
-----------------------
Final cross-domain combined Cox model on PPMI data.
Mirrors Domain/run_final_combined_model.py exactly.

Selection logic (per domain)
    1. p < 0.05 in domain multivariable model  -> include (significant)
    2. No significant features but C-index >= C_INDEX_MIN -> include best (marginal)
    3. C-index < C_INDEX_MIN -> exclude

Output
    Results/Cox/Domain_PPMI/final_combined_model/
        feature_selection_report.txt
        vif_table.csv
        correlation_matrix.csv
        correlation_heatmap.png
        multivariable_cox_results.csv
        multivariable_forest_plot.png
        multivariable_model_performance.txt
        schoenfeld_test.csv
        schoenfeld_residuals_plot.png
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
# Paths & imports
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Domain_PPMI/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

RESULTS_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PPMI"
OUTPUT_DIR  = RESULTS_DIR / "final_combined_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessed long-format data
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "PPMI" / "Preprocessed" / "Cox"
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

from ppmi_feature_map import DOMAINS, apply_feature_inversions
from ppmi_data_loader import load_ppmi, build_ppmi_long_format
from cox_diagnostics  import run_schoenfeld_test

_spec_multi = importlib.util.spec_from_file_location(
    "multivariable_cox",
    COMMON_DIR / "multivariable_cox_feature_importance.py",
)
multi = importlib.util.module_from_spec(_spec_multi)
_spec_multi.loader.exec_module(multi)

iterative_vif_selection    = multi.iterative_vif_selection
save_vif_results           = multi.save_vif_results
standardize_features       = multi.standardize_features
fit_cox_model              = multi.fit_cox_model
build_results_table        = multi.build_results_table
evaluate_model_performance = multi.evaluate_model_performance
plot_forest                = multi.plot_forest

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
P_CUTOFF      = 0.05
C_INDEX_MIN   = 0.60
VIF_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Feature selection from domain results
# ---------------------------------------------------------------------------

def select_feature_from_domain(domain: str, summary_row: pd.Series) -> dict:
    domain_dir   = RESULTS_DIR / domain
    results_path = domain_dir / "multivariable_cox_results.csv"

    if not results_path.exists():
        return {"domain": domain, "feature": None,
                "status": "excluded", "reason": "no multivariable results file"}

    results = pd.read_csv(results_path)
    if "domain" in results.columns:
        results = results.drop(columns=["domain"])
    if results.empty:
        return {"domain": domain, "feature": None,
                "status": "excluded", "reason": "empty results table"}

    results  = results.sort_values("p").reset_index(drop=True)
    top_row  = results.iloc[0]
    top_feat = top_row["Feature"]
    top_p    = top_row["p"]
    top_hr   = top_row.get("HR", np.nan)
    c_index  = summary_row.get("c_index", np.nan)
    n_events = summary_row.get("n_events", 0)

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
        sep, "PPMI FINAL COMBINED MODEL - FEATURE SELECTION REPORT", sep, "",
        f"Selection criteria",
        f"  Included (significant) : p < {P_CUTOFF} in domain multivariable model",
        f"  Included (marginal)    : best feature when domain C-index >= {C_INDEX_MIN}",
        f"  Excluded               : C-index < {C_INDEX_MIN}",
        "", sep, "PER-DOMAIN DECISIONS", sep, "",
    ]

    for sel in selections:
        tag = {"included_significant": "[INCLUDED - significant]",
               "included_marginal":    "[INCLUDED - marginal]",
               "excluded":             "[EXCLUDED]"}.get(sel["status"], f"[{sel['status']}]")
        lines.append(f"  {sel['domain']:<30s}  {tag}")
        lines.append(f"    Feature : {sel.get('feature') or '-'}")
        lines.append(f"    Reason  : {sel.get('reason', '')}")
        hr_str = f"{sel['HR']:.3f}" if sel.get("HR") and pd.notna(sel["HR"]) else "-"
        p_str  = f"{sel['p']:.4f}"  if sel.get("p")  and pd.notna(sel["p"])  else "-"
        ci_str = f"{sel['c_index']:.3f}" if sel.get("c_index") and pd.notna(sel["c_index"]) else "-"
        lines.append(f"    HR={hr_str}  p={p_str}  Domain C-index={ci_str}")
        lines.append("")

    lines += [sep, "FINAL FEATURE SET (entering combined model)", sep, ""]
    for i, f in enumerate(final_features, 1):
        lines.append(f"  {i:2d}. {f}")
    lines += ["", sep]

    report = "\n".join(lines)
    print("\n" + report)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[Output] Selection report -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 72)
    print("PPMI - FINAL COMBINED CROSS-DOMAIN COX MODEL")
    print("=" * 72)

    # Load domain summary
    summary_path = RESULTS_DIR / "domain_multivariable_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run run_ppmi_multivariable.py first.\n{summary_path}")
    summary_df = pd.read_csv(summary_path).set_index("domain")

    # Select best feature per domain
    print("\n[Selection] Choosing best feature from each domain...\n")
    selections = []
    for domain in DOMAINS:
        if domain not in summary_df.index:
            selections.append({"domain": domain, "feature": None,
                                "status": "excluded", "reason": "domain not in summary"})
            continue
        sel = select_feature_from_domain(domain, summary_df.loc[domain])
        selections.append(sel)
        tag      = sel["status"].replace("_", " ").upper()
        feat_str = sel.get("feature") or "-"
        print(f"  {domain:<30s}  {tag:<28s}  {feat_str}")

    # Collect candidates (deduplicated)
    candidate_features = []
    seen = set()
    for s in selections:
        f = s.get("feature")
        if f and f not in seen:
            seen.add(f)
            candidate_features.append(f)

    print(f"\n[Selection] {len(candidate_features)} candidate features: "
          f"{candidate_features}")

    if not candidate_features:
        raise ValueError("No features selected from any domain.")

    # Load data
    print("\n[Data] Loading PPMI...")
    df = load_ppmi()

    # Build long format
    long_df = build_ppmi_long_format(df, candidate_features)
    # Drop rows where any candidate feature is NaN
    feat_cols_present = [f for f in candidate_features if f in long_df.columns]
    long_df = long_df.dropna(subset=feat_cols_present, how="any")

    if long_df.empty or long_df["event"].sum() < 5:
        raise ValueError(f"Too few events ({int(long_df['event'].sum())}).")

    n_patients  = long_df["id"].nunique()
    n_events    = int(long_df["event"].sum())
    n_intervals = len(long_df)
    print(f"[Long format] Patients: {n_patients} | Events: {n_events} | "
          f"Intervals: {n_intervals}")

    # Feature inversions
    long_df, inverted_feats = apply_feature_inversions(long_df, candidate_features)

    # VIF screening
    final_features, vif_df = iterative_vif_selection(
        long_df, candidate_features, vif_threshold=VIF_THRESHOLD
    )
    if not final_features:
        raise ValueError("No features remain after VIF screening.")

    epv = n_events / len(final_features)
    print(f"[Final] {len(final_features)} features after VIF. "
          f"EPV = {epv:.1f}  ({'LOW' if epv < 10 else 'ok'})")

    corr_matrix = long_df[final_features].corr()
    save_vif_results(vif_df, corr_matrix, OUTPUT_DIR)

    # Save long-format data to central Data folder
    long_df.to_csv(PREPROCESSED_DATA_DIR / "PPMI_domain_long_format_data.csv",
                   index=False)

    # Write selection report
    write_selection_report(selections, final_features,
                           OUTPUT_DIR / "feature_selection_report.txt")

    # Standardise
    long_df, _ = standardize_features(long_df, final_features)

    # Fit
    ctv = fit_cox_model(long_df, final_features)

    # Results
    results = build_results_table(ctv, sort_by="p")
    feat_to_domain = {s["feature"]: s["domain"]
                      for s in selections if s.get("feature")}
    results.insert(1, "domain", results["Feature"].map(feat_to_domain).fillna("-"))
    results["inverted"] = results["Feature"].isin(inverted_feats)

    print("\n" + "=" * 72)
    print("PPMI FINAL COMBINED MODEL - RESULTS")
    print("=" * 72)
    print(results.to_string(index=False))
    print("=" * 72)

    results_path = OUTPUT_DIR / "multivariable_cox_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[Output] Results -> {results_path}")

    evaluate_model_performance(ctv, long_df, final_features,
                               OUTPUT_DIR / "multivariable_model_performance.txt")

    run_schoenfeld_test(long_df, ctv, final_features, OUTPUT_DIR)

    plot_forest(results.drop(columns=["domain"], errors="ignore"),
                OUTPUT_DIR / "multivariable_forest_plot.png")

    elapsed = time.time() - total_start
    print(f"\n[Done] Total time: {elapsed:.0f}s")
    print(f"[Output] All results -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
