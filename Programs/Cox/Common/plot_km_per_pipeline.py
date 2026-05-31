"""
plot_km_per_pipeline.py
-----------------------
Generate a Kaplan-Meier survival plot stratified by Cox risk-score
quartile, for a specified pipeline (literature, domain, or top-8).

For each pipeline:
  1. Load the final feature list from that pipeline's
     multivariable_cox_results.csv.
  2. Build the ParkWest counting-process long format on those features.
  3. Apply PIGD inversion + apply_feature_inversions.
  4. Standardise and fit a multivariable Cox time-varying model.
  5. Compute each patient's Cox risk score (linear predictor) at their
     first interval.
  6. Stratify patients into quartiles of the risk score.
  7. Fit Kaplan-Meier per quartile and run a multivariate log-rank test.
  8. Save a PNG showing all four survival curves with the log-rank
     p-value annotated in the title.

Usage
-----
    python plot_km_per_pipeline.py literature
    python plot_km_per_pipeline.py top8
    python plot_km_per_pipeline.py domain     # (regenerates the
                                              #  existing domain plot)
    python plot_km_per_pipeline.py all        # runs all three
"""
from __future__ import annotations

import sys
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and shared imports
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Common/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR                                   # this file lives in Common/
sys.path.insert(0, str(COMMON_DIR))

from domain_feature_lists import apply_feature_inversions   # noqa: E402

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
standardize_features      = multi.standardize_features
fit_cox_model             = multi.fit_cox_model

# ---------------------------------------------------------------------------
# Pipeline registry: pipeline label -> (feature CSV path, output dir,
#                                       output PNG name, display label)
# ---------------------------------------------------------------------------
RESULTS_ROOT = PROJECT_ROOT / "Results" / "Cox"

PIPELINES: dict[str, tuple[Path, Path, str, str]] = {
    "literature": (
        RESULTS_ROOT / "Litterature_PW" / "multivariable_cox_results.csv",
        RESULTS_ROOT / "Litterature_PW",
        "km_survival_by_risk_quartile_literature.png",
        "Literature-based",
    ),
    "domain": (
        RESULTS_ROOT / "Domain_PW" / "final_combined_model" / "multivariable_cox_results.csv",
        RESULTS_ROOT / "Domain_PW" / "final_combined_model",
        "km_survival_by_risk_quartile_domain.png",
        "Domain-stratified",
    ),
    "top8": (
        RESULTS_ROOT / "Top8_PW" / "multivariable_cox_results.csv",
        RESULTS_ROOT / "Top8_PW",
        "km_survival_by_risk_quartile_top8.png",
        "Top-8 univariate",
    ),
}

VIF_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------
def plot_km_for_pipeline(pipeline: str) -> dict:
    """Generate a KM-by-risk-quartile plot for one pipeline.
    Returns a dict with the log-rank p-value and cohort counts.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Options: {list(PIPELINES)}")
    feature_csv, output_dir, output_name, display_label = PIPELINES[pipeline]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not feature_csv.exists():
        raise FileNotFoundError(
            f"Feature CSV not found for pipeline '{pipeline}': {feature_csv}\n"
            f"Run the corresponding Cox pipeline first."
        )

    print("=" * 72)
    print(f"Pipeline: {display_label}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load the final features for this pipeline
    # ------------------------------------------------------------------
    features = pd.read_csv(feature_csv)["Feature"].tolist()
    print(f"[Features] {len(features)} from {feature_csv.name}: {features}")

    # ------------------------------------------------------------------
    # 2. Load ParkWest and build long format
    # ------------------------------------------------------------------
    df = load_and_filter(uni.DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()

    long_df = build_long_multi_feature(df, features)

    # ------------------------------------------------------------------
    # 3. PIGD inversion (4 - PIGD), same as the source pipelines
    # ------------------------------------------------------------------
    if "PIGD" in long_df.columns:
        long_df["PIGD"] = 4 - long_df["PIGD"]
        print("[PIGD] Inverted (4 - PIGD).")

    long_df, _ = apply_feature_inversions(long_df, features)

    # ------------------------------------------------------------------
    # 4. VIF screening (no-op since these features already passed VIF)
    # ------------------------------------------------------------------
    final_features, _ = iterative_vif_selection(
        long_df, features, vif_threshold=VIF_THRESHOLD
    )
    print(f"[VIF] {len(final_features)} features after refinement.")

    # ------------------------------------------------------------------
    # 5. Standardise + fit Cox
    # ------------------------------------------------------------------
    long_df, _ = standardize_features(long_df, final_features)
    ctv = fit_cox_model(long_df, final_features)
    betas = ctv.params_

    # ------------------------------------------------------------------
    # 6. Compute patient-level risk score (first interval Xβ)
    # ------------------------------------------------------------------
    patient_first = (
        long_df.sort_values(["id", "start"])
               .groupby("id", as_index=False)
               .first()
    )
    patient_first["risk_score"] = (
        patient_first[final_features].values @ betas.values
    )

    patient_outcomes = long_df.groupby("id", as_index=False).agg(
        event=("event", "max"),
        max_stop=("stop", "max"),
    )
    patient_outcomes["time_years"] = patient_outcomes["max_stop"] / 365.25

    km_df = patient_first[["id", "risk_score"]].merge(
        patient_outcomes[["id", "event", "time_years"]], on="id"
    )
    km_df["quartile"] = pd.qcut(
        km_df["risk_score"], q=4,
        labels=["Q1 (lowest risk)", "Q2", "Q3", "Q4 (highest risk)"],
    )

    # ------------------------------------------------------------------
    # 7. Fit KM per quartile + plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    quartile_colors = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"]

    for q, color in zip(km_df["quartile"].cat.categories, quartile_colors):
        sub = km_df[km_df["quartile"] == q]
        n_q  = len(sub)
        n_ev = int(sub["event"].sum())
        kmf.fit(sub["time_years"], sub["event"],
                label=f"{q}  (n={n_q}, events={n_ev})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    lr = multivariate_logrank_test(
        km_df["time_years"], km_df["quartile"], km_df["event"]
    )

    ax.set_xlabel("Time from baseline (years)", fontsize=12)
    ax.set_ylabel("Kaplan-Meier survival  S(t)", fontsize=12)
    ax.set_title(
        f"K-M survival by Cox risk-score quartile — {display_label} pipeline\n"
        f"Log-rank test across quartiles: p = {lr.p_value:.4g}",
        fontsize=13,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()

    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved -> {out_path}")
    print(f"[LogRank] p = {lr.p_value:.4g}  chi2 = {lr.test_statistic:.2f}  df = {lr.degrees_of_freedom}")

    return {
        "pipeline": display_label,
        "n_patients": len(km_df),
        "n_events": int(km_df["event"].sum()),
        "log_rank_p": lr.p_value,
        "log_rank_chi2": lr.test_statistic,
        "log_rank_df": lr.degrees_of_freedom,
        "output_path": out_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_km_per_pipeline.py {literature|domain|top8|all}")
        sys.exit(1)
    target = sys.argv[1].lower()

    if target == "all":
        results = [plot_km_for_pipeline(p) for p in PIPELINES]
    elif target in PIPELINES:
        results = [plot_km_for_pipeline(target)]
    else:
        raise ValueError(f"Unknown target '{target}'. Options: {list(PIPELINES) + ['all']}")

    if len(results) > 1:
        print()
        print("=" * 72)
        print("CROSS-PIPELINE LOG-RANK SUMMARY")
        print("=" * 72)
        print(f"{'Pipeline':<22s}  {'n':>5s}  {'events':>6s}  {'log-rank chi2':>13s}  {'p-value':>10s}")
        print("-" * 72)
        for r in results:
            print(f"{r['pipeline']:<22s}  {r['n_patients']:>5d}  {r['n_events']:>6d}  "
                  f"{r['log_rank_chi2']:>13.2f}  {r['log_rank_p']:>10.4g}")


if __name__ == "__main__":
    main()
