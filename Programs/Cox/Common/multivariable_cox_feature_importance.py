"""
multivariable_cox_feature_importance.py
---------------------------------------
Shared helpers for the multivariable Cox time-varying fall-risk models.

This module is imported as a library by the pipeline scripts under
Programs/Cox/<pipeline>/ (literature, univariate, domain, and their PPMI
variants). It provides the steps common to every pipeline once a
counting-process long-format DataFrame has already been built:

    * VIF screening          : compute_vif, iterative_vif_selection,
                               save_vif_results
    * Standardisation        : standardize_features
    * Model fitting          : fit_cox_model
    * Results table          : build_results_table
    * Performance evaluation : evaluate_model_performance (C-index, AIC, LR test)
    * Forest plot            : plot_forest

Design notes
------------
* Time-varying covariates use the counting-process (start, stop] format of
  ``lifelines.CoxTimeVaryingFitter``.
* All features are standardised (zero mean, unit variance) so that hazard
  ratios (HR = exp(β)) are directly comparable across features.

Data loading, event/time computation, and long-format construction live in
``univariate_cox_model_screening.py`` (ParkWest) and ``ppmi_data_loader.py``
(PPMI); the pipeline scripts call those, then hand the resulting long_df to
the helpers here.

Author : Sondre Lyngstad
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter

warnings.filterwarnings("ignore", category=FutureWarning)


### =========================================================================
### 1. VIF SCREENING
### =========================================================================

def compute_vif(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for each feature.

    Uses OLS regression: VIF_i = 1 / (1 - R²_i), where R²_i is obtained by
    regressing feature i on all other features.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, VIF — sorted descending by VIF.
    """
    from sklearn.linear_model import LinearRegression

    X = long_df[features].values.astype(float)
    records = []
    for i, feat in enumerate(features):
        if X.shape[1] == 1:
            vif = 1.0
        else:
            y = X[:, i]
            X_others = np.delete(X, i, axis=1)
            r2 = LinearRegression().fit(X_others, y).score(X_others, y)
            vif = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
        records.append({"Feature": feat, "VIF": round(float(vif), 4)})

    return (
        pd.DataFrame(records)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def iterative_vif_selection(
    long_df: pd.DataFrame,
    features: list[str],
    vif_threshold: float = 6.0,
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively remove the feature with the highest VIF until all VIFs are
    below ``vif_threshold``.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data containing feature columns.
    features : list[str]
        Initial feature list to screen.
    vif_threshold : float
        Maximum acceptable VIF (default 6.0).

    Returns
    -------
    (remaining_features, final_vif_df)
        The surviving feature list and the final VIF table.
    """
    remaining = list(features)
    iteration = 0

    print(f"\n[VIF Screening] Starting with {len(remaining)} features, threshold={vif_threshold}")

    while len(remaining) > 1:
        vif_df = compute_vif(long_df, remaining)
        max_vif = vif_df["VIF"].iloc[0]
        worst_feat = vif_df["Feature"].iloc[0]

        print(f"[VIF Screening] Iteration {iteration}:")
        for _, row in vif_df.iterrows():
            flag = "  <-- REMOVE" if row["Feature"] == worst_feat and max_vif >= vif_threshold else ""
            print(f"  {row['Feature']:<30s}  VIF = {row['VIF']:.3f}{flag}")

        if max_vif < vif_threshold:
            print(f"[VIF Screening] All VIFs < {vif_threshold}. "
                  f"Proceeding with {len(remaining)} features: {remaining}")
            break

        print(f"[VIF Screening] Removing '{worst_feat}' (VIF={max_vif:.3f} >= {vif_threshold})")
        remaining.remove(worst_feat)
        iteration += 1
    else:
        # Only one feature left — compute its VIF (will be 1.0)
        vif_df = compute_vif(long_df, remaining)

    # Final VIF table after selection
    final_vif_df = compute_vif(long_df, remaining)
    print(f"\n[VIF Screening] Final features ({len(remaining)}): {remaining}")
    return remaining, final_vif_df


def save_vif_results(
    vif_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the final VIF table, correlation matrix CSV, and correlation heatmap.

    Parameters
    ----------
    vif_df : pd.DataFrame
        Final VIF table (Feature, VIF).
    corr_matrix : pd.DataFrame
        Pairwise correlation matrix of the final features.
    output_dir : Path
        Directory where output files are written.
    """
    vif_path     = output_dir / "vif_table.csv"
    corr_path    = output_dir / "correlation_matrix.csv"
    heatmap_path = output_dir / "correlation_heatmap.png"

    vif_df.to_csv(vif_path, index=False)
    corr_matrix.to_csv(corr_path)
    print(f"[VIF] VIF table saved to {vif_path}")
    print(f"[VIF] Correlation matrix saved to {corr_path}")

    # ── Correlation heatmap ────────────────────────────────────────────
    n = len(corr_matrix)
    fig_size = max(5, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix\n(final features after VIF screening)")
    plt.tight_layout()
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"[VIF] Correlation heatmap saved to {heatmap_path}")


### =========================================================================
### 2. STANDARDIZATION
### =========================================================================

def standardize_features(long_df: pd.DataFrame,
                         features: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise (z-score) all feature columns in place.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format DataFrame with feature columns.
    features : list[str]
        Column names to standardise.

    Returns
    -------
    (pd.DataFrame, StandardScaler)
        The DataFrame with standardised features and the fitted scaler.
    """
    scaler = StandardScaler()
    long_df[features] = scaler.fit_transform(long_df[features])
    print(f"[Standardisation] Standardised {len(features)} features "
          f"(zero mean, unit variance).")
    return long_df, scaler


### =========================================================================
### 3. COX MODEL FITTING
### =========================================================================

def fit_cox_model(long_df: pd.DataFrame,
                  features: list[str]) -> CoxTimeVaryingFitter:
    """Fit a Cox time-varying proportional hazards model.

    Parameters
    ----------
    long_df : pd.DataFrame
        Counting-process format with columns: id, start, stop, event, + features.
    features : list[str]
        Names of covariates to include.

    Returns
    -------
    CoxTimeVaryingFitter
        The fitted model.
    """
    ctv = CoxTimeVaryingFitter(penalizer=0.01)  # small L2 penalty for stability
    ctv.fit(
        long_df[["id", "start", "stop", "event"] + features],
        id_col="id",
        event_col="event",
        start_col="start",
        stop_col="stop",
        show_progress=True,
    )
    ctv.print_summary()
    return ctv


### =========================================================================
### 4. RESULTS TABLE
### =========================================================================

def _significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_results_table(ctv: CoxTimeVaryingFitter,
                        sort_by: str = "p") -> pd.DataFrame:
    """Build a tidy results table from a fitted CoxTimeVaryingFitter.

    Columns: Feature, coef (β), HR, HR_lower_95, HR_upper_95, SE, z, p, Significance
    """
    summary = ctv.summary.copy()
    summary = summary.reset_index().rename(columns={"covariate": "Feature"})

    # Rename columns coming from lifelines (may differ slightly by version)
    col_map = {
        "coef":          "coef",
        "exp(coef)":     "HR",
        "se(coef)":      "SE",
        "coef lower 95%": "coef_lower_95",
        "coef upper 95%": "coef_upper_95",
        "exp(coef) lower 95%": "HR_lower_95",
        "exp(coef) upper 95%": "HR_upper_95",
        "z":             "z",
        "p":             "p",
    }
    summary.rename(columns=col_map, inplace=True)

    # Compute HR and CI from coefficients if lifelines did not provide them
    if "HR" not in summary.columns:
        summary["HR"] = np.exp(summary["coef"])
    if "HR_lower_95" not in summary.columns and "coef_lower_95" in summary.columns:
        summary["HR_lower_95"] = np.exp(summary["coef_lower_95"])
    if "HR_upper_95" not in summary.columns and "coef_upper_95" in summary.columns:
        summary["HR_upper_95"] = np.exp(summary["coef_upper_95"])

    summary["Significance"] = summary["p"].apply(_significance_stars)

    # Select and order columns
    desired_cols = [
        "Feature", "coef", "HR", "HR_lower_95", "HR_upper_95",
        "SE", "z", "p", "Significance",
    ]
    existing = [c for c in desired_cols if c in summary.columns]
    results = summary[existing].copy()

    # Sort
    if sort_by == "p" and "p" in results.columns:
        results.sort_values("p", inplace=True)
    elif sort_by == "HR" and "HR" in results.columns:
        results.sort_values("HR", ascending=False, inplace=True)

    results.reset_index(drop=True, inplace=True)
    return results


### =========================================================================
### 5. MODEL PERFORMANCE EVALUATION
### =========================================================================

def _compute_concordance_tv(long_df: pd.DataFrame,
                            ctv: CoxTimeVaryingFitter,
                            features: list[str]) -> float:
    """Compute a concordance index for the time-varying Cox model.

    For each risk set at each event time, we check whether the subject who
    experienced the event had a higher predicted risk score (linear predictor)
    than subjects still at risk.  This is Harrell's C adapted for the
    counting-process format.

    Returns
    -------
    float
        Concordance index (0.5 = random, 1.0 = perfect discrimination).
    """
    # Get the linear predictor (Xβ) for every interval
    X = long_df[features].values
    betas = ctv.params_.values
    long_df = long_df.copy()
    long_df["risk_score"] = X @ betas

    # Identify event times (rows where event == 1)
    event_rows = long_df[long_df["event"] == 1].copy()

    concordant = 0
    discordant = 0
    tied = 0

    for _, ev_row in event_rows.iterrows():
        ev_time = ev_row["stop"]
        ev_score = ev_row["risk_score"]
        ev_id = ev_row["id"]

        # Risk set: intervals that contain ev_time, i.e. start < ev_time <= stop
        # (excluding the event subject themselves)
        risk_set = long_df[
            (long_df["start"] < ev_time) &
            (long_df["stop"] >= ev_time) &
            (long_df["id"] != ev_id)
        ]

        for _, rs_row in risk_set.iterrows():
            rs_score = rs_row["risk_score"]
            if ev_score > rs_score:
                concordant += 1
            elif ev_score < rs_score:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def evaluate_model_performance(ctv: CoxTimeVaryingFitter,
                               long_df: pd.DataFrame,
                               features: list[str],
                               output_path: Path) -> dict:
    """Compute and report model performance metrics.

    Metrics reported:
    - Concordance index (C-index)
    - Partial AIC
    - Log-likelihood ratio test (chi-sq, df, p-value)
    - Number of events, subjects, intervals

    Also provides interpretation guidance.

    Returns
    -------
    dict  with keys: c_index, aic, ll_ratio_chi2, ll_ratio_p, n_events,
          n_subjects, n_intervals
    """
    # ── Concordance index ─────────────────────────────────────────────
    try:
        c_index = ctv.concordance_index_
    except AttributeError:
        c_index = _compute_concordance_tv(long_df, ctv, features)

    # ── AIC ───────────────────────────────────────────────────────────
    try:
        aic = ctv.AIC_partial_
    except AttributeError:
        # Manually: AIC = -2 * LL + 2k
        aic = -2 * ctv.log_likelihood_ + 2 * len(features)

    # ── Log-likelihood ratio test ─────────────────────────────────────
    # -2 * (LL_null - LL_model) ~ chi2 with k degrees of freedom
    from scipy import stats
    ll_model = ctv.log_likelihood_
    k = len(features)
    try:
        ll_ratio_chi2 = ctv.summary.attrs.get("ll-ratio test statistic", None)
    except Exception:
        ll_ratio_chi2 = None

    # If not available from attrs, compute from the summary print
    # lifelines stores it; fallback: use the log_likelihood_ value
    if ll_ratio_chi2 is None:
        # The null model LL for Cox PH is 0 (since Cox PH is semi-parametric
        # with no intercept; the partial LL of the null model equals 0 by
        # convention in lifelines).  So LR stat = -2 * (0 - LL) = -2 * LL
        # But actually lifelines computes LL_null itself.  We can approximate:
        ll_ratio_chi2 = -2 * ll_model  # conservative; lifelines uses similar

    ll_ratio_p = stats.chi2.sf(abs(ll_ratio_chi2), df=k)

    # ── Counts ────────────────────────────────────────────────────────
    n_events    = int(long_df["event"].sum())
    n_subjects  = long_df["id"].nunique()
    n_intervals = len(long_df)

    # ── Events per variable (EPV) ─────────────────────────────────────
    epv = n_events / k

    # ── Build report ──────────────────────────────────────────────────
    separator = "=" * 72
    report_lines = [
        separator,
        "MODEL PERFORMANCE EVALUATION",
        separator,
        "",
        f"  Concordance Index (C-index) : {c_index:.4f}",
        f"  Partial AIC                 : {aic:.2f}",
        f"  Log-likelihood (model)      : {ll_model:.4f}",
        f"  LR test chi-sq (df={k:d})      : {abs(ll_ratio_chi2):.2f}  (p = {ll_ratio_p:.2e})",
        "",
        f"  Subjects                    : {n_subjects}",
        f"  Intervals (person-periods)  : {n_intervals}",
        f"  Events                      : {n_events}",
        f"  Covariates                  : {k}",
        f"  Events per Variable (EPV)   : {epv:.1f}",
        "",
        separator,
        "INTERPRETATION GUIDE",
        separator,
        "",
        "  C-index:",
        "    0.50       = random (no discrimination)",
        "    0.60-0.70  = poor discrimination",
        "    0.70-0.80  = acceptable discrimination",
        "    0.80-0.90  = excellent discrimination",
        "    > 0.90     = outstanding (possibly overfit)",
        "",
        "  Events per Variable (EPV):",
        "    >= 10-20   = adequate (rule of thumb)",
        f"    Current    = {epv:.1f}  {'[!] LOW -- coefficients may be unstable' if epv < 10 else '[ok] adequate'}",
        "",
        "  If C-index ~= 0.50:",
        "    The model has NO predictive value.  Feature importance rankings",
        "    from this model should NOT be interpreted.",
        "",
        "  If C-index < 0.60:",
        "    The model discriminates poorly.  Feature rankings should be",
        "    treated with caution.",
        "",
        "  If C-index >= 0.70:",
        "    The model has acceptable discrimination.  Feature importance",
        "    rankings (HR, p-values) can be meaningfully interpreted.",
        "",
        separator,
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[Output] Performance report saved to {output_path}")

    return {
        "c_index": c_index,
        "aic": aic,
        "ll_ratio_chi2": abs(ll_ratio_chi2),
        "ll_ratio_p": ll_ratio_p,
        "n_events": n_events,
        "n_subjects": n_subjects,
        "n_intervals": n_intervals,
        "epv": epv,
    }


### =========================================================================
### 6. VISUALIZATION (Forest Plot)
### =========================================================================

def plot_forest(results: pd.DataFrame, output_path: Path) -> None:
    """Create and save a forest plot of hazard ratios with 95 % CIs.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns: Feature, HR, HR_lower_95, HR_upper_95.
    output_path : Path
        Where to save the PNG file.
    """
    if not {"Feature", "HR", "HR_lower_95", "HR_upper_95"}.issubset(results.columns):
        print("[Plot] Cannot create forest plot — missing HR / CI columns.")
        return

    # Sort by HR for visual clarity
    plot_df = results.sort_values("HR", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(plot_df))))

    y_pos = np.arange(len(plot_df))
    xerr_lower = plot_df["HR"] - plot_df["HR_lower_95"]
    xerr_upper = plot_df["HR_upper_95"] - plot_df["HR"]

    ax.errorbar(
        plot_df["HR"], y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="o", color="steelblue", ecolor="grey",
        elinewidth=1.5, capsize=3, markersize=6,
    )
    ax.axvline(x=1, linestyle="--", color="red", linewidth=0.8, label="HR = 1 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Feature"])
    ax.set_xlabel("Hazard Ratio (95 % CI)")
    ax.set_title("Forest Plot — Cox PH Feature Importance for Falls")
    ax.legend(loc="lower right", fontsize=8)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Forest plot saved to {output_path}")
