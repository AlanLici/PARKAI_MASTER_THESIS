"""
cox_diagnostics.py
------------------
Diagnostic tests for the Cox proportional hazards assumption.

Schoenfeld residual test
    For each feature, computes the residual at each event (actual feature
    value of the patient who fell minus the weighted mean across the risk set).
    If residuals correlate with time, the hazard ratio is NOT constant over
    time -> proportional hazards assumption is violated for that feature.

    Interpretation
        p > 0.05  : PH assumption holds  -> HR is a reliable summary
        p < 0.05  : PH assumption violated -> HR is a time-average; interpret
                    with caution and note as a limitation

Usage
    from cox_diagnostics import run_schoenfeld_test
    run_schoenfeld_test(long_df, ctv, features, output_dir)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Schoenfeld residuals
# ---------------------------------------------------------------------------

def _compute_schoenfeld_residuals(long_df: pd.DataFrame,
                                   ctv,
                                   features: list[str]) -> pd.DataFrame:
    """
    Compute Schoenfeld residuals for a fitted CoxTimeVaryingFitter.

    For each event i with event time t_i:
        r_ij = x_ij - sum_k(w_k * x_kj) / sum_k(w_k)

    where:
        x_ij  = feature j value for the patient who fell at t_i
        w_k   = exp(beta * x_k) for each subject k in the risk set at t_i
        r_ij  = Schoenfeld residual for feature j at event i

    Returns
    -------
    pd.DataFrame  columns: event_time, patient_id, <feature_1>, ..., <feature_n>
    """
    betas = ctv.params_.values          # (n_features,)
    long_df = long_df.copy()
    X = long_df[features].values        # (n_intervals, n_features)
    long_df["_lp"] = X @ betas          # linear predictor per interval

    event_rows = long_df[long_df["event"] == 1]
    residual_rows = []

    for _, ev in event_rows.iterrows():
        t      = ev["stop"]
        ev_id  = ev["id"]
        ev_x   = ev[features].values.astype(float)

        # Risk set: all intervals that contain t  (start < t <= stop)
        risk_set = long_df[
            (long_df["start"] < t) &
            (long_df["stop"]  >= t)
        ]
        if len(risk_set) == 0:
            continue

        weights = np.exp(risk_set["_lp"].values)
        w_sum   = weights.sum()
        if w_sum == 0:
            continue

        X_rs           = risk_set[features].values.astype(float)
        weighted_mean  = (weights[:, None] * X_rs).sum(axis=0) / w_sum
        residual       = ev_x - weighted_mean

        row = {"event_time": t, "patient_id": ev_id}
        for j, feat in enumerate(features):
            row[feat] = residual[j]
        residual_rows.append(row)

    return pd.DataFrame(residual_rows)


# ---------------------------------------------------------------------------
# Schoenfeld test (correlation + plot)
# ---------------------------------------------------------------------------

def run_schoenfeld_test(long_df: pd.DataFrame,
                        ctv,
                        features: list[str],
                        output_dir: Path) -> pd.DataFrame | None:
    """
    Test the proportional hazards assumption via Schoenfeld residuals.

    For each feature:
        - Computes Schoenfeld residuals at each event
        - Tests Spearman correlation of residuals with event time
        - p < 0.05 -> PH assumption violated for that feature

    Saves
    -----
    schoenfeld_test.csv            : summary table (one row per feature)
    schoenfeld_residuals_plot.png  : residuals vs time, one panel per feature
    """
    output_dir = Path(output_dir)

    print("\n[Schoenfeld] Computing residuals...")
    resid_df = _compute_schoenfeld_residuals(long_df, ctv, features)

    if resid_df.empty:
        print("[Schoenfeld] No residuals computed — skipping.")
        return None

    # ── Spearman correlation test per feature ─────────────────────────────
    test_rows = []
    for feat in features:
        if feat not in resid_df.columns:
            continue
        times  = resid_df["event_time"].values
        resids = resid_df[feat].values
        mask   = ~(np.isnan(times) | np.isnan(resids))
        if mask.sum() < 5:
            test_rows.append({
                "feature": feat,
                "rho": np.nan, "p_value": np.nan,
                "ph_assumption": "insufficient data",
            })
            continue

        rho, p = stats.spearmanr(times[mask], resids[mask])
        test_rows.append({
            "feature":       feat,
            "rho":           round(rho, 4),
            "p_value":       round(p,   4),
            "ph_assumption": "holds" if p > 0.05 else "VIOLATED",
        })

    test_df = pd.DataFrame(test_rows).sort_values("p_value")

    # Save CSV
    csv_path = output_dir / "schoenfeld_test.csv"
    test_df.to_csv(csv_path, index=False)

    # Print summary
    print("\n[Schoenfeld Test] Proportional Hazards Assumption")
    print("-" * 55)
    print(test_df.to_string(index=False))
    n_violated = (test_df["ph_assumption"] == "VIOLATED").sum()
    print(f"\n  {n_violated}/{len(test_df)} features violate PH (p < 0.05)")
    print(f"[Schoenfeld] Table saved to {csv_path}")

    # ── Residuals plot ────────────────────────────────────────────────────
    n_feats = len(features)
    n_cols  = min(3, n_feats)
    n_rows  = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes_flat[i]

        if feat not in resid_df.columns:
            ax.set_visible(False)
            continue

        times  = resid_df["event_time"].values
        resids = resid_df[feat].values
        mask   = ~(np.isnan(times) | np.isnan(resids))

        row    = test_df[test_df["feature"] == feat]
        p_val  = row["p_value"].values[0] if len(row) > 0 else np.nan
        rho    = row["rho"].values[0]     if len(row) > 0 else np.nan
        violated = (not np.isnan(p_val)) and (p_val < 0.05)
        color  = "crimson" if violated else "steelblue"

        ax.scatter(times[mask], resids[mask],
                   alpha=0.5, s=18, color=color, zorder=3)
        ax.axhline(0, linestyle="--", color="grey", linewidth=0.8)

        # Lowess-style smooth trend using running mean
        if mask.sum() >= 8:
            sorted_idx = np.argsort(times[mask])
            t_sorted   = times[mask][sorted_idx]
            r_sorted   = resids[mask][sorted_idx]
            window     = max(5, mask.sum() // 5)
            smooth     = pd.Series(r_sorted).rolling(
                window, center=True, min_periods=3).mean().values
            ax.plot(t_sorted, smooth, color=color, linewidth=1.8, zorder=4)

        tag = "VIOLATED" if violated else "holds"
        p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "n/a"
        r_str = f"{rho:.2f}"   if not np.isnan(rho)   else "n/a"
        ax.set_title(f"{feat}\nrho={r_str}  p={p_str}  [{tag}]",
                     color=color, fontsize=8.5, fontweight="bold" if violated else "normal")
        ax.set_xlabel("Event time (days)", fontsize=8)
        ax.set_ylabel("Schoenfeld residual", fontsize=8)
        sns.despine(ax=ax)

    # Hide unused panels
    for i in range(n_feats, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "Schoenfeld Residuals — Proportional Hazards Assumption Test\n"
        "Red panels (p < 0.05): PH violated — HR is a time-average, interpret with caution",
        fontsize=10,
    )
    plt.tight_layout()
    plot_path = output_dir / "schoenfeld_residuals_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Schoenfeld] Plot saved to {plot_path}")

    return test_df
