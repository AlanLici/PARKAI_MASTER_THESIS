"""
plot_domain_importance.py
-------------------------
Two complementary visualisations of domain and feature importance:

1. Domain importance bar chart
   - One bar per domain, height = C-index from the domain multivariable model
   - Reference line at C=0.5 (random) and C=0.7 (acceptable discrimination)
   - Answers: "which clinical domain best discriminates fallers from non-fallers?"

2. Feature HR heatmap
   - Rows = top features per domain (from univariate screening), grouped by domain
   - Colour = univariate HR (after inversion so all HRs > 1 = risk-increasing)
   - Grey = non-significant (p >= 0.05)
   - Answers: "what does the feature importance landscape look like within each domain?"

Output
    Results/Cox/Domain_PW/domain_cindex_parkwest.png
    Results/Cox/Domain_PW/feature_hr_heatmap.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Domain/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

RESULTS_DIR = PROJECT_ROOT / "Results" / "Cox" / "Domain_PW"
OUTPUT_DIR  = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from domain_feature_lists import DOMAINS, INVERT_FEATURES

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
P_CUTOFF   = 0.05
TOP_N      = 8       # max features per domain shown in heatmap
MIN_CINDEX = 0.50    # reference line

# Consistent domain colours
DOMAIN_COLORS = {
    "Motor_Examination":        "#E63946",
    "ADL_Motor_Daily_Living":   "#F4A261",
    "Motor_Complications":      "#E9C46A",
    "Non_Motor_Symptoms":       "#2A9D8F",
    "Cognition":                "#457B9D",
    "Mood_Psychiatric":         "#9B72CF",
    "Sleep":                    "#6D6875",
    "Autonomic":                "#A8DADC",
    "Apathy_Fatigue":           "#F77F00",
    "Quality_of_Life":          "#52B788",
    "Comorbidities":            "#ADB5BD",
}

# Short display labels
DOMAIN_LABELS = {
    "Motor_Examination":        "Motor Exam",
    "ADL_Motor_Daily_Living":   "ADL",
    "Motor_Complications":      "Motor Comp.",
    "Non_Motor_Symptoms":       "Non-Motor",
    "Cognition":                "Cognition",
    "Mood_Psychiatric":         "Mood",
    "Sleep":                    "Sleep",
    "Autonomic":                "Autonomic",
    "Apathy_Fatigue":           "Apathy/Fatigue",
    "Quality_of_Life":          "Quality of Life",
    "Comorbidities":            "Comorbidities",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_domain_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "domain_multivariable_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run run_domain_multivariable.py first.\n{path}")
    return pd.read_csv(path)


def load_univariate_results() -> pd.DataFrame:
    path = RESULTS_DIR / "all_domains_univariate.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run run_domain_univariate.py first.\n{path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plot 1 — Domain C-index bar chart
# ---------------------------------------------------------------------------

def _cindex_ci(c: float, n_events: int, n_patients: int,
               z: float = 1.96) -> tuple[float, float]:
    """
    Approximate 95% CI for C-index using the Hanley-McNeil formula.
    n1 = events (positives), n2 = non-events (negatives).
    """
    n1 = max(n_events, 1)
    n2 = max(n_patients - n_events, 1)
    q1 = c / (2.0 - c)
    q2 = 2.0 * c ** 2 / (1.0 + c)
    var = (c * (1 - c) + (n1 - 1) * (q1 - c ** 2) + (n2 - 1) * (q2 - c ** 2)) / (n1 * n2)
    se  = np.sqrt(max(var, 0))
    return max(0.0, c - z * se), min(1.0, c + z * se)


def plot_domain_cindex(summary_df: pd.DataFrame) -> None:
    # Filter to domains with valid C-index and sort descending
    df = summary_df[summary_df["c_index"].notna()].copy()
    df["label"] = df["domain"].map(DOMAIN_LABELS).fillna(df["domain"])
    df["color"] = df["domain"].map(DOMAIN_COLORS).fillna("#ADB5BD")
    df = df.sort_values("c_index", ascending=True).reset_index(drop=True)

    # Compute CI bounds
    cis = [
        _cindex_ci(row["c_index"], row["n_events"], row["n_patients"])
        for _, row in df.iterrows()
    ]
    df["ci_lo"] = [ci[0] for ci in cis]
    df["ci_hi"] = [ci[1] for ci in cis]
    df["err_lo"] = df["c_index"] - df["ci_lo"]
    df["err_hi"] = df["ci_hi"] - df["c_index"]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(df))))

    bars = ax.barh(
        df["label"], df["c_index"],
        color=df["color"], edgecolor="white", linewidth=0.5,
        height=0.65,
    )

    # 95% CI error bars
    ax.errorbar(
        df["c_index"],
        np.arange(len(df)),
        xerr=[df["err_lo"].values, df["err_hi"].values],
        fmt="none",
        ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2,
        zorder=5,
    )

    # Reference lines
    ax.axvline(0.5, color="black",  linestyle="--", linewidth=0.9,
               label="C = 0.50 (random)")
    ax.axvline(0.7, color="dimgrey", linestyle=":",  linewidth=0.9,
               label="C = 0.70 (acceptable)")

    # Value labels on bars
    for _, row in df.iterrows():
        ax.text(row["ci_hi"] + 0.008, df.index[df["label"] == row["label"]][0],
                f"{row['c_index']:.3f}", va="center", ha="left", fontsize=8.5)

    ax.set_xlabel("Concordance Index (C-index)", fontsize=11)
    ax.set_title(
        "Domain Importance — Multivariable Cox C-index\n"
        "(how well each domain alone discriminates fallers from non-fallers)\n"
        "Error bars = 95% CI (Hanley-McNeil approximation)",
        fontsize=10,
    )
    ax.set_xlim(0.40, max(df["ci_hi"]) + 0.08)
    ax.legend(fontsize=8, loc="lower right")
    sns.despine(left=True)
    plt.tight_layout()

    out = OUTPUT_DIR / "domain_cindex_parkwest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 1] Domain C-index bar chart saved to {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Feature HR heatmap
# ---------------------------------------------------------------------------

def _apply_inversion_to_hr(row: pd.Series) -> float:
    """If feature should be inverted, return 1/HR so all HRs are risk-increasing."""
    feat = row["Feature"]
    hr   = row["HR"]
    if pd.isna(hr) or hr <= 0:
        return np.nan
    if feat in INVERT_FEATURES:
        return 1.0 / hr
    return hr


def plot_feature_heatmap(uni_df: pd.DataFrame) -> None:
    # Apply inversion to HRs
    uni_df = uni_df.copy()
    uni_df["HR_adj"] = uni_df.apply(_apply_inversion_to_hr, axis=1)

    domain_order = [d for d in DOMAINS if d != "Comorbidities"]

    # Build per-domain top-N lists (rows = rank 1..TOP_N, cols = domains)
    domain_features = {}   # domain -> list of (feature, HR_adj, sig)
    for domain in domain_order:
        domain_data = uni_df[
            (uni_df["domain"] == domain) &
            (uni_df["converged"] == True) &
            (uni_df["HR_adj"].notna())
        ].sort_values("p").head(TOP_N)

        entries = []
        for _, r in domain_data.iterrows():
            entries.append({
                "feature": r["Feature"],
                "HR_adj":  r["HR_adj"],
                "sig":     r["p"] < P_CUTOFF,
            })
        domain_features[domain] = entries

    # Filter to domains that have at least one entry
    active_domains = [d for d in domain_order if domain_features.get(d)]
    if not active_domains:
        print("[Plot 2] No data to plot.")
        return

    n_cols = len(active_domains)
    n_rows = TOP_N

    # Build 2-D arrays: HR values and significance mask
    hr_grid   = np.full((n_rows, n_cols), np.nan)
    sig_grid  = np.zeros((n_rows, n_cols), dtype=bool)
    label_grid = np.full((n_rows, n_cols), "", dtype=object)

    for col_i, domain in enumerate(active_domains):
        for row_i, entry in enumerate(domain_features[domain]):
            hr_grid[row_i, col_i]    = entry["HR_adj"]
            sig_grid[row_i, col_i]   = entry["sig"]
            label_grid[row_i, col_i] = entry["feature"]

    # Mask: grey out non-significant OR empty cells
    mask_grey = ~sig_grid | np.isnan(hr_grid)
    mask_col  = sig_grid & ~np.isnan(hr_grid)

    vmax = np.nanpercentile(hr_grid[mask_col], 95) if mask_col.any() else 2.0
    vmax = max(vmax, 2.0)

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.5), max(5, n_rows * 0.7)))

    # Coloured cells (significant)
    sns.heatmap(
        hr_grid,
        mask=mask_grey,
        ax=ax,
        cmap="YlOrRd",
        vmin=1.0, vmax=vmax,
        annot=label_grid,
        fmt="",
        annot_kws={"size": 7},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Hazard Ratio (adjusted direction)", "shrink": 0.6},
        xticklabels=[DOMAIN_LABELS.get(d, d) for d in active_domains],
        yticklabels=[f"Rank {i+1}" for i in range(n_rows)],
    )

    # Grey cells (non-significant or empty)
    sns.heatmap(
        hr_grid,
        mask=~mask_grey,
        ax=ax,
        cmap=matplotlib.colors.ListedColormap(["#E8E8E8"]),
        vmin=0, vmax=1,
        annot=label_grid,
        fmt="",
        annot_kws={"size": 6.5, "color": "#999999"},
        linewidths=0.5, linecolor="white",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    # Colour-code x-axis labels by domain
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(
        [DOMAIN_LABELS.get(d, d) for d in active_domains],
        rotation=35, ha="right", fontsize=9,
    )
    for tick, domain in zip(ax.get_xticklabels(), active_domains):
        tick.set_color(DOMAIN_COLORS.get(domain, "#333333"))
        tick.set_fontweight("bold")

    ax.set_ylabel("Feature rank within domain (by univariate p-value)", fontsize=9)
    ax.set_title(
        f"Feature HR Heatmap — Top {TOP_N} features per domain (univariate Cox)\n"
        f"Cell text = feature name  |  Colour = HR (higher = stronger risk)\n"
        f"Grey = non-significant (p >= {P_CUTOFF})  |  Empty = fewer than {TOP_N} features",
        fontsize=9, pad=12,
    )

    sig_patch   = mpatches.Patch(facecolor="#D95F02", label="Significant (p < 0.05)")
    nosig_patch = mpatches.Patch(facecolor="#E8E8E8", label="Non-significant / empty")
    ax.legend(handles=[sig_patch, nosig_patch],
              loc="upper right", bbox_to_anchor=(1.0, -0.18),
              fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out = OUTPUT_DIR / "feature_hr_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 2] Feature HR heatmap saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("DOMAIN & FEATURE IMPORTANCE VISUALISATION")
    print("=" * 60)

    summary_df = load_domain_summary()
    uni_df     = load_univariate_results()

    plot_domain_cindex(summary_df)
    plot_feature_heatmap(uni_df)

    print("\n[Done] Both plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
