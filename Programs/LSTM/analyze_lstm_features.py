import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


DATA_PATH = (
    r"../../Data/ParkVest_ClinicalData_with_Metadata.xlsx"
)

# ── Visit prefixes ─────────────────────────────────────────────────────────
MODEL_VISIT_PREFIXES = [
    "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
]

# ── LSTM literature feature set (30 features) ─────────────────────────────
FEATURES = [
    # Motor
    "PIGD", "UPO_part2_tot", "UPO_part3_tot", "UPO27", "UPO15", "UPO_part4_tot",
    # Cognition
    "STROOPW", "MMSE_tot", "IQCODE_mean", "STROOPC", "MCI_15SD_MDScriteria",
    # Mood / Psychiatric
    "madrs10", "madrs_tot", "npi8hi",
    # Sleep
    "pdss02", "pdss_tot", "epwo6",
    # Autonomic
    "URINDYS", "CONSTIPAT", "PressStand_Dia", "SEON",
    # Fatigue / Apathy
    "fss2", "fss_sum", "fss5",
    # Quality of Life
    "sf36pf", "sf36pcs", "PHYFUN",
    # Demographics / Time (computed or static)
    "HYON",
]

# Note: age_at_visit and disease_duration_years are computed features in the
# LSTM pipeline, not raw columns. They are excluded here since they don't
# exist as visit-level columns in the raw data.


def resolve_column(df: pd.DataFrame, visit: str, feature: str) -> str | None:
    """Resolve actual column name for a (visit, feature) pair."""
    candidates = [f"{visit}_{feature}", f"{visit}_{feature}_ON"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_aggregate_features(features_to_analyze: list[str]) -> pd.DataFrame:
    """Load data and aggregate each feature by averaging across all visits."""
    print("[Loading] Reading Excel file...")
    df = pd.read_excel(DATA_PATH)

    # Remove controls
    df = df[~df["BL_CASE"].astype(str).str.contains("K", case=False, na=False)].copy()
    print(f"[Loaded] {len(df)} PD patients\n")

    aggregated_data = pd.DataFrame()

    for feature in features_to_analyze:
        # Static baseline features
        if feature.startswith("BL_") or feature == "Time_PDfirst_BL":
            if feature in df.columns:
                aggregated_data[feature] = df[feature]
                print(f"  + {feature:25} [static baseline feature]")
            else:
                print(f"  - {feature:25} [NOT FOUND]")
        else:
            # Dynamic features: collect across visits and average
            visit_columns = []
            for visit in MODEL_VISIT_PREFIXES:
                col = resolve_column(df, visit, feature)
                if col is not None:
                    visit_columns.append(col)

            if visit_columns:
                aggregated_data[feature] = df[visit_columns].mean(axis=1)
                print(f"  + {feature:25} [averaged across {len(visit_columns)} visits]")
            else:
                print(f"  - {feature:25} [NOT FOUND in any visit]")

    aggregated_data = aggregated_data.dropna()
    print(f"\n[Prepared] {len(aggregated_data)} complete records\n")

    return aggregated_data


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for each feature."""
    vif_list = []

    for i, col in enumerate(data.columns):
        y = data.iloc[:, i]
        X = data.drop(columns=[col])

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf

        vif_list.append({
            "Feature": col,
            "VIF": vif,
            "R_squared": r_squared,
        })

    return pd.DataFrame(vif_list).sort_values("VIF", ascending=False)


def rank_features_by_collinearity(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Rank features by average absolute correlation."""
    avg_corr = corr_matrix.abs().mean() - 1 / len(corr_matrix)
    avg_corr = avg_corr.sort_values(ascending=False)

    return pd.DataFrame({
        "Feature": avg_corr.index,
        "Avg Absolute Correlation": avg_corr.values,
    }).reset_index(drop=True)


def find_problematic_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Identify feature pairs with |correlation| >= threshold."""
    pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                pairs.append({
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": corr_val,
                })

    if pairs:
        return pd.DataFrame(pairs).sort_values("Correlation", ascending=False, key=abs)
    return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path):
    """Generate and save correlation heatmap."""
    n = len(corr_matrix)
    figsize = max(12, n * 0.45)
    plt.figure(figsize=(figsize, figsize * 0.85))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
        square=True,
    )
    plt.title(
        f"Feature Correlation Matrix — LSTM Literature Set ({n} features)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Plot saved] {output_path}")


def main():
    features_to_analyze = FEATURES

    print("\n" + "=" * 70)
    print(f"COLLINEARITY ANALYSIS — LSTM LITERATURE FEATURES ({len(features_to_analyze)})")
    print("=" * 70 + "\n")

    # Load and aggregate
    data = load_and_aggregate_features(features_to_analyze)

    # Compute
    print("[Computing] Correlation matrix and VIF...\n")
    corr_matrix = data.corr()
    vif_data = calculate_vif(data)
    collinearity_rank = rank_features_by_collinearity(corr_matrix)
    problematic_pairs = find_problematic_pairs(corr_matrix, threshold=0.7)

    # Print results
    print("=" * 70)
    print("1. RANKING BY COLLINEARITY (Most -> Least Correlated)")
    print("=" * 70)
    print(collinearity_rank.to_string(index=False))

    print("\n" + "=" * 70)
    print("2. VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 70)
    print("   VIF < 5 = acceptable | VIF 5-10 = moderate | VIF > 10 = severe")
    print()
    vif_display = vif_data.copy()
    vif_display["VIF"] = vif_display["VIF"].apply(lambda x: f"{x:.2f}" if x != np.inf else "inf")
    vif_display["R^2"] = vif_display["R_squared"].apply(lambda x: f"{x:.3f}")
    print(vif_display[["Feature", "VIF", "R^2"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("3. PROBLEMATIC PAIRS (|r| >= 0.7)")
    print("=" * 70)
    if not problematic_pairs.empty:
        print(problematic_pairs.to_string(index=False))
    else:
        print("No highly correlated pairs found.")

    print("\n" + "=" * 70)
    print("4. FULL CORRELATION MATRIX")
    print("=" * 70)
    print(corr_matrix.to_string())

    # Save outputs
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    collinearity_rank.to_csv(output_dir / "feature_ranking.csv", index=False)
    vif_for_csv = vif_data.copy()
    vif_for_csv["VIF"] = vif_for_csv["VIF"].apply(lambda x: 999999 if x == np.inf else x)
    vif_for_csv.to_csv(output_dir / "vif_results.csv", index=False)
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    if not problematic_pairs.empty:
        problematic_pairs.to_csv(output_dir / "problematic_pairs.csv", index=False)

    plot_correlation_heatmap(corr_matrix, output_dir / "correlation_heatmap.png")

    print("\n" + "=" * 70)
    print(f"[Results saved] {output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
