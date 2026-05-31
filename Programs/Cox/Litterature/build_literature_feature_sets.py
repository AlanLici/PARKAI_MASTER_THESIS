"""
build_literature_feature_sets.py
--------------------------------
Build three literature-feature sets of sizes 20, 15, 10 by:
    1. Ranking the literature features by univariate Cox p-value
       (reusing results/univariate_results.csv if present, otherwise
       running the univariate screen).
    2. Taking the top-N as the initial set.
    3. Iteratively computing VIF; if any feature has VIF >= 6, drop the
       worst one and pull in the next-by-p-value feature from the pool.
       Repeat until either all VIF < 6 or the pool is empty.

For each size, three files are written under
Results/Cox/Litterature_PW/litterature_sets/:
    top<N>_features.csv          — feature table (ranked by selection order;
                                    columns: Feature, HR, p, VIF, etc.)
    vif_replacement_log_top<N>.txt — per-iteration VIF + swaps
    feature_sets.txt             — combined human-readable summary (all sizes)

Usage:
    python Programs/Cox/Litterature/build_literature_feature_sets.py
"""

import sys
import importlib.util
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Litterature/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
COMMON_DIR   = SCRIPT_DIR.parent / "Common"                 # .../Programs/Cox/Common/
sys.path.insert(0, str(COMMON_DIR))

# Shared modules
spec_uni = importlib.util.spec_from_file_location(
    "uni", COMMON_DIR / "univariate_cox_model_screening.py")
uni = importlib.util.module_from_spec(spec_uni)
spec_uni.loader.exec_module(uni)

spec_multi = importlib.util.spec_from_file_location(
    "multi", COMMON_DIR / "multivariable_cox_feature_importance.py")
multi = importlib.util.module_from_spec(spec_multi)
spec_multi.loader.exec_module(multi)

from domain_feature_lists import apply_feature_inversions  # noqa: E402

# Reuse LITERATURE_FEATURES + add_computed_features + validate_features from the
# original literature pipeline so this script stays in sync with that file.
spec_lit = importlib.util.spec_from_file_location(
    "lit_orig", SCRIPT_DIR / "run_pipeline.py")
lit = importlib.util.module_from_spec(spec_lit)
spec_lit.loader.exec_module(lit)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TARGET_SIZES   = [20, 15, 10]
VIF_THRESHOLD  = 6.0
MAX_COEF       = 3.0   # quasi-separation guard (matches select_top_features)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Litterature_PW" / "litterature_sets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# VIF-with-replacement
# ---------------------------------------------------------------------------
def build_set_with_vif_replacement(
    df_wide: pd.DataFrame,
    ranked_features: list[str],
    target_size: int,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Top-N by p-value, then iteratively swap out the worst-VIF feature
    for the next-best by p-value until all VIF < threshold or pool is empty.

    Returns
    -------
    (final_features, final_vif_df, log_lines)
    """
    initial = list(ranked_features[:target_size])
    pool    = list(ranked_features[target_size:])

    current = list(initial)
    log = [
        f"VIF REPLACEMENT LOG  -  target size = {target_size}",
        "=" * 60,
        "",
        f"Initial top-{target_size} (by p-value):",
        *(f"  {i:2d}. {f}" for i, f in enumerate(initial, 1)),
        "",
        f"Replacement pool ({len(pool)}): {pool}",
        "",
    ]
    iteration = 0

    while len(current) > 1:
        long_df = uni.build_long_multi_feature(df_wide, current)
        if long_df.empty or long_df["event"].sum() < 5:
            log.append(f"[iter {iteration}] insufficient events -- stopping")
            break

        if "PIGD" in long_df.columns and "PIGD" in current:
            long_df["PIGD"] = 4 - long_df["PIGD"]
        long_df, _ = apply_feature_inversions(long_df, current)

        present = [f for f in current if f in long_df.columns]
        vif_df = multi.compute_vif(long_df, present)
        max_vif = vif_df["VIF"].iloc[0]
        worst   = vif_df["Feature"].iloc[0]

        vif_str = ", ".join(f"{r['Feature']}={r['VIF']:.2f}"
                            for _, r in vif_df.iterrows())
        log.append(f"[iter {iteration}] VIF: {vif_str}")

        if max_vif < VIF_THRESHOLD:
            log.append(f"[iter {iteration}] all VIF < {VIF_THRESHOLD} -- done")
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
            log.append(
                f"[iter {iteration}] removed {worst} (VIF={max_vif:.2f}), "
                f"replaced with {replacement}")
        else:
            log.append(
                f"[iter {iteration}] removed {worst} (VIF={max_vif:.2f}), "
                f"pool empty -- continuing with {len(current)} features")
        iteration += 1

    # Final rebuild
    final_long = uni.build_long_multi_feature(df_wide, current)
    if "PIGD" in final_long.columns and "PIGD" in current:
        final_long["PIGD"] = 4 - final_long["PIGD"]
    final_long, _ = apply_feature_inversions(final_long, current)
    present_final = [f for f in current if f in final_long.columns]
    final_vif = (multi.compute_vif(final_long, present_final)
                 if len(present_final) > 1 else
                 pd.DataFrame({"Feature": present_final, "VIF": [1.0] * len(present_final)}))

    log.append("")
    log.append(f"Final features ({len(current)}):")
    for i, f in enumerate(current, 1):
        log.append(f"  {i:2d}. {f}")
    return current, final_vif, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("BUILD LITERATURE FEATURE SETS  (VIF-refined top 20 / 15 / 10)")
    print("=" * 72)

    # ── Load + preprocess (mirrors litterature pipeline) ─────────────────
    df = uni.load_and_filter(uni.DATA_PATH)
    df = uni.compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    df = lit.add_computed_features(df)
    print(f"[Cohort] {len(df)} patients with valid time > 0.")

    available, missing = lit.validate_features(df, lit.LITERATURE_FEATURES)
    if missing:
        print(f"[Features] Missing from data: {missing}")
    screened = uni.apply_v9_guard(df, available)
    v9_excluded = [f for f in available if f not in screened]
    if v9_excluded:
        print(f"[V9 Guard] Excluded: {v9_excluded}")
    print(f"[Features] {len(screened)} features survived V9 guard.")

    # ── Reuse existing screening results if available ────────────────────
    cached_csv = PROJECT_ROOT / "Results" / "Cox" / "Litterature_PW" / "univariate_results.csv"
    if cached_csv.exists():
        print(f"[Screening] Reusing cached results: {cached_csv}")
        screening = pd.read_csv(cached_csv)
        screening = screening[screening["Feature"].isin(screened)].copy()
    else:
        print("[Screening] Running fresh univariate screen ...")
        screening = uni.run_univariate_screening(df, screened)

    # Apply the same selection guards as select_top_features:
    converged = screening[screening["converged"] == True].copy()
    plausible = converged[
        (converged["coef"].abs() <= MAX_COEF) &
        (converged["p"].notna())
    ].sort_values("p").reset_index(drop=True)
    ranked = plausible["Feature"].tolist()
    print(f"[Ranking] {len(ranked)} converged + |coef|<={MAX_COEF} + p-known features.")

    if len(ranked) < min(TARGET_SIZES):
        raise ValueError(
            f"Only {len(ranked)} features pass screening; cannot build sets "
            f"of size {min(TARGET_SIZES)}+."
        )

    # ── Build sets ────────────────────────────────────────────────────────
    summary = [
        "=" * 72,
        "LITERATURE FEATURE SETS  -  VIF-REFINED TOP 20 / 15 / 10",
        "=" * 72,
        "",
        f"Initial pool : {len(ranked)} literature features (V9-passed, "
        f"converged, |coef|<={MAX_COEF}, ranked by univariate p-value)",
        f"Selection    : top-N by p-value, then iterative VIF replacement",
        f"VIF threshold: {VIF_THRESHOLD}  (drop highest-VIF feature, swap in "
        f"next-best from pool, repeat until all VIF < threshold)",
        "",
    ]

    for size in TARGET_SIZES:
        if len(ranked) < size:
            print(f"\n[Size {size}] Skipping: only {len(ranked)} features.")
            summary.append(f"Size {size}: skipped (only {len(ranked)} features available)")
            continue

        print(f"\n{'-'*72}")
        print(f"BUILDING TOP-{size}")
        print(f"{'-'*72}")
        current, final_vif_df, log = build_set_with_vif_replacement(
            df, ranked, size)

        # Build feature table: p, HR, n, VIF, ranked by selection order.
        meta = plausible[plausible["Feature"].isin(current)].copy()
        order_map = {f: i for i, f in enumerate(current)}
        meta["selection_rank"] = meta["Feature"].map(order_map)
        meta = meta.sort_values("selection_rank").reset_index(drop=True)

        meta = meta.merge(final_vif_df, on="Feature", how="left")
        keep_cols = [c for c in
                     ["selection_rank", "Feature", "HR", "HR_lower_95",
                      "HR_upper_95", "p", "z", "n_patients", "n_events",
                      "n_intervals", "VIF"]
                     if c in meta.columns]
        meta = meta[keep_cols]

        out_csv = OUTPUT_DIR / f"top{size}_features.csv"
        meta.to_csv(out_csv, index=False)
        log_path = OUTPUT_DIR / f"vif_replacement_log_top{size}.txt"
        log_path.write_text("\n".join(log) + "\n", encoding="utf-8")

        print(f"  Final features ({len(current)}):")
        for i, feat in enumerate(current, 1):
            row = meta[meta["Feature"] == feat].iloc[0]
            p   = row["p"]
            hr  = row["HR"]
            vif = row.get("VIF", float("nan"))
            print(f"    {i:2d}. {feat:<28s}  HR={hr:.3f}  p={p:.4e}  VIF={vif:.2f}")
        print(f"  Saved {out_csv.name}, {log_path.name}")

        # Summary section
        summary.append("-" * 72)
        summary.append(f"TOP-{size}  (final n = {len(current)})")
        summary.append("-" * 72)
        for i, feat in enumerate(current, 1):
            row = meta[meta["Feature"] == feat].iloc[0]
            p   = row["p"]
            hr  = row["HR"]
            vif = row.get("VIF", float("nan"))
            p_str = f"p={p:.4e}" if pd.notna(p) else "p=n/a"
            vif_str = f"VIF={vif:.2f}" if pd.notna(vif) else "VIF=n/a"
            summary.append(f"  {i:2d}. {feat:<28s}  HR={hr:.3f}  {p_str}  {vif_str}")
        summary.append("")

    summary.append("=" * 72)
    summary_path = OUTPUT_DIR / "feature_sets.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"\n[Done] Outputs saved in {OUTPUT_DIR}")
    print(f"  feature_sets.txt")
    for size in TARGET_SIZES:
        print(f"  top{size}_features.csv  +  vif_replacement_log_top{size}.txt")


if __name__ == "__main__":
    main()
