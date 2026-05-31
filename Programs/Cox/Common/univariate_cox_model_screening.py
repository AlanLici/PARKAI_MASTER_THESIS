"""
Univariate Cox Screening — Test Every Feature Individually
===========================================================

This module defines the data-loading, event/time calculation, and long-format
conversion shared across the Cox pipelines, and fits a **separate univariate**
Cox time-varying model for *each* dynamic feature found in the dataset. The
pipeline scripts under Programs/Cox/<pipeline>/ import these functions directly.

Workflow
--------
1.  Load data, compute events/times (same as cox_feat_hr.py).
2.  Auto-discover all dynamic features by scanning column names for visit
    prefixes (V4_, V5_, …).  A feature is "dynamic" if it appears under at
    least 2 visit prefixes.
3.  For each discovered feature, build a one-column long-format dataset,
    standardise, fit CoxTimeVaryingFitter, and record HR / p-value.
4.  Rank all features by p-value (and HR) and save the full table to CSV.

Author : Sondre Lyngstad
Date   : 2026-03-12
"""

### =========================================================================
### 1. IMPORTS AND CONFIGURATION
### =========================================================================

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../Common/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/

DATA_PATH = (
    PROJECT_ROOT / "Data" / "ParkWest" / "Raw"
    / "ParkVest_ClinicalData_with_Metadata.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cox" / "Univariate_Screening"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_CSV       = OUTPUT_DIR / "univariate_results.csv"

# ── Visit prefixes ─────────────────────────────────────────────────────────
ALL_VISIT_PREFIXES = [
    "BL", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
]
MODEL_VISIT_PREFIXES = [
    "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
]

# ── BL inclusion flag ─────────────────────────────────────────────────────
# When True, BL is treated as an additional feature timepoint:
#   - feature discovery counts BL toward the >=2-visits dynamic check
#   - get_feature_schedule / coverage / V9 guard see BL as a scheduled visit
#   - the long-format builder seeds prev_vals with each patient's BL value
#     before the V4..V21 loop, so BL feature values LOCF into the first
#     interval. BL is NOT added to event detection (compute_event_time still
#     skips BL for UPO13/UPO14 checks because of the medication artifact).
# Pipelines flip this to True via `uni.INCLUDE_BL = True` after import.
INCLUDE_BL = False


def _active_visit_prefixes() -> list[str]:
    """Visit prefixes treated as feature timepoints (BL prepended if INCLUDE_BL)."""
    return ["BL"] + MODEL_VISIT_PREFIXES if INCLUDE_BL else list(MODEL_VISIT_PREFIXES)


# ── Event thresholds (same as literature model) ───────────────────────────
UPO13_THRESHOLD = 1
UPO14_THRESHOLD = 3

# ── Screening settings ────────────────────────────────────────────────────
MIN_VISITS_FOR_FEATURE = 2   # feature must appear under >= 2 visit prefixes
P_VALUE_CUTOFF         = 0.05  # for selecting top features
TOP_N_FEATURES         = 13   # max features for multivariate model (match EPV)

# ── Data quality settings ────────────────────────────────────────────────
MISSINGNESS_THRESHOLD  = 0.58  # patient must have data at >= 58% of attended scheduled visits
MAX_FIRST_VISIT        = "V9"  # feature must have first measurement by this visit

# ── Features to EXCLUDE from screening ─────────────────────────────────────
# Event-definition columns (would be circular to use as predictors)
# DateVisit, CASE, and administrative/identifier columns
EXCLUDE_PATTERNS = [
    r"^UPO13$",          # part of event definition
    r"^UPO14$",          # part of event definition
    r"DateVisit",        # not a clinical feature
    r"CASE$",            # patient identifier
    r"^Date",            # date columns
]


### =========================================================================
### 2. DATA LOADING & EVENT/TIME CALCULATION
### =========================================================================

def load_and_filter(path: str) -> pd.DataFrame:
    """Load Excel data and remove control patients (BL_CASE contains 'K')."""
    df = pd.read_excel(path)
    n_total = len(df)
    df = df[~df["BL_CASE"].astype(str).str.contains("K", case=False, na=False)].copy()
    n_pd = len(df)
    print(f"[Preprocessing] Loaded {n_total} rows; kept {n_pd} PD patients.")
    return df


def _resolve_column(df: pd.DataFrame, visit: str, feature: str) -> str | None:
    """Return actual column name for (visit, feature), trying _ON suffix."""
    candidates = [f"{visit}_{feature}", f"{visit}_{feature}_ON"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# SPSS stores datetimes as seconds since 1582-10-14 (Gregorian epoch).
# Some Vn_DateVisit columns (V15, V16, V18, V19, V20) in the ParkWest Excel
# come from an SPSS export where the datetime conversion was lost — they
# read as raw floats around 1.3e10–1.4e10 instead of Timestamps.
SPSS_DATE_OFFSET_SEC = 12_219_379_200  # seconds from 1582-10-14 to 1970-01-01
_SPSS_VALUE_MIN = 1e10  # ~1899; clinical visits will land above this
_SPSS_VALUE_MAX = 1.5e10  # ~2058


def _parse_date_value(val):
    """Parse one date cell, handling Excel datetimes AND SPSS-encoded floats."""
    if isinstance(val, (int, float)) and not pd.isna(val) \
            and _SPSS_VALUE_MIN < val < _SPSS_VALUE_MAX:
        return pd.Timestamp(val - SPSS_DATE_OFFSET_SEC, unit="s")
    return pd.to_datetime(val, dayfirst=True, errors="coerce")


def _parse_date(series: pd.Series) -> pd.Series:
    return series.apply(_parse_date_value)


def compute_event_time(df: pd.DataFrame) -> pd.DataFrame:
    """Determine whether/when each patient had a fall event (from V4 onward)."""
    bl_dates = _parse_date(df["BL_DateVisit"])
    events, event_visits, time_days = [], [], []

    for idx, row in df.iterrows():
        bl_date = bl_dates.loc[idx]
        fell, fell_visit, last_valid_date = False, None, bl_date

        for visit in ALL_VISIT_PREFIXES:
            date_col = f"{visit}_DateVisit"
            if date_col not in df.columns:
                continue
            visit_date = _parse_date_value(row.get(date_col))
            if pd.isna(visit_date):
                continue
            last_valid_date = visit_date

            if visit == "BL":
                continue

            upo13_col = _resolve_column(df, visit, "UPO13")
            upo13_val = np.nan
            if upo13_col is not None:
                upo13_val = pd.to_numeric(row.get(upo13_col), errors="coerce")

            upo14_col = _resolve_column(df, visit, "UPO14")
            upo14_val = np.nan
            if upo14_col is not None:
                upo14_val = pd.to_numeric(row.get(upo14_col), errors="coerce")

            meets_upo13 = (not np.isnan(upo13_val)) and upo13_val >= UPO13_THRESHOLD
            meets_upo14 = (not np.isnan(upo14_val)) and upo14_val >= UPO14_THRESHOLD
            if meets_upo13 or meets_upo14:
                fell, fell_visit = True, visit
                break

        if pd.isna(bl_date):
            dt = np.nan
        elif fell and fell_visit:
            ev_date = _parse_date_value(row.get(f"{fell_visit}_DateVisit"))
            dt = (ev_date - bl_date).days if pd.notna(ev_date) else np.nan
        else:
            dt = (last_valid_date - bl_date).days if pd.notna(last_valid_date) else np.nan

        events.append(int(fell))
        event_visits.append(fell_visit)
        time_days.append(dt)

    df = df.copy()
    df["event"], df["event_visit"], df["time_days"] = events, event_visits, time_days
    print(f"[Event/Time] {sum(events)} patients experienced a fall event.")
    return df


### =========================================================================
### 3. AUTO-DISCOVER DYNAMIC FEATURES
### =========================================================================

def _should_exclude(feature_name: str) -> bool:
    """Check if a feature should be excluded from screening."""
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, feature_name, re.IGNORECASE):
            return True
    return False


def discover_dynamic_features(df: pd.DataFrame) -> list[str]:
    """Scan all column names to find features that appear under multiple visits.

    A column like V4_HYON, V5_HYON, V7_HYON → base feature "HYON".
    A column like V4_UPO_part2_tot_ON → base feature "UPO_part2_tot" (strip _ON).

    Returns a sorted list of unique base feature names that appear under at
    least MIN_VISITS_FOR_FEATURE visit prefixes.
    """
    # Build regex to match visit-prefixed columns
    # Prefixes: V4_, V5_, ..., V21_ (we use MODEL_VISIT_PREFIXES)
    feature_visit_counts: dict[str, set[str]] = {}

    active_prefixes = _active_visit_prefixes()
    for col in df.columns:
        for visit in active_prefixes:
            prefix = f"{visit}_"
            if col.startswith(prefix):
                base = col[len(prefix):]
                # Strip trailing _ON suffix
                if base.endswith("_ON"):
                    base = base[:-3]
                if base and not _should_exclude(base):
                    if base not in feature_visit_counts:
                        feature_visit_counts[base] = set()
                    feature_visit_counts[base].add(visit)
                break  # matched this visit, don't try longer prefixes

    # Keep features appearing in >= MIN_VISITS_FOR_FEATURE visits
    dynamic_features = sorted(
        feat for feat, visits in feature_visit_counts.items()
        if len(visits) >= MIN_VISITS_FOR_FEATURE
    )

    print(f"[Discovery] Found {len(dynamic_features)} dynamic features "
          f"(appear in >= {MIN_VISITS_FOR_FEATURE} visits).")
    return dynamic_features


### =========================================================================
### 4. DATA QUALITY HELPERS
### =========================================================================

def get_feature_schedule(df: pd.DataFrame, feature: str) -> list[str]:
    """Return the visits where a feature structurally exists (has a column with data)."""
    schedule = []
    for v in _active_visit_prefixes():
        col = _resolve_column(df, v, feature)
        if col is not None and df[col].notna().sum() > 0:
            schedule.append(v)
    return schedule


def compute_visit_medians(df: pd.DataFrame, features: list[str]) -> dict:
    """Compute cross-sectional median for each feature at each visit."""
    medians: dict[str, dict[str, float]] = {}
    for feat in features:
        medians[feat] = {}
        for v in _active_visit_prefixes():
            col = _resolve_column(df, v, feat)
            if col is not None:
                vals = pd.to_numeric(df[col], errors="coerce")
                med = vals.median()
                if not np.isnan(med):
                    medians[feat][v] = med
    return medians


def _patient_feature_coverage(row, df: pd.DataFrame, feature: str,
                              schedule: list[str]) -> float:
    """Fraction of attended scheduled visits where the patient has real data."""
    attended = 0
    has_data = 0
    for v in schedule:
        date_col = f"{v}_DateVisit"
        if date_col not in df.columns:
            continue
        visit_date = _parse_date_value(row.get(date_col))
        if pd.isna(visit_date):
            continue
        attended += 1
        col = _resolve_column(df, v, feature)
        if col is not None:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if not np.isnan(val):
                has_data += 1
    return has_data / attended if attended > 0 else 0.0


def apply_v9_guard(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Remove features whose first measurement is after MAX_FIRST_VISIT."""
    prefixes = _active_visit_prefixes()
    max_idx = prefixes.index(MAX_FIRST_VISIT) if MAX_FIRST_VISIT in prefixes else len(prefixes)
    passed = []
    for feat in features:
        schedule = get_feature_schedule(df, feat)
        if not schedule:
            continue
        first_idx = prefixes.index(schedule[0]) if schedule[0] in prefixes else len(prefixes)
        if first_idx > max_idx:
            print(f"  [V9 Guard] Excluding {feat} (starts at {schedule[0]})")
            continue
        passed.append(feat)
    return passed


### =========================================================================
### 5. BUILD LONG FORMAT FOR A SINGLE FEATURE
### =========================================================================

def build_long_single_feature(df: pd.DataFrame,
                              feature: str) -> pd.DataFrame | None:
    """Build counting-process long-format data for ONE dynamic feature.

    Applies V9 guard, 58% missingness threshold, and median fill for leading NaN.
    Returns None if the feature produces too few usable intervals.
    """
    # V9 guard
    schedule = get_feature_schedule(df, feature)
    if not schedule:
        return None
    prefixes = _active_visit_prefixes()
    max_idx = prefixes.index(MAX_FIRST_VISIT) if MAX_FIRST_VISIT in prefixes else len(prefixes)
    if prefixes.index(schedule[0]) > max_idx:
        return None

    # Pre-compute median for leading NaN fill (from first scheduled visit)
    visit_medians = compute_visit_medians(df, [feature])
    feat_medians = visit_medians.get(feature, {})
    first_sched = schedule[0]
    leading_fill = feat_medians.get(first_sched, np.nan)

    records: list[dict] = []

    for idx, row in df.iterrows():
        patient_id = row["BL_CASE"]
        bl_date = _parse_date_value(row.get("BL_DateVisit"))
        if pd.isna(bl_date):
            continue

        # Missingness check: patient coverage for this feature
        cov = _patient_feature_coverage(row, df, feature, schedule)
        if cov < MISSINGNESS_THRESHOLD:
            continue

        event_visit = row["event_visit"]
        patient_event = int(row["event"])

        prev_date = bl_date
        prev_val = np.nan

        # ── BL pre-pass ──────────────────────────────────────────────
        # Seed prev_val with the patient's BL value (if present), so it
        # LOCFs into the first V4..V21 interval. The BL iteration itself
        # doesn't create a record (zero-length BL->BL interval).
        if INCLUDE_BL:
            bl_col = _resolve_column(df, "BL", feature)
            if bl_col is not None:
                bl_val = pd.to_numeric(row.get(bl_col), errors="coerce")
                if not np.isnan(bl_val):
                    prev_val = bl_val

        for visit in MODEL_VISIT_PREFIXES:
            date_col = f"{visit}_DateVisit"
            if date_col not in df.columns:
                continue
            visit_date = _parse_date_value(row.get(date_col))
            if pd.isna(visit_date):
                continue

            start = (prev_date - bl_date).days
            stop = (visit_date - bl_date).days

            if stop < 0 or stop <= start:
                continue

            interval_event = 1 if (patient_event == 1 and event_visit == visit) else 0

            col = _resolve_column(df, visit, feature)
            val = np.nan
            if col is not None:
                val = pd.to_numeric(row.get(col), errors="coerce")
            # LOCF
            if np.isnan(val) and not np.isnan(prev_val):
                val = prev_val
            # Median fill for leading NaN
            if np.isnan(val) and not np.isnan(leading_fill):
                val = leading_fill
            if not np.isnan(val):
                prev_val = val

            records.append({
                "id": patient_id,
                "start": start,
                "stop": stop,
                "event": interval_event,
                feature: val,
            })

            prev_date = visit_date
            if interval_event == 1:
                break

    if not records:
        return None

    long_df = pd.DataFrame(records)
    long_df.dropna(subset=[feature], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    if long_df["event"].sum() < 5 or long_df["id"].nunique() < 10:
        return None

    return long_df


### =========================================================================
### 5. UNIVARIATE SCREENING
### =========================================================================

def run_univariate_screening(df: pd.DataFrame,
                             features: list[str]) -> pd.DataFrame:
    """Fit a separate Cox model for each feature and collect HR/p-value.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format data (after event/time computation, time-filtered).
    features : list[str]
        List of base feature names to test.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, coef, HR, HR_lower_95, HR_upper_95, SE, z, p,
                 n_patients, n_events, n_intervals, converged
    """
    results = []
    n_total = len(features)
    t0 = time.time()

    for i, feat in enumerate(features, 1):
        if i % 25 == 0 or i == 1 or i == n_total:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n_total - i) if i > 0 else 0
            print(f"  [{i}/{n_total}] Screening '{feat}' ... "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        long_df = build_long_single_feature(df, feat)
        if long_df is None:
            results.append({
                "Feature": feat, "coef": np.nan, "HR": np.nan,
                "HR_lower_95": np.nan, "HR_upper_95": np.nan,
                "SE": np.nan, "z": np.nan, "p": np.nan,
                "n_patients": 0, "n_events": 0, "n_intervals": 0,
                "converged": False, "note": "insufficient data",
            })
            continue

        # Standardise the single feature
        scaler = StandardScaler()
        long_df[[feat]] = scaler.fit_transform(long_df[[feat]])

        # Check for zero-variance (constant after standardisation)
        if long_df[feat].std() < 1e-10:
            results.append({
                "Feature": feat, "coef": np.nan, "HR": np.nan,
                "HR_lower_95": np.nan, "HR_upper_95": np.nan,
                "SE": np.nan, "z": np.nan, "p": np.nan,
                "n_patients": long_df["id"].nunique(),
                "n_events": int(long_df["event"].sum()),
                "n_intervals": len(long_df),
                "converged": False, "note": "zero variance",
            })
            continue

        # Fit univariate Cox model
        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(
                long_df[["id", "start", "stop", "event", feat]],
                id_col="id", event_col="event",
                start_col="start", stop_col="stop",
                show_progress=False,
            )
            s = ctv.summary
            coef = s["coef"].iloc[0]
            hr = np.exp(coef)
            se = s["se(coef)"].iloc[0]
            z = s["z"].iloc[0]
            p = s["p"].iloc[0]

            # CI
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
        except Exception as e:
            coef = hr = hr_lo = hr_hi = se = z = p = np.nan
            converged = False

        results.append({
            "Feature": feat,
            "coef": coef,
            "HR": hr,
            "HR_lower_95": hr_lo,
            "HR_upper_95": hr_hi,
            "SE": se,
            "z": z,
            "p": p,
            "n_patients": long_df["id"].nunique(),
            "n_events": int(long_df["event"].sum()),
            "n_intervals": len(long_df),
            "converged": converged,
            "note": "" if converged else "failed to converge",
        })

    results_df = pd.DataFrame(results)
    # Sort by p-value (significant first)
    results_df.sort_values("p", inplace=True, na_position="last")
    results_df.reset_index(drop=True, inplace=True)
    return results_df


### =========================================================================
### 6. FEATURE SELECTION & MULTIVARIATE LONG-FORMAT BUILDER
### =========================================================================

def select_top_features(screening_results: pd.DataFrame,
                        df_wide: pd.DataFrame,
                        p_cutoff: float = P_VALUE_CUTOFF,
                        max_features: int = TOP_N_FEATURES,
                        corr_threshold: float = 0.70) -> list[str]:
    """Select top features from univariate screening with deduplication.

    Selection criteria:
    1. p < p_cutoff in univariate model
    2. Model converged
    3. Ranked by p-value (lowest first)
    4. **Correlated features are deduplicated**: if a candidate is correlated
       (|r| > corr_threshold) with an already-selected feature, it is skipped.
       This prevents multicollinearity from collapsing the multivariate model.
    5. Capped at max_features

    Parameters
    ----------
    screening_results : pd.DataFrame
        Output of run_univariate_screening.
    df_wide : pd.DataFrame
        Wide-format data (for computing pairwise correlations via V4 columns).
    p_cutoff : float
        Significance threshold.
    max_features : int
        Maximum number of features to select.
    corr_threshold : float
        If |correlation| > this between a candidate and any already-selected
        feature, the candidate is skipped.
    """
    # Filter: converged, significant, and plausible coefficient
    # |coef| > 3 after standardisation indicates quasi-separation or
    # near-constant features producing artifact extreme HRs
    MAX_COEF = 3.0
    sig = screening_results[
        (screening_results["p"] < p_cutoff) &
        (screening_results["converged"] == True) &
        (screening_results["coef"].abs() <= MAX_COEF)
    ].copy()
    sig.sort_values("p", inplace=True)

    n_excluded = len(screening_results[
        (screening_results["p"] < p_cutoff) &
        (screening_results["converged"] == True) &
        (screening_results["coef"].abs() > MAX_COEF)
    ])
    if n_excluded > 0:
        print(f"[Selection] Excluded {n_excluded} features with |coef| > {MAX_COEF} "
              f"(quasi-separation artifacts).")

    print(f"\n[Selection] {len(sig)} features with p < {p_cutoff}.")
    print(f"[Selection] Applying correlation deduplication (|r| > {corr_threshold} -> skip).")

    # Build a quick correlation lookup using V4 columns (first model visit)
    # This avoids building full long-format for all candidates.
    def _get_v4_values(feat_name: str) -> pd.Series | None:
        """Get V4 values for a feature (for correlation computation)."""
        col = _resolve_column(df_wide, "V4", feat_name)
        if col is None:
            return None
        return pd.to_numeric(df_wide[col], errors="coerce")

    selected: list[str] = []
    skipped: list[tuple[str, str, float]] = []  # (feat, corr_with, r)

    for _, row in sig.iterrows():
        if len(selected) >= max_features:
            break

        candidate = row["Feature"]
        cand_vals = _get_v4_values(candidate)

        # If we can't compute correlation, accept the feature
        if cand_vals is None or cand_vals.isna().all():
            selected.append(candidate)
            continue

        # Check correlation with every already-selected feature
        too_correlated = False
        for sel_feat in selected:
            sel_vals = _get_v4_values(sel_feat)
            if sel_vals is None or sel_vals.isna().all():
                continue
            # Pairwise correlation (drop NaN pairs)
            mask = cand_vals.notna() & sel_vals.notna()
            if mask.sum() < 20:
                continue
            r = cand_vals[mask].corr(sel_vals[mask])
            if abs(r) > corr_threshold:
                too_correlated = True
                skipped.append((candidate, sel_feat, r))
                break

        if not too_correlated:
            selected.append(candidate)

    print(f"[Selection] Selected {len(selected)} non-redundant features "
          f"(skipped {len(skipped)} correlated duplicates).")
    if skipped:
        print(f"\n  Skipped features (correlated with already-selected):")
        for feat, corr_with, r in skipped[:20]:
            print(f"    {feat:<30s}  |r|={abs(r):.2f}  with {corr_with}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    return selected


def build_long_multi_feature(df: pd.DataFrame,
                             features: list[str]) -> pd.DataFrame:
    """Build counting-process long-format data for MULTIPLE dynamic features.

    Applies V9 guard, 58% missingness threshold per patient-feature,
    and median fill for leading NaN (before first real measurement).
    """
    # ── V9 guard ──────────────────────────────────────────────────────────
    features = apply_v9_guard(df, features)
    if not features:
        return pd.DataFrame()

    # ── Pre-compute schedules and medians ─────────────────────────────────
    schedules: dict[str, list[str]] = {}
    for feat in features:
        schedules[feat] = get_feature_schedule(df, feat)

    visit_medians = compute_visit_medians(df, features)
    leading_fills: dict[str, float] = {}
    for feat in features:
        sched = schedules[feat]
        if sched and sched[0] in visit_medians.get(feat, {}):
            leading_fills[feat] = visit_medians[feat][sched[0]]

    # ── Pre-compute per-patient-feature coverage ──────────────────────────
    patient_reliable: dict[tuple, set] = {}
    n_dropped_patients = 0
    for idx, row in df.iterrows():
        reliable = set()
        for feat in features:
            cov = _patient_feature_coverage(row, df, feat, schedules[feat])
            if cov >= MISSINGNESS_THRESHOLD:
                reliable.add(feat)
        patient_reliable[idx] = reliable

    # ── Build long format ─────────────────────────────────────────────────
    records: list[dict] = []

    for idx, row in df.iterrows():
        patient_id = row["BL_CASE"]
        bl_date = _parse_date_value(row.get("BL_DateVisit"))
        if pd.isna(bl_date):
            continue

        reliable = patient_reliable[idx]
        if not reliable:
            continue

        event_visit = row["event_visit"]
        patient_event = int(row["event"])

        prev_date = bl_date
        prev_vals: dict[str, float] = {}

        # ── BL pre-pass ──────────────────────────────────────────────
        # Seed prev_vals with each patient's BL feature values (if any)
        # so they LOCF into the first V4..V21 interval. BL itself does
        # not get its own record (zero-length BL->BL interval).
        if INCLUDE_BL:
            for feat in features:
                if feat not in reliable:
                    continue
                bl_col = _resolve_column(df, "BL", feat)
                if bl_col is None:
                    continue
                bl_val = pd.to_numeric(row.get(bl_col), errors="coerce")
                if not np.isnan(bl_val):
                    prev_vals[feat] = bl_val

        for visit in MODEL_VISIT_PREFIXES:
            date_col = f"{visit}_DateVisit"
            if date_col not in df.columns:
                continue
            visit_date = _parse_date_value(row.get(date_col))
            if pd.isna(visit_date):
                continue

            start = (prev_date - bl_date).days
            stop = (visit_date - bl_date).days

            if stop < 0 or stop <= start:
                continue

            interval_event = 1 if (patient_event == 1 and event_visit == visit) else 0

            feat_vals: dict[str, float] = {}
            for feat in features:
                if feat not in reliable:
                    feat_vals[feat] = np.nan
                    continue

                col = _resolve_column(df, visit, feat)
                val = np.nan
                if col is not None:
                    val = pd.to_numeric(row.get(col), errors="coerce")
                # LOCF
                if np.isnan(val) and feat in prev_vals:
                    val = prev_vals[feat]
                # Median fill for leading NaN
                if np.isnan(val) and feat in leading_fills:
                    val = leading_fills[feat]
                feat_vals[feat] = val
                if not np.isnan(val):
                    prev_vals[feat] = val

            record = {"id": patient_id, "start": start, "stop": stop,
                      "event": interval_event}
            record.update(feat_vals)
            records.append(record)

            prev_date = visit_date
            if interval_event == 1:
                break

    long_df = pd.DataFrame(records)
    long_df.dropna(subset=features, inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    long_df = long_df.groupby("id").filter(lambda g: len(g) >= 1)
    return long_df


def _significance_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


### =========================================================================
### MAIN
### =========================================================================

def main() -> None:
    print("=" * 72)
    print("UNIVARIATE COX SCREENING — ALL FEATURES")
    print("=" * 72)

    # ── Load & preprocess ─────────────────────────────────────────────
    df = load_and_filter(DATA_PATH)
    df = compute_event_time(df)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    print(f"[Time Filter] {len(df)} patients with valid time > 0.")

    # ── Discover all dynamic features ─────────────────────────────────
    all_features = discover_dynamic_features(df)
    print(f"\nFeatures to screen:")
    for i, f in enumerate(all_features, 1):
        print(f"  {i:3d}. {f}")

    # ── Run univariate screening ───────────────────────────────────────
    print(f"\n{'='*72}")
    print("RUNNING UNIVARIATE SCREENING")
    print(f"{'='*72}")
    t0 = time.time()
    screening = run_univariate_screening(df, all_features)
    elapsed = time.time() - t0
    print(f"\n[Screening] Completed in {elapsed:.1f} seconds.")

    # Ensure sorted from smallest to largest p-value before saving
    screening.sort_values("p", inplace=True, na_position="last")
    screening.reset_index(drop=True, inplace=True)

    # ── Save full screening results ───────────────────────────────────
    screening.to_csv(OUTPUT_CSV, index=False)
    print(f"[Output] Full univariate results saved to {OUTPUT_CSV}")

    # ── Show top 30 ───────────────────────────────────────────────────
    converged = screening[screening["converged"] == True].copy()
    n_sig = len(converged[converged["p"] < P_VALUE_CUTOFF])
    print(f"\n{'='*72}")
    print(f"TOP 30 FEATURES (by p-value) — {n_sig} total with p < {P_VALUE_CUTOFF}")
    print(f"{'='*72}")
    top30 = converged.head(30)[
        ["Feature", "HR", "HR_lower_95", "HR_upper_95", "p", "n_events"]
    ].copy()
    top30["Sig"] = top30["p"].apply(_significance_stars)
    print(top30.to_string(index=False))

    print(f"\n{'='*72}")
    print("[Done] Univariate screening complete.")
    print(f"  Results folder: {OUTPUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
