"""
ppmi_data_loader.py
-------------------
Loads, merges, filters, and reshapes PPMI data into the counting-process
long format expected by the Cox time-varying model pipeline.

Pipeline
--------
1. Load all relevant PPMI CSV files (headers + data, no individual rows printed)
2. Filter to PD cohort (COHORT = 1)
3. Age-filter: keep patients with enrollment age >= AGE_CUTOFF so that the
   selected subsample has mean baseline age ~= ParkWest (68.1 yrs)
4. Merge all feature files on PATNO + EVENT_ID
5. Compute PIGD composite score
6. Compute GDS_TOT (sum of 15 GDS items with standard scoring)
7. Define fall event: FLNFR1W == 1 (from V04 onwards only)
8. Build counting-process long format (start, stop, event, features)
   using LOCF within each patient

Output
------
load_ppmi()           -> merged wide-ish DataFrame (one row per patient-visit)
build_ppmi_long_format(df, features) -> counting-process DataFrame
                                        columns: id, start, stop, event, <features>

Constants used by downstream scripts
-------------------------------------
PPMI_DATA_DIR  : path to raw PPMI CSVs
TARGET_MEAN_AGE: ParkWest baseline mean age (68.1)
AGE_CUTOFF     : enrollment age threshold that achieves TARGET_MEAN_AGE
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent              # .../Common/
PROJECT_ROOT  = SCRIPT_DIR.parents[2]                        # .../Alan_Sondre/
PPMI_DATA_DIR = PROJECT_ROOT / "Data" / "PPMI" / "Raw"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TARGET_MEAN_AGE = 68.1   # ParkWest baseline mean age to match

# ── Data quality settings (matching ParkWest pipeline) ────────────────────
MISSINGNESS_THRESHOLD = 0.58  # patient needs data at >= 58% of attended visits
MAX_FIRST_VISIT = "V08"       # feature must have first measurement by this visit
# Visits where the Determination_of_Freezing_and_Falls questionnaire
# (FLNFR12M / FLNFR1W) is administered. BL is included so that pre-study
# falls are captured for baseline-faller exclusion.
FALL_VISITS = {"BL","V04","V06","V08","V10","V12","V13","V14","V15",
               "V16","V17","V18","V19","V20","V21","V22","R01","R04","R06","R14"}

# Ordered visit sequence for sorting (approximate chronological order)
VISIT_ORDER = [
    "SC", "BL",
    "V01","V02","V03","V04","V05","V06","V07","V08","V09","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
    "R01","R04","R06","R14",
]


# ---------------------------------------------------------------------------
# Step 1 — Load individual files (headers + data merged; no rows printed)
# ---------------------------------------------------------------------------

def _load(fname: str, **kwargs) -> pd.DataFrame:
    path = PPMI_DATA_DIR / fname
    if not path.exists():
        print(f"  [MISSING] {fname}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_ppmi_raw() -> pd.DataFrame:
    """
    Load and merge all PPMI feature files into a single long-format DataFrame
    (one row per PATNO x EVENT_ID).  Only PATNO, EVENT_ID, and feature
    columns are kept — no REC_ID, PAG_NAME, ORIG_ENTRY, LAST_UPDATE, etc.
    """
    drop_cols = {"REC_ID", "PAG_NAME", "ORIG_ENTRY", "LAST_UPDATE",
                 "NUPSOURC", "PTCGBOTH", "INFODT"}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("[Load] Reading PPMI source files...")

    # --- Participant status (cohort + enrollment age) ----------------------
    status = _load("Participant_Status_13Jan2026.csv",
                   usecols=["PATNO","COHORT","ENROLL_AGE"])

    # --- Age at visit -------------------------------------------------------
    age_visit = _load("Age_at_visit_13Jan2026.csv")   # PATNO, EVENT_ID, AGE_AT_VISIT
    # Source file sometimes has multiple rows per PATNO+EVENT_ID with
    # slightly different AGE_AT_VISIT (visit conducted across multiple days).
    # Collapse to a single row per visit using the earliest age so the
    # merge backbone has exactly one row per patient-visit.
    if not age_visit.empty:
        age_visit = (age_visit
                     .sort_values(["PATNO","EVENT_ID","AGE_AT_VISIT"])
                     .drop_duplicates(subset=["PATNO","EVENT_ID"], keep="first"))

    # --- Demographics (sex) -------------------------------------------------
    demo = _load("Demographics_13Jan2026.csv",
                 usecols=["PATNO","EVENT_ID","SEX"])
    # Keep one sex entry per patient (taken at SC or earliest available)
    demo = demo.sort_values("EVENT_ID").drop_duplicates("PATNO")[["PATNO","SEX"]]

    # --- UPDRS Part I (rater) -----------------------------------------------
    up1r = clean(_load("MDS-UPDRS_Part_I_13Jan2026.csv"))

    # --- UPDRS Part I (patient questionnaire) -------------------------------
    up1p = clean(_load("MDS-UPDRS_Part_I_Patient_Questionnaire_13Jan2026.csv"))

    # --- UPDRS Part II (patient questionnaire) ------------------------------
    up2  = clean(_load("MDS_UPDRS_Part_II__Patient_Questionnaire_13Jan2026.csv"))

    # --- UPDRS Part III -----------------------------------------------------
    up3_full = clean(_load("MDS-UPDRS_Part_III_13Jan2026.csv"))
    # Select one row per PATNO+EVENT_ID, preferring:
    #   1. ON-state   (PDSTATE == 2)  — matches ParkWest's ON-state convention,
    #                                    preserves cross-cohort comparability
    #   2. Not applicable / NaN       — typically untreated early-stage patients
    #                                    (unmedicated, but no formal ON/OFF state)
    #   3. OFF-state  (PDSTATE == 1)  — last resort; unmedicated by washout
    # Without this, the dedup below would randomly keep whichever row appeared
    # first, silently mixing ON- and OFF-state measurements across patients.
    if "PDSTATE" in up3_full.columns:
        # Assign priority: ON=0 (best, matches ParkWest), NaN/untreated=1, OFF=2 (worst)
        def _pdstate_priority(s):
            # PDSTATE stores strings "OFF"/"ON" (not integers 1/2 as docs suggest)
            return s.map({"ON": 0, 2: 0, "OFF": 2, 1: 2}).fillna(1).astype(int)

        up3_full = up3_full.copy()
        up3_full["_priority"] = _pdstate_priority(up3_full["PDSTATE"])
        up3_full = (
            up3_full
            .sort_values(["PATNO", "EVENT_ID", "_priority"])
            .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
            .drop(columns="_priority")
        )
        vc = up3_full["PDSTATE"].value_counts(dropna=False)
        n_off = vc.get("OFF", vc.get(1, 0))
        n_on  = vc.get("ON",  vc.get(2, 0))
        n_nan = up3_full["PDSTATE"].isna().sum()
        print(f"[UPDRS III] After state selection: "
              f"OFF={n_off}, untreated/NaN={n_nan}, ON={n_on} "
              f"(total {len(up3_full)} rows)")
    up3 = up3_full.copy()

    # --- UPDRS Part IV ------------------------------------------------------
    up4  = clean(_load("MDS-UPDRS_Part_IV__Motor_Complications_13Jan2026.csv"))

    # --- Schwab & England ---------------------------------------------------
    schwab = clean(_load(
        "Modified_Schwab___England_Activities_of_Daily_Living_13Jan2026.csv"))

    # --- MoCA ---------------------------------------------------------------
    moca = clean(_load("Montreal_Cognitive_Assessment__MoCA__13Jan2026.csv"))

    # --- GDS ----------------------------------------------------------------
    gds  = clean(_load(
        "Geriatric_Depression_Scale__Short_Version__13Jan2026.csv"))

    # --- PDSS-2 online ------------------------------------------------------
    pdss = _load(
        "Parkinson_s_Disease_Sleep_Scale__PDSS-2___Online__13Jan2026.csv")
    pdss = pdss.drop(columns=[c for c in
                               ["MODIFIED_AT","CREATED_AT","RESPONSE_STATUS",
                                "SURVEY_VERSION"] if c in pdss.columns])

    # --- Epworth ------------------------------------------------------------
    epworth = clean(_load("Epworth_Sleepiness_Scale_13Jan2026.csv"))

    # --- SCOPA-AUT ----------------------------------------------------------
    scopa = clean(_load("SCOPA-AUT_13Jan2026.csv"))

    # --- Falls --------------------------------------------------------------
    falls_cols = ["PATNO","EVENT_ID","FLNFR1W","FLNFR12M",
                  "FRZGT1W","FLLDRVIS","FLLERVIS","FLLHOSP"]
    falls = _load("Determination_of_Freezing_and_Falls_13Jan2026.csv",
                  usecols=falls_cols)

    # ── Merge all on PATNO + EVENT_ID ──────────────────────────────────────
    print("[Load] Merging files...")
    key = ["PATNO","EVENT_ID"]

    # Start from age_visit as the backbone (every patient-visit with age)
    df = age_visit.copy()

    for src in [up1r, up1p, up2, up3, up4, schwab, moca, gds, pdss,
                epworth, scopa, falls]:
        if src.empty:
            continue
        # Some files may have duplicate PATNO+EVENT_ID (e.g. ON/OFF exams)
        # Keep first occurrence per patient-visit to avoid row explosion
        src_dedup = src.drop_duplicates(subset=key, keep="first")
        df = df.merge(src_dedup, on=key, how="left")

    # Merge sex (patient-level, no EVENT_ID)
    df = df.merge(demo, on="PATNO", how="left")

    # Replace PPMI sentinel value 101 ("UR" / unable to rate) with NaN.
    # Applied to UPDRS items (NP1*, NP2*, NP3*, NP4*) which are scored 0-4.
    # Without this, items like NP3PSTBL=101 silently inflate PIGD and any
    # UPDRS-based feature used downstream. NHY (Hoehn & Yahr) is treated
    # separately since it can validly take values 0-5.
    updrs_cols = [c for c in df.columns if c.startswith(("NP1","NP2","NP3","NP4"))]
    n_replaced = 0
    for c in updrs_cols:
        mask = df[c] == 101
        n_replaced += int(mask.sum())
        df.loc[mask, c] = np.nan
    if "NHY" in df.columns:
        mask = df["NHY"] == 101
        n_replaced += int(mask.sum())
        df.loc[mask, "NHY"] = np.nan
    if n_replaced:
        print(f"[Sentinel] Replaced {n_replaced} '101' (UR/unable-to-rate) "
              f"values with NaN across {len(updrs_cols)+1} UPDRS/NHY columns")

    print(f"[Load] Raw merged shape: {df.shape} "
          f"({df['PATNO'].nunique()} patients, {df['EVENT_ID'].nunique()} unique visits)")
    return df, status


# ---------------------------------------------------------------------------
# Step 2 — Filter cohort + age matching
# ---------------------------------------------------------------------------

def filter_cohort(df: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """Keep PD patients (COHORT=1) and apply age cutoff to match ParkWest."""

    pd_patnos = set(status.loc[status["COHORT"] == 1, "PATNO"])
    df = df[df["PATNO"].isin(pd_patnos)].copy()
    print(f"[Filter] PD cohort: {df['PATNO'].nunique()} patients")

    # Merge enrollment age
    enroll_age = status[["PATNO","ENROLL_AGE"]].drop_duplicates()
    df = df.merge(enroll_age, on="PATNO", how="left")

    # Find the enrollment age cutoff that gives mean ~= TARGET_MEAN_AGE
    # We do this empirically: try thresholds and pick the one minimising |mean - target|
    thresholds = np.arange(55, 75, 0.5)
    base_ages  = df.drop_duplicates("PATNO")["ENROLL_AGE"].dropna()

    best_cut, best_diff, best_n = 0, 999, 0
    for t in thresholds:
        sub  = base_ages[base_ages >= t]
        if len(sub) < 50:
            break
        diff = abs(sub.mean() - TARGET_MEAN_AGE)
        if diff < best_diff:
            best_diff, best_cut, best_n = diff, t, len(sub)

    print(f"[AgeFilter] Cutoff >= {best_cut:.1f} yrs  "
          f"-> {best_n} patients  "
          f"-> mean enroll age = {base_ages[base_ages >= best_cut].mean():.1f} "
          f"(target {TARGET_MEAN_AGE})")

    keep = set(df.loc[df["ENROLL_AGE"] >= best_cut, "PATNO"])
    df   = df[df["PATNO"].isin(keep)].copy()
    print(f"[AgeFilter] Retained {df['PATNO'].nunique()} patients, "
          f"{len(df)} visit-rows")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Derived features
# ---------------------------------------------------------------------------

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PIGD score and GDS total."""

    # PIGD: mean of (NP2WALK, NP2FREZ, NP3GAIT, NP3FRZGT, NP3PSTBL)
    # Standard PPMI PIGD formula (Stebbins et al.). All 5 items must be
    # present; otherwise PIGD is NaN for that row. Partial means are not
    # clinically valid and can produce misleading values.
    pigd_items = ["NP2WALK","NP2FREZ","NP3GAIT","NP3FRZGT","NP3PSTBL"]
    available  = [c for c in pigd_items if c in df.columns]
    if len(available) == 5:
        df["PIGD"] = df[available].mean(axis=1, skipna=False)
        n_valid = int(df["PIGD"].notna().sum())
        print(f"[PIGD] Computed from all 5 items "
              f"({n_valid} rows with complete items)")
    else:
        print(f"[PIGD] Insufficient columns: {available}")

    # GDS total: sum 15 items, standard scoring
    # Depressed answers: for negative items (GDSDROPD etc) 1=yes=depressed
    # For positive items (GDSSATIS, GDSGSPIR, GDSHAPPY, GDSALIVE, GDSENRGY)
    # 0=yes=NOT depressed, so depressed answer = 1 - item
    positive_items = {"GDSSATIS","GDSGSPIR","GDSHAPPY","GDSALIVE","GDSENRGY"}
    gds_items = ["GDSSATIS","GDSDROPD","GDSEMPTY","GDSBORED","GDSGSPIR",
                 "GDSAFRAD","GDSHAPPY","GDSHLPLS","GDSHOME","GDSMEMRY",
                 "GDSALIVE","GDSWRTLS","GDSENRGY","GDSHOPLS","GDSBETER"]
    gds_available = [c for c in gds_items if c in df.columns]
    if gds_available:
        scored = df[gds_available].copy()
        for col in gds_available:
            if col in positive_items:
                scored[col] = 1 - scored[col]   # flip: 0=depressed answer
        df["GDS_TOT"] = scored.sum(axis=1, skipna=False)
        print(f"[GDS] GDS_TOT computed from {len(gds_available)} items")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Fall event definition
# ---------------------------------------------------------------------------

def add_fall_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event definition matching ParkWest: NP2WALK >= 1 OR NP2FREZ >= 3.

    ParkWest uses UPO13 >= 1 (falling) OR UPO14 >= 3 (freezing).
    MDS-UPDRS equivalents:
        UPO13 (falling)  -> NP2WALK (walking & balance, Part II)
        UPO14 (freezing) -> NP2FREZ (freezing of gait, Part II)

    event = 1 if NP2WALK >= 1 or NP2FREZ >= 3 (same thresholds as PW).
    event = 0 if both items are known and below threshold.
    event = NaN if both items are missing at that visit.
    """
    df["event"] = np.nan
    has_walk = df.get("NP2WALK") is not None
    has_frez = df.get("NP2FREZ") is not None

    if has_walk or has_frez:
        walk_ok = df["NP2WALK"].notna() if has_walk else pd.Series(False, index=df.index)
        frez_ok = df["NP2FREZ"].notna() if has_frez else pd.Series(False, index=df.index)
        known = walk_ok | frez_ok

        walk_event = (df["NP2WALK"] >= 1) if has_walk else pd.Series(False, index=df.index)
        frez_event = (df["NP2FREZ"] >= 3) if has_frez else pd.Series(False, index=df.index)

        df.loc[known, "event"] = (walk_event | frez_event).astype(int)
        df.loc[~known, "event"] = np.nan

    n_events = int((df["event"] == 1).sum())
    n_nonevents = int((df["event"] == 0).sum())
    n_unknown = int(df["event"].isna().sum())
    print(f"[Event] NP2WALK>=1 or NP2FREZ>=3: "
          f"{n_events} event / {n_nonevents} non-event / {n_unknown} unknown rows")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Build counting-process long format
# ---------------------------------------------------------------------------

def _visit_rank(event_id: str) -> int:
    try:
        return VISIT_ORDER.index(event_id)
    except ValueError:
        return len(VISIT_ORDER)


def build_ppmi_long_format(df: pd.DataFrame,
                            features: list[str]) -> pd.DataFrame:
    """
    Convert the merged PPMI visit-level DataFrame into counting-process
    (start, stop, event) format suitable for CoxTimeVaryingFitter.

    For each patient:
      - Sort visits by age_at_visit
      - Apply LOCF for missing feature values (using ALL visits, so that
        feature updates propagate even through visits where falls weren't
        assessed)
      - Drop visits with unknown fall status (event is NaN) so that only
        visits with a confirmed fall outcome serve as interval endpoints
      - For each consecutive pair of kept visits (i, i+1):
          start = age_days[i]
          stop  = age_days[i+1]
          event = fall indicator AT visit i+1
          covariates = feature values AT visit i (pre-interval values)

    Only intervals where stop > start are kept.
    Follow-up is truncated at the first event interval (mirrors ParkWest).

    Parameters
    ----------
    df       : merged patient-visit DataFrame from load_ppmi()
    features : list of feature column names to include

    Returns
    -------
    pd.DataFrame with columns: id, start, stop, event, <features>
    """
    # Dedup features list: the same feature can appear in several DOMAINS
    # (e.g. NP1APAT in Mood and Apathy), which would otherwise create
    # duplicate columns in `sub` and turn scalar lookups into 2-row Series.
    seen: set[str] = set()
    uniq_features = [f for f in features
                     if f in df.columns and not (f in seen or seen.add(f))]
    keep_cols = ["PATNO","EVENT_ID","AGE_AT_VISIT","event"] + uniq_features
    sub = df[keep_cols].copy()

    # Convert age to days (relative to patient minimum age in dataset)
    sub["age_days"] = sub["AGE_AT_VISIT"] * 365.25

    # Sort visits per patient
    sub["_rank"] = sub["EVENT_ID"].apply(_visit_rank)
    sub = sub.sort_values(["PATNO","_rank","AGE_AT_VISIT"])

    # ── DATA QUALITY FILTER 1: Visit guard ────────────────────────────────
    # Remove features whose first non-null observation is after MAX_FIRST_VISIT.
    # Mirrors ParkWest's V9 guard.
    max_rank = (_visit_rank(MAX_FIRST_VISIT)
                if MAX_FIRST_VISIT in VISIT_ORDER else len(VISIT_ORDER))
    filtered_features = []
    for f in uniq_features:
        feat_data = sub.loc[sub[f].notna(), "_rank"]
        if feat_data.empty:
            continue
        if feat_data.min() <= max_rank:
            filtered_features.append(f)
        else:
            print(f"  [Visit Guard] Excluding '{f}' "
                  f"(first data at {VISIT_ORDER[int(feat_data.min())]}, "
                  f"cutoff {MAX_FIRST_VISIT})")
    uniq_features = filtered_features

    sub = sub.drop(columns="_rank")

    if not uniq_features:
        return pd.DataFrame()

    # ── DATA QUALITY FILTER 2: Per-patient missingness ────────────────────
    # For each patient-feature pair, compute fraction of attended visits
    # with non-null data. Set feature to NaN for patients below threshold.
    for f in uniq_features:
        coverage = (sub.groupby("PATNO")[f]
                    .transform(lambda s: s.notna().sum() / len(s)))
        sub.loc[coverage < MISSINGNESS_THRESHOLD, f] = np.nan

    # ── DATA QUALITY FILTER 3: Median fill for leading NaN ────────────────
    # Before LOCF, fill leading NaN (before first real measurement) with
    # the cross-sectional median of that feature. Prevents patients from
    # being dropped just because their first visit(s) are missing.
    feat_cols = [f for f in uniq_features if f in sub.columns]
    for f in feat_cols:
        median_val = sub[f].median()
        if pd.isna(median_val):
            continue
        # Identify leading NaN per patient: rows before the first valid index
        first_valid = sub.groupby("PATNO")[f].transform(
            lambda s: s.first_valid_index()
        )
        # Rows where the feature is NaN AND index < first valid index
        # (i.e., leading NaN before any real measurement)
        leading_mask = sub[f].isna() & (sub.index < first_valid)
        if leading_mask.any():
            sub.loc[leading_mask, f] = median_val

    # ── LOCF per patient ──────────────────────────────────────────────────
    # Done on ALL visits so feature updates propagate even through visits
    # where falls weren't assessed.
    for col in feat_cols:
        sub[col] = sub.groupby("PATNO")[col].ffill()

    # Keep only rows with a known fall outcome as interval endpoints.
    # Rows with event=NaN (questionnaire not administered/answered) are dropped
    # so they don't get silently counted as non-events.
    sub = sub[sub["event"].notna()].copy()
    sub["event"] = sub["event"].astype(int)

    # Exclude prevalent (baseline) fallers: if a patient's first known-outcome
    # visit already reports a fall (event == 1, i.e. NP2WALK >= 1 or NP2FREZ >= 3),
    # they had the event before follow-up began and must be dropped. Mirrors
    # ParkWest's baseline-faller exclusion. Without this, prevalent and incident
    # fallers get pooled and hazard ratios are biased.
    first_rows = (sub.sort_values(["PATNO", "age_days"])
                     .drop_duplicates("PATNO", keep="first"))
    baseline_fallers = set(first_rows.loc[first_rows["event"] == 1, "PATNO"])
    if baseline_fallers:
        print(f"[BaselineFallers] Excluding {len(baseline_fallers)} patients "
              f"with a fall recorded at their first known-outcome visit")
        sub = sub[~sub["PATNO"].isin(baseline_fallers)].copy()

    intervals = []
    feat_present = [f for f in uniq_features if f in sub.columns]

    for patno, grp in sub.groupby("PATNO"):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < 2:
            continue

        # Compute patient-level time offset (first visit = day 0)
        t0 = grp["age_days"].iloc[0]

        for i in range(n - 1):
            start = grp["age_days"].iloc[i]   - t0
            stop  = grp["age_days"].iloc[i+1] - t0
            if stop <= start:
                continue

            ev = int(grp["event"].iloc[i + 1])

            row = {"id": patno, "start": start, "stop": stop, "event": ev}
            for f in feat_present:
                row[f] = grp[f].iloc[i]   # covariate value at START of interval
            intervals.append(row)

            # Truncate at first event — mirrors ParkWest behaviour where
            # follow-up ends at the visit where a fall is first recorded.
            if ev == 1:
                break

    long_df = pd.DataFrame(intervals)
    if long_df.empty:
        return long_df

    # Drop rows where all features are NaN (patient had no usable data)
    long_df = long_df.dropna(subset=feat_present, how="all")

    n_pat    = long_df["id"].nunique()
    n_events = int(long_df["event"].sum())
    print(f"[LongFormat] {n_pat} patients | {n_events} events | "
          f"{len(long_df)} intervals")
    return long_df


# ---------------------------------------------------------------------------
# Step 6 — Discover dynamic features (appear in >= 2 visits per patient)
# ---------------------------------------------------------------------------

def discover_dynamic_features(df: pd.DataFrame,
                               candidate_features: list[str],
                               min_visits: int = 2) -> list[str]:
    """
    Return features from candidate_features that are non-null in >= min_visits
    visits for at least some patients.  Mirrors ParkWest's discover_dynamic_features.
    """
    dynamic = []
    for feat in candidate_features:
        if feat not in df.columns:
            continue
        # Count visits with non-null values per patient, take max
        max_obs = df.groupby("PATNO")[feat].count().max()
        if max_obs >= min_visits:
            dynamic.append(feat)
    print(f"[Dynamic] {len(dynamic)}/{len(candidate_features)} features "
          f"have >= {min_visits} observations in at least one patient")
    return dynamic


# ---------------------------------------------------------------------------
# Main convenience function
# ---------------------------------------------------------------------------

def load_ppmi(verbose: bool = True) -> pd.DataFrame:
    """
    Full load pipeline: raw -> cohort filter -> age filter -> derived features
    -> fall events.  Returns the merged visit-level DataFrame ready for
    build_ppmi_long_format().
    """
    df_raw, status = load_ppmi_raw()
    df = filter_cohort(df_raw, status)
    df = compute_derived_features(df)
    df = add_fall_event(df)

    if verbose:
        n_pat     = df["PATNO"].nunique()
        n_events  = int((df["event"] == 1).sum())
        n_nonfall = int((df["event"] == 0).sum())
        n_unknown = int(df["event"].isna().sum())
        print(f"\n[PPMI Ready] {n_pat} patients | "
              f"{n_events} fall / {n_nonfall} non-fall / "
              f"{n_unknown} unknown visit rows")

    return df


# ---------------------------------------------------------------------------
# Run as script — print summary stats (no individual rows)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_ppmi()
    print("\n[Summary]")
    print(f"  Patients : {df['PATNO'].nunique()}")
    print(f"  Visits   : {len(df)}")
    print(f"  Fall events: {int((df['event'] == 1).sum())}")
    print(f"  Visits with falls data: "
          f"{df[df['EVENT_ID'].isin(FALL_VISITS)]['PATNO'].nunique()} patients")
    print(f"  Mean age at enroll: {df.drop_duplicates('PATNO')['ENROLL_AGE'].mean():.1f}")
