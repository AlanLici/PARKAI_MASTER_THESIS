"""
ppmi/cross_cohort_mapping.py
-----------------------------
Comprehensive ParkWest -> PPMI feature mapping used by the LSTM feature-set
exporter and (optionally) the PPMI Cox validation pipelines.

A single entry in MAP looks like:
    "<ParkWest_feature>": {
        "ppmi" : str | list[str] | None,   # PPMI column(s) or None to drop
        "kind" : "direct" | "derived" | "proxy" | "drop",
        "note" : str,                       # human-readable rationale
    }

  - direct  : 1:1 mapping (same construct, same scale family)
  - derived : ParkWest feature must be computed from multiple PPMI items (sum)
  - proxy   : conceptually close but instrument-different (flag in writeup)
  - drop    : no PPMI equivalent

translate(features) returns a list of dicts in input order. write_report
writes a human-readable txt + a flat unique PPMI feature list.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# UPDRS items: ParkWest UPO numbering matches the ORIGINAL UPDRS, not MDS-UPDRS.
# Mappings below were verified against the ParkWest Metadata sheet labels.
# ---------------------------------------------------------------------------
MAP: dict[str, dict] = {

    # ── UPDRS Part I (UPO1-4) → MDS-UPDRS Part I rater items ──────────────
    "UPO1":  {"ppmi": "NP1COG",  "kind": "direct", "note": "intellectual impairment → NP1COG (cognitive impairment)"},
    "UPO2":  {"ppmi": "NP1HALL", "kind": "proxy",  "note": "thought disorder → NP1HALL (hallucinations); closest MDS-UPDRS construct"},
    "UPO3":  {"ppmi": "NP1DPRS", "kind": "direct", "note": "depression → NP1DPRS"},
    "UPO4":  {"ppmi": "NP1APAT", "kind": "direct", "note": "motivation/initiative → NP1APAT (apathy)"},

    # ── UPDRS Part II (UPO5-17) → MDS-UPDRS Part II items ────────────────
    "UPO5":  {"ppmi": "NP2SPCH", "kind": "direct", "note": "speech ADL → NP2SPCH"},
    "UPO6":  {"ppmi": "NP2SALV", "kind": "direct", "note": "salivation ADL → NP2SALV"},
    "UPO7":  {"ppmi": "NP2SWAL", "kind": "direct", "note": "swallowing ADL → NP2SWAL"},
    "UPO8":  {"ppmi": "NP2HWRT", "kind": "direct", "note": "handwriting ADL → NP2HWRT"},
    "UPO9":  {"ppmi": "NP2EAT",  "kind": "direct", "note": "cutting food ADL → NP2EAT"},
    "UPO10": {"ppmi": "NP2DRES", "kind": "direct", "note": "dressing ADL → NP2DRES"},
    "UPO11": {"ppmi": "NP2HYGN", "kind": "direct", "note": "hygiene ADL → NP2HYGN"},
    "UPO12": {"ppmi": "NP2TURN", "kind": "direct", "note": "turning in bed ADL → NP2TURN"},
    "UPO13": {"ppmi": None,      "kind": "drop",   "note": "falling ADL — used as the fall event; cannot be a predictor"},
    "UPO14": {"ppmi": None,      "kind": "drop",   "note": "freezing ADL — used as the fall event; cannot be a predictor"},
    "UPO15": {"ppmi": None,      "kind": "drop",   "note": "walking ADL — closest MDS-UPDRS item NP2WALK is the event proxy; dropped"},
    "UPO16": {"ppmi": "NP2TRMR", "kind": "direct", "note": "tremor in extremities → NP2TRMR"},
    "UPO17": {"ppmi": None,      "kind": "drop",   "note": "sensory complaints — no MDS-UPDRS Part II equivalent"},

    # ── UPDRS Part III (UPO18-31) → MDS-UPDRS Part III items ─────────────
    "UPO18":  {"ppmi": "NP3SPCH",  "kind": "direct", "note": "speech (motor) → NP3SPCH"},
    "UPO19":  {"ppmi": "NP3FACXP", "kind": "direct", "note": "facial expression → NP3FACXP"},
    "UPO20a": {"ppmi": "NP3RTALJ", "kind": "direct", "note": "tremor at rest face/lips/chin → NP3RTALJ"},
    "UPO20b": {"ppmi": "NP3RTARU", "kind": "direct", "note": "tremor at rest right arm → NP3RTARU"},
    "UPO20c": {"ppmi": "NP3RTALU", "kind": "direct", "note": "tremor at rest left arm → NP3RTALU"},
    "UPO20d": {"ppmi": "NP3RTARL", "kind": "direct", "note": "tremor at rest right leg → NP3RTARL"},
    "UPO20e": {"ppmi": "NP3RTALL", "kind": "direct", "note": "tremor at rest left leg → NP3RTALL"},
    "UPO21a": {"ppmi": "NP3PTRMR", "kind": "direct", "note": "postural tremor right → NP3PTRMR"},
    "UPO21b": {"ppmi": "NP3PTRML", "kind": "direct", "note": "postural tremor left → NP3PTRML"},
    "UPO22a": {"ppmi": "NP3RIGN",  "kind": "direct", "note": "rigidity neck → NP3RIGN"},
    "UPO22b": {"ppmi": "NP3RIGRU", "kind": "direct", "note": "rigidity right arm → NP3RIGRU"},
    "UPO22c": {"ppmi": "NP3RIGLU", "kind": "direct", "note": "rigidity left arm → NP3RIGLU"},
    "UPO22d": {"ppmi": "NP3RIGRL", "kind": "direct", "note": "rigidity right leg → NP3RIGRL"},
    "UPO22e": {"ppmi": "NP3RIGLL", "kind": "direct", "note": "rigidity left leg → NP3RIGLL"},
    "UPO23a": {"ppmi": "NP3FTAPR", "kind": "direct", "note": "finger taps right → NP3FTAPR"},
    "UPO23b": {"ppmi": "NP3FTAPL", "kind": "direct", "note": "finger taps left → NP3FTAPL"},
    "UPO24a": {"ppmi": "NP3HMOVR", "kind": "direct", "note": "hand movements right → NP3HMOVR"},
    "UPO24b": {"ppmi": "NP3HMOVL", "kind": "direct", "note": "hand movements left → NP3HMOVL"},
    "UPO25a": {"ppmi": "NP3PRSPR", "kind": "direct", "note": "rapid alternating movements right → NP3PRSPR"},
    "UPO25b": {"ppmi": "NP3PRSPL", "kind": "direct", "note": "rapid alternating movements left → NP3PRSPL"},
    "UPO26a": {"ppmi": "NP3LGAGR", "kind": "direct", "note": "leg agility right → NP3LGAGR"},
    "UPO26b": {"ppmi": "NP3LGAGL", "kind": "direct", "note": "leg agility left → NP3LGAGL"},
    "UPO27":  {"ppmi": "NP3RISNG", "kind": "direct", "note": "arising from chair → NP3RISNG"},
    "UPO28":  {"ppmi": "NP3POSTR", "kind": "direct", "note": "posture → NP3POSTR"},
    "UPO29":  {"ppmi": "NP3GAIT",  "kind": "direct", "note": "gait → NP3GAIT"},
    "UPO30":  {"ppmi": "NP3PSTBL", "kind": "direct", "note": "postural stability → NP3PSTBL"},
    "UPO31":  {"ppmi": "NP3BRADY", "kind": "direct", "note": "body bradykinesia → NP3BRADY"},

    # ── UPDRS Part IV (UPO32-39) → MDS-UPDRS Part IV items ───────────────
    # NB: original UPDRS Part IV is more granular than MDS-UPDRS Part IV;
    #     several mappings are proxies, not 1:1.
    "UPO32": {"ppmi": "NP4WDYSK", "kind": "direct", "note": "dyskinesias duration → NP4WDYSK"},
    "UPO33": {"ppmi": "NP4DYSKI", "kind": "direct", "note": "dyskinesias disability → NP4DYSKI"},
    "UPO34": {"ppmi": None,       "kind": "drop",   "note": "painful dyskinesias — no clean MDS-UPDRS Part IV equivalent"},
    "UPO35": {"ppmi": "NP4DYSTN", "kind": "proxy",  "note": "early-morning dystonia → NP4DYSTN (painful off-state dystonia)"},
    "UPO36": {"ppmi": "NP4FLCTI", "kind": "proxy",  "note": "predictable off-periods → NP4FLCTI (complexity of fluctuations, internal)"},
    "UPO37": {"ppmi": "NP4FLCTX", "kind": "proxy",  "note": "unpredictable off-periods → NP4FLCTX (complexity of fluctuations, external)"},
    "UPO38": {"ppmi": None,       "kind": "drop",   "note": "sudden off-periods — merged into NP4FLCTX in MDS-UPDRS; not separately scored"},
    "UPO39": {"ppmi": "NP4OFF",   "kind": "direct", "note": "proportion of off-time per day → NP4OFF"},

    # ── UPDRS totals (ParkWest computed) ──────────────────────────────────
    "UPO_part1_tot":    {"ppmi": ["NP1RTOT", "NP1PTOT"], "kind": "derived", "note": "MDS-UPDRS Part I total = NP1RTOT + NP1PTOT"},
    "UPO_part2_tot":    {"ppmi": "NP2PTOT", "kind": "direct", "note": "UPDRS Part II total → NP2PTOT"},
    "UPO_part2_ON_tot": {"ppmi": "NP2PTOT", "kind": "direct", "note": "UPDRS Part II total (ON) → NP2PTOT (PPMI Part II total)"},
    "UPO_part3_tot":    {"ppmi": "NP3TOT",  "kind": "direct", "note": "UPDRS Part III total → NP3TOT"},
    "UPO_part4_tot":    {"ppmi": "NP4TOT",  "kind": "direct", "note": "UPDRS Part IV total → NP4TOT"},

    # ── UPN (alternative ParkWest UPDRS variant) ──────────────────────────
    # If UPN items appear they typically mirror UPO items; same mappings apply.
    "UPN_part1_tot":    {"ppmi": ["NP1RTOT", "NP1PTOT"], "kind": "derived", "note": "Mirrors UPO_part1_tot"},
    "UPN2.12":          {"ppmi": "NP2TURN", "kind": "direct", "note": "UPN turning in bed → NP2TURN"},
    "UPN2.13":          {"ppmi": None, "kind": "drop", "note": "UPN walking (event proxy) — dropped"},
    "UPN4.3":           {"ppmi": "NP4DYSKI", "kind": "proxy", "note": "UPN Part IV item — mapped to NP4DYSKI"},

    # ── PIGD composite ────────────────────────────────────────────────────
    "PIGD":         {"ppmi": "PIGD", "kind": "direct", "note": "PIGD computed in both cohorts (different items, same construct)"},
    "PIGD_nevner":  {"ppmi": None, "kind": "drop", "note": "ParkWest-internal denominator (count of PIGD items); no PPMI equivalent"},
    "PIGD_YN":      {"ppmi": None, "kind": "drop", "note": "ParkWest-internal yes/no flag for PIGD subtype; no PPMI equivalent"},

    # ── Hoehn & Yahr, Schwab & England ────────────────────────────────────
    "HYON":         {"ppmi": "NHY",     "kind": "direct", "note": "Hoehn & Yahr (ON state) → NHY"},
    "SEON":         {"ppmi": "MSEADLG", "kind": "direct", "note": "Schwab & England (ON) → MSEADLG (Modified S&E)"},

    # ── Cognition (ParkWest uses Stroop, MMSE, IQCODE; PPMI has MoCA) ─────
    "STROOPW":          {"ppmi": "MCATOT", "kind": "proxy", "note": "Stroop word → MoCA total (proxy; different cognitive test, same domain)"},
    "STROOPC":          {"ppmi": None,     "kind": "drop",  "note": "Stroop colour — no PPMI equivalent (MCATOT already used for STROOPW)"},
    "STROOPCW":         {"ppmi": None,     "kind": "drop",  "note": "Stroop colour-word — no PPMI equivalent"},
    "MMSE_tot":         {"ppmi": "MCATOT", "kind": "proxy", "note": "MMSE total → MoCA total (proxy; different scale)"},
    "MCI_15SD_MDScriteria": {"ppmi": None, "kind": "drop", "note": "MCI flag (1.5 SD MDS criteria); no equivalent in PPMI feature set"},
    "PDD":              {"ppmi": None,     "kind": "drop",  "note": "PD dementia flag; no equivalent in PPMI feature set"},
    "IQCODE_mean":      {"ppmi": None,     "kind": "drop",  "note": "IQCODE not collected in PPMI"},
    "IQCODE_mean_imp":  {"ppmi": None,     "kind": "drop",  "note": "IQCODE not collected in PPMI"},
    "iqc7":             {"ppmi": None,     "kind": "drop",  "note": "IQCODE item; not in PPMI"},
    "iqc12":            {"ppmi": None,     "kind": "drop",  "note": "IQCODE item; not in PPMI"},
    "iqc14":            {"ppmi": None,     "kind": "drop",  "note": "IQCODE item; not in PPMI"},
    "iqc23":            {"ppmi": None,     "kind": "drop",  "note": "IQCODE item; not in PPMI"},
    "iqc_tot":          {"ppmi": None,     "kind": "drop",  "note": "IQCODE total; not in PPMI"},

    # ── Mood / Psychiatric ────────────────────────────────────────────────
    "madrs10":     {"ppmi": "GDS_TOT", "kind": "proxy", "note": "MADRS item 10 → GDS total (proxy; different mood scale)"},
    "madrs_tot":   {"ppmi": "GDS_TOT", "kind": "proxy", "note": "MADRS total → GDS total (proxy)"},
    "madrs7":      {"ppmi": "GDS_TOT", "kind": "proxy", "note": "MADRS item 7 → GDS total (proxy)"},
    "HAD4":        {"ppmi": None, "kind": "drop", "note": "HADS not collected in PPMI"},
    "HAD7":        {"ppmi": None, "kind": "drop", "note": "HADS not collected in PPMI"},
    "HAD13":       {"ppmi": None, "kind": "drop", "note": "HADS not collected in PPMI"},
    "HAD_sumAD":   {"ppmi": "GDS_TOT", "kind": "proxy", "note": "HADS A+D sum → GDS total (proxy)"},
    "HAD_sumA":    {"ppmi": None, "kind": "drop", "note": "HADS Anxiety subscale; no clean PPMI equivalent"},
    "npi7hi":      {"ppmi": "NP1ANXS", "kind": "proxy", "note": "NPI item 7 (anxiety) → NP1ANXS"},
    "npi8hi":      {"ppmi": "NP1HALL", "kind": "proxy", "note": "NPI item 8 → NP1HALL (hallucinations)"},
    "npi_sum_HxI": {"ppmi": None, "kind": "drop", "note": "NPI severity x interference sum; no PPMI equivalent"},

    # ── Sleep ─────────────────────────────────────────────────────────────
    "pdss02":      {"ppmi": "PDSS_DFCLTY_STAY_ASLEEP_OL", "kind": "direct", "note": "PDSS item 2 (staying asleep) → PDSS_DFCLTY_STAY_ASLEEP_OL"},
    "pdss09":      {"ppmi": None, "kind": "drop", "note": "PDSS item 9 — no PPMI online PDSS-2 equivalent identified in current feature set"},
    "pdss_tot":    {"ppmi": None, "kind": "drop", "note": "PDSS total — not provided as a column in PPMI loader (only items)"},
    "sleephou":    {"ppmi": None, "kind": "drop", "note": "Hours of sleep per night; no equivalent in PPMI loader"},
    "sleepqua":    {"ppmi": None, "kind": "drop", "note": "Sleep quality; no equivalent in PPMI loader"},
    "epwo5":       {"ppmi": "ESS5", "kind": "direct", "note": "Epworth item 5 → ESS5"},
    "epwo6":       {"ppmi": "ESS6", "kind": "direct", "note": "Epworth item 6 → ESS6"},
    "dayslee3":    {"ppmi": "NP1SLPD", "kind": "proxy", "note": "Frequency of daytime sleep → NP1SLPD (MDS-UPDRS daytime sleepiness)"},
    "dayslee4":    {"ppmi": "NP1SLPD", "kind": "proxy", "note": "Duration of daytime sleep → NP1SLPD (proxy, same item)"},
    "dayslee5":    {"ppmi": "NP1SLPD", "kind": "proxy", "note": "Sudden sleep onset → NP1SLPD (proxy)"},

    # ── Autonomic ─────────────────────────────────────────────────────────
    "URINDYS":         {"ppmi": "NP1URIN", "kind": "direct", "note": "urinary dysfunction → NP1URIN"},
    "CONSTIPAT":       {"ppmi": "NP1CNST", "kind": "direct", "note": "constipation → NP1CNST"},
    "Lightheadedness": {"ppmi": "NP1LTHD", "kind": "direct", "note": "lightheadedness on standing → NP1LTHD"},
    "PressStand_Dia":  {"ppmi": None, "kind": "drop", "note": "Orthostatic standing diastolic BP; not in PPMI loader"},
    "OLFACTOR":        {"ppmi": None, "kind": "drop", "note": "Olfaction test (UPSIT-like); not in current PPMI loader"},
    "SWEAT":           {"ppmi": None, "kind": "drop", "note": "Sweating; no clean MDS-UPDRS Part I equivalent"},

    # ── Fatigue ───────────────────────────────────────────────────────────
    "fss2":     {"ppmi": "NP1FATG", "kind": "proxy", "note": "FSS item 2 → NP1FATG (single MDS-UPDRS fatigue item, proxy)"},
    "fss3":     {"ppmi": "NP1FATG", "kind": "proxy", "note": "FSS item 3 → NP1FATG (proxy)"},
    "fss5":     {"ppmi": "NP1FATG", "kind": "proxy", "note": "FSS item 5 → NP1FATG (proxy)"},
    "fss_sum":  {"ppmi": "NP1FATG", "kind": "proxy", "note": "FSS sum → NP1FATG (proxy)"},
    "fss_mean": {"ppmi": "NP1FATG", "kind": "proxy", "note": "FSS mean → NP1FATG (proxy)"},

    # ── Stress ────────────────────────────────────────────────────────────
    "stress4":    {"ppmi": None, "kind": "drop", "note": "Stress scale; no PPMI equivalent"},
    "stress12":   {"ppmi": None, "kind": "drop", "note": "Stress scale; no PPMI equivalent"},
    "stress_tot": {"ppmi": None, "kind": "drop", "note": "Stress total; no PPMI equivalent"},

    # ── Quality of life (SF-36 family) ────────────────────────────────────
    "sf36pf":   {"ppmi": None, "kind": "drop", "note": "SF-36 physical function — SF-36 not collected in PPMI"},
    "sf36pf_z": {"ppmi": None, "kind": "drop", "note": "SF-36 physical function z-score — not in PPMI"},
    "sf36pcs":  {"ppmi": None, "kind": "drop", "note": "SF-36 physical component summary — not in PPMI"},
    "sf36mcs":  {"ppmi": None, "kind": "drop", "note": "SF-36 mental component summary — not in PPMI"},
    "PHYFUN":   {"ppmi": None, "kind": "drop", "note": "SF36-RAND physical functioning — not in PPMI"},
    "TPF":      {"ppmi": None, "kind": "drop", "note": "SF36-RAND physical functioning t-score — not in PPMI"},
    # hl3a-hl3j are SF-36 physical-function items
    "hl3a": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3b": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3c": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3d": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3e": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3f": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3g": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3h": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3i": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},
    "hl3j": {"ppmi": None, "kind": "drop", "note": "SF-36 item — not in PPMI"},

    # ── Demographics ──────────────────────────────────────────────────────
    "age_at_visit":           {"ppmi": "AGE_AT_VISIT", "kind": "direct", "note": "Age at visit"},
    "disease_duration_years": {"ppmi": None, "kind": "drop", "note": "Disease duration in years; not computed in current PPMI loader"},
    "2RedusertInntekt":       {"ppmi": None, "kind": "drop", "note": "Reduced income flag (Norwegian); ParkWest-specific demographic"},
    "DRIVLICENSE":            {"ppmi": None, "kind": "drop", "note": "Driving license; not in PPMI feature set"},
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def translate(features: list[str]) -> list[dict]:
    """For each ParkWest feature, return a dict with keys:
        parkwest, ppmi (list[str]|None), kind, note, status
    `ppmi` is always a list (possibly empty); `status` is "mapped" or "dropped"
    or "unmapped" if the feature is not in MAP at all.
    """
    out = []
    for f in features:
        if f not in MAP:
            out.append({"parkwest": f, "ppmi": [], "kind": "unmapped",
                        "note": "Not in cross-cohort map; please add.",
                        "status": "unmapped"})
            continue
        entry = MAP[f]
        ppmi = entry["ppmi"]
        kind = entry["kind"]
        note = entry["note"]
        if ppmi is None:
            ppmi_list = []
            status = "dropped"
        elif isinstance(ppmi, list):
            ppmi_list = ppmi
            status = "mapped"
        else:
            ppmi_list = [ppmi]
            status = "mapped"
        out.append({"parkwest": f, "ppmi": ppmi_list, "kind": kind,
                    "note": note, "status": status})
    return out


def format_report(entries: list[dict], source_name: str) -> str:
    """Build a human-readable txt report."""
    sep = "=" * 78
    lines = [
        sep,
        f"PPMI MAPPING:  {source_name}",
        sep, "",
        "Each ParkWest feature is mapped to the closest PPMI column(s) using the",
        "cross-cohort feature map (ppmi/cross_cohort_mapping.py). Mappings flagged",
        "as 'proxy' are conceptually close but use a different instrument; 'derived'",
        "means the PPMI side is computed as the sum of the listed items.",
        "",
        sep, "PER-FEATURE MAPPING", sep, "",
        f"  {'ParkWest':<25s}  {'PPMI':<35s}  {'Kind':<8s}",
        f"  {'-'*25}  {'-'*35}  {'-'*8}",
    ]
    for e in entries:
        ppmi_str = " + ".join(e["ppmi"]) if e["ppmi"] else "(dropped)"
        lines.append(f"  {e['parkwest']:<25s}  {ppmi_str:<35s}  {e['kind']:<8s}")
        lines.append(f"      {e['note']}")
        lines.append("")

    n_mapped  = sum(1 for e in entries if e["status"] == "mapped")
    n_dropped = sum(1 for e in entries if e["status"] == "dropped")
    n_unmapped = sum(1 for e in entries if e["status"] == "unmapped")
    lines += [
        sep,
        "SUMMARY",
        sep,
        f"  Input features         : {len(entries)}",
        f"  Successfully mapped    : {n_mapped}",
        f"  Dropped (no equivalent): {n_dropped}",
        f"  Unmapped (missing map) : {n_unmapped}",
        "",
    ]

    unique_ppmi = []
    seen = set()
    for e in entries:
        for col in e["ppmi"]:
            if col not in seen:
                seen.add(col)
                unique_ppmi.append(col)
    lines += [
        sep,
        "PPMI FEATURE LIST  (unique columns ready for LSTM input)",
        sep, "",
    ]
    for i, col in enumerate(unique_ppmi, 1):
        lines.append(f"  {i:3d}. {col}")
    lines += [
        "",
        f"  Total unique PPMI columns: {len(unique_ppmi)}",
        sep,
    ]
    return "\n".join(lines) + "\n"


def write_report(entries: list[dict], source_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_report(entries, source_name), encoding="utf-8")
    print(f"[Mapping] Wrote {output_path.name} "
          f"({sum(1 for e in entries if e['status']=='mapped')} mapped, "
          f"{sum(1 for e in entries if e['status']=='dropped')} dropped, "
          f"{sum(1 for e in entries if e['status']=='unmapped')} unmapped)")
