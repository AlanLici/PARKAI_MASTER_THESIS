"""
domain_feature_lists.py
-----------------------
Categorises the ~700 ParkWest features into clinical domain subgroups for
hierarchical Cox fall-risk analysis.

IMPORTANT – data privacy
    Only column *headers* are loaded (nrows=0).  No patient values are ever
    read, stored, or printed.

Output (written by the standalone main() to results/ next to this script)
    results/domain_feature_map.csv
        One row per (domain, feature) pair with availability info.
    results/domain_summary.csv
        One row per domain: total features defined, how many found in data,
        and which visit variants were detected.

Note
    Only DOMAINS, INVERT_FEATURES, VISIT_PREFIXES, and apply_feature_inversions
    are imported by the Cox pipelines. The main() below is a standalone audit
    tool that reports which domain features exist in the ParkWest data.
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # .../Common/
PROJECT_ROOT = SCRIPT_DIR.parents[2]                    # .../Alan_Sondre/
DATA_PATH = (
    PROJECT_ROOT / "Data" / "ParkWest" / "Raw"
    / "ParkVest_ClinicalData_with_Metadata.xlsx"
)

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Domain feature definitions
# Keys   : base feature name (no visit prefix, e.g. "MMSE_tot" not "BL_MMSE_tot")
# Values : True  = summary / total score  (preferred for first-pass analysis)
#          False = individual item        (use only if summary not available
#                                          or for within-domain drill-down)
#
# Exclusions noted explicitly:
#   UPO13 (falling ADL) and UPO14 (freezing ADL) are the *event definition*
#   and must never be used as predictors.
# ---------------------------------------------------------------------------

DOMAINS: dict[str, dict[str, bool]] = {

    # ── 1. MOTOR EXAMINATION (UPDRS-III / MDS-UPDRS-III) ──────────────────
    "Motor_Examination": {
        # Summary scores
        "UPO_part3_tot":  True,
        "UPN_part3_tot":  True,
        "PIGD":           True,   # Postural Instability & Gait Disorder score
        "PIGD_ratio":     True,
        "HYON":           True,   # Hoehn & Yahr (ON state)
        "SEON":           True,   # Schwab & England (ON state)
        # UPO-III individual items
        "UPO18":  False,  # speech (motor)
        "UPO19":  False,  # facial expression
        "UPO20a": False,  # tremor at rest – face/lips/chin
        "UPO20b": False,  # tremor at rest – right arm
        "UPO20c": False,  # tremor at rest – left arm
        "UPO20d": False,  # tremor at rest – right leg
        "UPO20e": False,  # tremor at rest – left leg
        "UPO21a": False,  # postural tremor – right arm
        "UPO21b": False,  # postural tremor – left arm
        "UPO22a": False,  # rigidity – neck
        "UPO22b": False,  # rigidity – right arm
        "UPO22c": False,  # rigidity – left arm
        "UPO22d": False,  # rigidity – right leg
        "UPO22e": False,  # rigidity – left leg
        "UPO23a": False,  # finger taps – right
        "UPO23b": False,  # finger taps – left
        "UPO24a": False,  # hand movements – right
        "UPO24b": False,  # hand movements – left
        "UPO25a": False,  # rapid alternating movements – right
        "UPO25b": False,  # rapid alternating movements – left
        "UPO26a": False,  # leg agility – right
        "UPO26b": False,  # leg agility – left
        "UPO27":  False,  # arising from chair
        "UPO28":  False,  # posture
        "UPO29":  False,  # gait
        "UPO30":  False,  # postural stability (pull test)
        "UPO31":  False,  # body bradykinesia
        # Plantar reflexes & other exam findings
        "PLANTAR":    False,  # plantar reflex
        "PLANTAR_V":  False,  # plantar reflex – left
        "PLANTAR_H":  False,  # plantar reflex – right
        "Plantar_reflex_":       False,  # plantar reflex (alt coding)
        "Plantar_reflex_comment": False, # plantar reflex comment
        "GAZEPALSY":  False,  # gaze palsy
        "HYOFF":      False,  # Hoehn & Yahr (OFF state)
        "SEOFF":      False,  # Schwab & England (OFF state)
        # UPN-III individual items (MDS-UPDRS)
        "UPN3.1":    False,  # speech
        "UPN3.2":    False,  # facial expression
        "UPN3.3neck": False, # rigidity neck
        "UPN3.3RUE": False,  # rigidity right arm
        "UPN3.3LUE": False,  # rigidity left arm
        "UPN3.3RLE": False,  # rigidity right leg
        "UPN3.3LLE": False,  # rigidity left leg
        "UPN3.4R":   False,  # finger taps right
        "UPN3.4L":   False,  # finger taps left
        "UPN3.5R":   False,  # hand movements right
        "UPN3.5L":   False,  # hand movements left
        "UPN3.6R":   False,  # pronation-supination right
        "UPN3.6L":   False,  # pronation-supination left
        "UPN3.7R":   False,  # toe tapping right
        "UPN3.7L":   False,  # toe tapping left
        "UPN3.8R":   False,  # leg agility right
        "UPN3.8L":   False,  # leg agility left
        "UPN3.9":    False,  # arising from chair
        "UPN3.10":   False,  # gait
        "UPN3.11":   False,  # freezing of gait
        "UPN3.12":   False,  # postural stability
        "UPN3.13":   False,  # posture
        "UPN3.14":   False,  # body bradykinesia (global)
        "UPN3.15R":  False,  # postural tremor right
        "UPN3.15L":  False,  # postural tremor left
        "UPN3.16R":  False,  # kinetic tremor right
        "UPN3.16L":  False,  # kinetic tremor left
        "UPN3.17LIP": False, # rest tremor face/lips/chin
        "UPN3.17RUE": False, # rest tremor right arm
        "UPN3.17LUE": False, # rest tremor left arm
        "UPN3.17RLE": False, # rest tremor right leg
        "UPN3.17LLE": False, # rest tremor left leg
        "UPN3.18":   False,  # constancy of rest tremor
    },

    # ── 2. ADL / MOTOR EXPERIENCES OF DAILY LIVING (UPDRS-II) ─────────────
    # NOTE: UPO13 (falling) and UPO14 (freezing) deliberately excluded —
    #       they form part of the fall event definition.
    "ADL_Motor_Daily_Living": {
        "UPO_part2_tot":    True,
        "UPO_part2_ON_tot": True,   # ON-state version
        "UPN_part2_tot":    True,
        "UPO5":  False,  # speech (ADL)
        "UPO6":  False,  # salivation
        "UPO7":  False,  # swallowing
        "UPO8":  False,  # handwriting
        "UPO9":  False,  # cutting food
        "UPO10": False,  # dressing
        "UPO11": False,  # hygiene
        "UPO12": False,  # turning in bed
        # UPO13 EXCLUDED – event definition (falling)
        # UPO14 EXCLUDED – event definition (freezing when walking)
        "UPO15": False,  # walking
        "UPO16a": False, # tremor right arm (ADL impact)
        "UPO16b": False, # tremor left arm (ADL impact)
        "UPO17": False,  # sensory complaints (ADL)
        # ADL scale (Katz / Barthel-style)
        "ADL1":  False, "ADL2":  False, "ADL3":  False, "ADL4":  False,
        "ADL5":  False, "ADL6":  False, "ADL7":  False, "ADL8":  False,
        "ADL9":  False, "ADL10": False,
        "ADL_tot": True,   # ADL total
        "ADLsum":  True,   # ADL sum score
        "UPN2.1":  False, "UPN2.2":  False, "UPN2.3":  False,
        "UPN2.4":  False, "UPN2.5":  False, "UPN2.6":  False,
        "UPN2.7":  False, "UPN2.8":  False, "UPN2.9":  False,
        "UPN2.10": False, "UPN2.11": False, "UPN2.12": False,
        "UPN2.13": False,
    },

    # ── 3. MOTOR COMPLICATIONS (UPDRS-IV) ──────────────────────────────────
    "Motor_Complications": {
        "UPO_part4_tot": True,
        "UPN_part4_tot": True,
        "UPO32": False,  # dyskinesias – duration
        "UPO33": False,  # dyskinesias – disability
        "UPO34": False,  # painful dyskinesias
        "UPO35": False,  # early morning dystonia
        "UPO36": False,  # OFF-periods predictable
        "UPO37": False,  # OFF-periods unpredictable
        "UPO38": False,  # sudden OFF-periods
        "UPO39": False,  # proportion of day in OFF
        "UPN4.1": False, "UPN4.2": False, "UPN4.3": False,
        "UPN4.4": False, "UPN4.5": False, "UPN4.6": False,
    },

    # ── 4. NON-MOTOR SYMPTOMS (UPDRS-I / NPI) ─────────────────────────────
    "Non_Motor_Symptoms": {
        "UPO_part1_tot": True,
        "UPN_part1_tot": True,
        "npi_sum_HxI":   True,   # NPI total frequency × intensity
        "npi_sum_Bel":   True,   # NPI total caregiver burden
        "UPO1":  False,  # intellectual impairment
        "UPO2":  False,  # thought disorder
        "UPO3":  False,  # depression
        "UPO4":  False,  # motivation / initiative
        "UPO40": False,  # anorexia / nausea / vomiting
        "UPO41": False,  # sleep disturbances
        "UPO42": False,  # orthostasis
        "UPN1.1":  False, "UPN1.2":  False, "UPN1.3":  False,
        "UPN1.4":  False, "UPN1.5":  False, "UPN1.6":  False,
        "UPN1.7":  False, "UPN1.8":  False, "UPN1.9":  False,
        "UPN1.10": False, "UPN1.11": False,
        # NPI individual items (frequency × intensity composite)
        "npi1hi":  False, "npi2hi":  False, "npi3hi":  False,
        "npi4hi":  False, "npi5hi":  False, "npi6hi":  False,
        "npi7hi":  False, "npi8hi":  False, "npi9hi":  False,
        "npi10hi": False, "npi11hi": False, "npi12hi": False,
        "Pain_OFF": False,  # pain (OFF state)
    },

    # ── 5. COGNITION ────────────────────────────────────────────────────────
    "Cognition": {
        "MMSE_tot":    True,   # Mini-Mental State Exam total
        "IQCODE_mean": True,   # Informant Questionnaire on Cognitive Decline
        "CFluency":    True,   # Verbal category fluency (animals/min)
        "CVLTIIA_tot": True,   # CVLT-II total learning (A1-A5)
        "STROOPW":     False,  # Stroop words (processing speed)
        "STROOPC":     False,  # Stroop colours
        "STROOPCW":    False,  # Stroop colour-word (inhibition)
        "CVLTII_A1": False, "CVLTII_A2": False, "CVLTII_A3": False,
        "CVLTII_A4": False, "CVLTII_A5": False,
        "CVLTII_DF": False,  # delayed free recall
        "CVLTII_LF": False,  # delayed cued recall
        "CVLTII_R":  False,  # recognition true positives
        "VOSPSIL":   False,  # VOSP silhouettes (visual perception)
        "VOSPCUBE":  False,  # VOSP cube analysis
        # IQCODE individual items (informant-rated cognitive decline)
        "iqc1":  False, "iqc2":  False, "iqc3":  False, "iqc4":  False,
        "iqc5":  False, "iqc6":  False, "iqc7":  False, "iqc8":  False,
        "iqc9":  False, "iqc10": False, "iqc11": False, "iqc12": False,
        "iqc13": False, "iqc14": False, "iqc15": False, "iqc16": False,
        "iqc17": False, "iqc18": False, "iqc19": False, "iqc20": False,
        "iqc21": False, "iqc22": False, "iqc23": False, "iqc24": False,
        "iqc25": False, "iqc26": False,
        "iqc_tot":     True,   # IQCODE total
        "iqcode_who":  False,  # IQCODE informant relationship
        # MMSE individual items
        "MMSE_1": False, "MMSE_2": False, "MMSE_3": False, "MMSE_4": False,
        "MMSE_5": False, "MMSE_6": False, "MMSE_7": False, "MMSE_8": False,
        "MMSE_9": False, "MMSE_10": False, "MMSE_11": False,
        "MMSE_12a": False, "MMSE_12b": False, "MMSE_12sum": False,
        "MMSE_13": False, "MMSE_14": False, "MMSE_15": False,
        "MMSE_16": False, "MMSE_17": False, "MMSE_18": False,
        "MMSE_19": False, "MMSE_20": False,
        # Dementia assessment (DSM-IV criteria)
        "Demens_A.Hukommelse1":    False, "Demens_A.Hukummelse2":     False,
        "Demens_A.Hukommelse3":    False, "Demens_A.Afasi":           False,
        "Demens_A.Apraksi":        False, "Demens_A.Agnosi":          False,
        "Demens_A.Planlegging":    False, "Demens_A.Organisering":    False,
        "Demens_A.Dømmekraft":     False, "Demens_A.Abstenkn":        False,
        "Demens_A.minst_ett_ja":   False,
        "Demens_B1.Husarbeid":     False, "Demens_B1.Økonomistyring": False,
        "Demens_B1.Venner":        False, "Demens_B1.Hobbyer":        False,
        "Demens_B1.Daglidagseoppg": False, "Demens_B1.Finne_fram":   False,
        "Demens_B1.minst_ett_ja":  False,
        "Demens_B2.neg_endr_huk":  False, "Demens_B2.daglig_funk_verre": False,
        "Demens_B2.minst_ett_ja":  False,
        "Demens_C.annen_sykdomsproses":  False,
        "Demens_C.annen_sykdomsprosess": False,
        "Demens_C.annen_sykdp.årsak":   False,
        "Demens_D.kriterium":      False,
        "Demens_tot":              True,   # dementia criteria total
        "Demens.relasjon":         False,
        "Demens.vurd_isdone":      False,
        "MCI_15SD_MDScriteria":    False,  # MCI by MDS criteria
        "PDD":                     False,  # PD dementia diagnosis
        "Perservasjoner":          False,  # perseverations
        "Vurdert.kognitivsvikt":   False,  # cognitive impairment assessed
        "Vurdert.kognitivsvikt.dato": False,
    },

    # ── 6. MOOD / PSYCHIATRIC ───────────────────────────────────────────────
    "Mood_Psychiatric": {
        "madrs_tot": True,   # MADRS total (depression severity)
        "madrs1":  False, "madrs2":  False, "madrs3":  False,
        "madrs4":  False, "madrs5":  False, "madrs6":  False,
        "madrs7":  False, "madrs8":  False, "madrs9":  False,
        "madrs10": False,
        # Hospital Anxiety and Depression Scale (HAD)
        "HAD1":  False, "HAD2":  False, "HAD3":  False, "HAD4":  False,
        "HAD5":  False, "HAD6":  False, "HAD7":  False, "HAD8":  False,
        "HAD9":  False, "HAD10": False, "HAD11": False, "HAD12": False,
        "HAD13": False, "HAD14": False,
        "HAD_sumA":  True,   # HAD anxiety subscale total
        "HAD_sumD":  True,   # HAD depression subscale total
        "HAD_sumAD": True,   # HAD combined total
        # Psychiatric symptoms (hallucinations, delusions)
        "PsykSymp.Hallu.Visuelle.intensitet": False,
        "PsykSymp.Hallu.Visuelle.varighet":   False,
        "PsykSymp.Hallu.Visuelle.hyppighet":  False,
        "PsykSymp.Hallu.Auditive.intensitet": False,
        "PsykSymp.Hallu.Auditive.varighet":   False,
        "PsykSymp.Hallu.Auditive.hyppighet":  False,
        "PsykSymp.Hallu.Taktile.intensitet":  False,
        "PsykSymp.Hallu.Taktile.varighet":    False,
        "PsykSymp.Hallu.Taktile.hyppighet":   False,
        "PsykSymp.Hallu.Lukt.intensitet":     False,
        "PsykSymp.Hallu.Lukt.varighet":       False,
        "PsykSymp.Hallu.Lukt.hyppighet":      False,
        "PsykSymp.Hallu.Smak.intensitet":     False,
        "PsykSymp.Hallu.Smak.varighet":       False,
        "PsykSymp.Hallu.Smak.hyppighet":      False,
        "PsykSymp.Vrang.Capgras.varighet":    False,
        "PsykSymp.Vrang.Capgras.hyppighet":   False,
        "PsykSymp.Vrang.Para.varighet":       False,
        "PsykSymp.Vrang.Para.hyppighet":      False,
        "PsykSymp.Vrang.Sjalusi.varighet":    False,
        "PsykSymp.Vrang.Sjalusi.hyppighet":   False,
        "PsykSymp.Vrang.Annet":               False,
        "PsykSymp.Vrang.Annet.varighet":      False,
        "PsykSymp.Vrang.Annet.hyppighet":     False,
        "PsykSymp.Vrang.Homemis.varighet":    False,
        "PsykSymp.Vrang.Homemis.hyppighet":   False,
        "PsykSymp.Illusjoner.varighet":       False,
        "PsykSymp.Illusjoner.hyppighet":      False,
        "PsykSymp.Minor.Presence.varighet":   False,
        "PsykSymp.Minor.Presence.hyppighet":  False,
        "PsykSymp.Minor.Passage.varighet":    False,
        "PsykSymp.Minor.Passage.hyppighet":   False,
        "PsykSymp.Info":                      False,
        "PsykSymp.Utfyllende.a":              False,
        "PsykSymp.Utfyllende.b":              False,
        "PsykSymp.Utfyllende.c_Isolert":      False,
        "PsykSymp.Utfyllende.c_Isolert.hvilke":           False,
        "PsykSymp.Utfyllende.c_Isolert_og_Samtidig":      False,
        "PsykSymp.Utfyllende.c_Samtidig":                 False,
        "PsykSymp.Utfyllende.c_Samtidig.hvilke":          False,
        "PsykSymp.Utfyllende.d":              False,
        # QUIP (Questionnaire for Impulsive-Compulsive Disorders)
        "QUIP.A1": False, "QUIP.A2": False, "QUIP.B1": False, "QUIP.B2": False,
        "QUIP.C1": False, "QUIP.C2": False, "QUIP.D1": False, "QUIP.D2": False,
        "QUIP.E1": False, "QUIP.E2": False, "QUIP.E3": False,
        "QUIP.F1": False, "QUIP.F2": False,
        # QUIP caregiver/patient supplement
        "QUIP_PAS.A1": False, "QUIP_PAS.A2": False,
        "QUIP_PAS.B1": False, "QUIP_PAS.B2": False,
        "QUIP_PAS.C1": False, "QUIP_PAS.C2": False,
        "QUIP_PAS.D1": False, "QUIP_PAS.D2": False,
        "QUIP_PAS.E1": False, "QUIP_PAS.E2": False, "QUIP_PAS.E3": False,
        "QUIP_PAS.F1": False, "QUIP_PAS.F2": False,
        "antallpsyksympt": False,  # count of psychiatric symptoms
    },

    # ── 7. SLEEP ────────────────────────────────────────────────────────────
    "Sleep": {
        "pdss_tot": True,   # Parkinson's Disease Sleep Scale total
        "epwo_tot": True,   # Epworth Sleepiness Scale total
        "RBD":      True,   # REM Sleep Behaviour Disorder (yes/no)
        "RLS_LMR":  True,   # Restless Legs Syndrome (diagnosed)
        "pdss01": False, "pdss02": False, "pdss03": False,
        "pdss04": False, "pdss05": False, "pdss06": False,
        "pdss07": False, "pdss08": False, "pdss09": False,
        "pdss10": False, "pdss11": False, "pdss12": False,
        "pdss13": False, "pdss14": False, "pdss15": False,
        "epwo1": False, "epwo2": False, "epwo3": False,
        "epwo4": False, "epwo5": False, "epwo6": False,
        "epwo7": False, "epwo8": False,
        # Sleep questionnaire items
        "sleep_height":  False, "sleep_weight":  False, "sleep_bath":    False,
        "sleep_headache": False, "sleep_rest_morning": False,
        "sleepqua": False, "sleephou": False, "sleeptim": False,
        "sleepsno": False, "sleepre1": False, "sleepre2": False,
        "sleepagi": False, "sleepfre": False, "sleepfre2": False,
        "sleeppro": False, "sleepmedicine": False, "sleepmedicine.which": False,
        # Daytime sleepiness items
        "dayslee1": False, "dayslee2": False, "dayslee3": False,
        "dayslee4": False, "dayslee5": False, "dayslee6": False,
        "dayslee7": False,
        "awakefre": False,  # awake frequency
        "time_to_bed": False,
        # Sleep problem type
        "type_sleep_problem_A": False,
        "type_sleep_problem_B": False,
        "type_sleep_problem_C": False,
        # Restless Legs Syndrome items
        "rls1": False, "rls2": False, "rls3": False, "rls4": False,
        "rls5": False, "rls6": False, "rls6a": False, "rls6b": False,
        "rls6c": False, "rls7": False, "rls8": False,
        "rlshi": False, "rlswei": False,
    },

    # ── 8. AUTONOMIC ────────────────────────────────────────────────────────
    "Autonomic": {
        # Blood pressure – orthostatic hypotension
        "PressHori_Sys":  False,  # supine systolic
        "PressHori_Dia":  False,  # supine diastolic
        "PressStand_Sys": False,  # standing systolic
        "PressStand_Dia": False,  # standing diastolic
        # Other autonomic features
        "URINDYS":   False,  # urinary dysfunction
        "SWEAT":     False,  # sweating
        "CONSTIPAT": False,  # constipation
        "OLFACTOR":  False,  # olfactory function (normal/abnormal)
        # Additional autonomic features
        "Saliva":              False,  # saliva / drooling
        "Saliva_OFF":          False,  # saliva (OFF state)
        "Swallowing":          False,  # swallowing difficulty
        "Swallowing_OFF":      False,  # swallowing (OFF state)
        "Lightheadedness":     False,  # lightheadedness / dizziness
        "Lightheadedness_OFF": False,  # lightheadedness (OFF state)
        "CONSTIPAT_OFF":       False,  # constipation (OFF state)
        "URINDYS_OFF":         False,  # urinary dysfunction (OFF state)
        "SWEAT_OFF":           False,  # sweating (OFF state)
    },

    # ── 9. APATHY & FATIGUE ─────────────────────────────────────────────────
    "Apathy_Fatigue": {
        "apati_tot": True,   # Starkstein Apathy Scale total
        "fss_mean":  True,   # Fatigue Severity Scale mean
        "fss_sum":   True,   # Fatigue Severity Scale sum
        "apati1":  False, "apati2":  False, "apati3":  False,
        "apati4":  False, "apati5":  False, "apati6":  False,
        "apati7":  False, "apati8":  False, "apati9":  False,
        "apati10": False, "apati11": False, "apati12": False,
        "apati13": False, "apati14": False,
        "fss1": False, "fss2": False, "fss3": False,
        "fss4": False, "fss5": False, "fss6": False,
        "fss7": False, "fss8": False, "fss9": False,
        # Perceived Stress Scale items
        "stress1":  False, "stress2":  False, "stress3":  False,
        "stress4":  False, "stress5":  False, "stress6":  False,
        "stress7":  False, "stress8":  False, "stress9":  False,
        "stress10": False, "stress11": False, "stress12": False,
        "stress13": False, "stress14": False, "stress15": False,
        "stress_tot": True,   # stress total
        "stress_sum": True,   # stress sum
    },

    # ── 10. QUALITY OF LIFE (SF-36) ─────────────────────────────────────────
    "Quality_of_Life": {
        "sf36pcs":  True,   # Physical Component Summary
        "sf36mcs":  True,   # Mental Component Summary
        "sf36pf":   False,  # physical functioning subscale
        "sf36rp":   False,  # role physical
        "sf36bp":   False,  # bodily pain
        "sf36gh":   False,  # general health
        "sf36vt":   False,  # vitality
        "sf36sf":   False,  # social functioning
        "sf36re":   False,  # role emotional
        "sf36mh":   False,  # mental health
        "PPHC":     False,  # RAND physical health composite
        "MMHC":     False,  # RAND mental health composite
        "GGHC":     False,  # RAND global health composite
        # SF-36 / RAND raw items (hl = health literacy / raw items)
        "hl1":  False, "hl2":  False,
        "hl3a": False, "hl3b": False, "hl3c": False, "hl3d": False,
        "hl3e": False, "hl3f": False, "hl3g": False, "hl3h": False,
        "hl3i": False, "hl3j": False,
        "hl4a": False, "hl4b": False, "hl4c": False, "hl4d": False,
        "hl5a": False, "hl5b": False, "hl5c": False,
        "hl6":  False, "hl7":  False, "hl8":  False,
        "hl9a": False, "hl9b": False, "hl9c": False, "hl9d": False,
        "hl9e": False, "hl9f": False, "hl9g": False, "hl9h": False,
        "hl9i": False, "hl10": False,
        "hl11a": False, "hl11b": False, "hl11c": False, "hl11d": False,
        # Satisfaction with Life Scale (SWLS)
        "SWLS.1_ideal":         False,
        "SWLS.2_livsforhold":   False,
        "SWLS.3_tilfredshet":   False,
        "SWLS.4_viktigeting":   False,
        "SWLS.5_forandretnoe":  False,
        "SWLS_tot":             True,   # SWLS total
        # SF-36 transformed / computed subscale scores
        "PHYFUN":  False,  # physical functioning (raw)
        "TPF":     False,  # transformed physical functioning
        "PHC":     False,  # physical health composite
        "TPHC":    False,  # transformed physical health composite
        "GHC":     False,  # general health composite
        "TGHC":    False,  # transformed general health composite
        "PAIN":    False,  # pain (uppercase)
        "Pain":    False,  # pain (lowercase)
        "TPA":     False,  # transformed pain
        "ENFAT":   False,  # energy / fatigue
        "TEF":     False,  # transformed energy / fatigue
        "MHC":     False,  # mental health composite
        "TMHC":    False,  # transformed mental health composite
        "ROLEP":   False,  # role physical (raw)
        "TRLP":    False,  # transformed role physical
        "ROLEE":   False,  # role emotional (raw)
        "TRLE":    False,  # transformed role emotional
        "SOCFUN":  False,  # social functioning (raw)
        "TSF":     False,  # transformed social functioning
        "GENH":    False,  # general health (raw)
        "TEWB":    False,  # transformed emotional well-being
        "EMOT":    False,  # emotional (raw)
        "TGHP":    False,  # transformed general health perceptions
        "nsvar_SF36": False,  # SF-36 response count
    },

    # ── 11. COMORBIDITIES ───────────────────────────────────────────────────
    "Comorbidities": {
        "CCI":             True,   # Charlson Comorbidity Index total
        "coronarydisease": False,
        "diabetes":        False,
        "stroke":          False,
        "Hyperten":        False,
        "atrieflimmer":    False,  # atrial fibrillation
        "hyperchol":       False,  # hypercholesterolaemia
        "depression":      False,
        "CCI_MI":    False, "CCI_CHF":  False, "CCI_PVD":  False,
        "CCI_CVA":   False, "CCI_COPD": False, "CCI_DM":   False,
        "CCI_RENAL": False,
    },
}

# ---------------------------------------------------------------------------
# Known visit prefixes (extend if more visits exist in the dataset)
# ---------------------------------------------------------------------------
VISIT_PREFIXES = ["BL", "V4", "V5", "V9", "V13", "V17"]

# ---------------------------------------------------------------------------
# FEATURE INVERSION MAP
# Features where the scale runs "higher = better / healthier", so they must
# be negated before fitting so that the resulting HR is always interpretable
# as: HR > 1 = higher impairment = higher fall risk.
#
# DO NOT add features here if their direction is ambiguous or if a low value
# could genuinely be protective (e.g. blood pressure, medication variables).
# Those should remain as-is so genuine protective effects can emerge naturally.
#
# Implementation: long_df[feat] = -long_df[feat]  (applied before standardisation)
# ---------------------------------------------------------------------------
INVERT_FEATURES: set[str] = {
    # ── Cognition: higher score = better performance ──────────────────────
    "MMSE_tot",
    "STROOPW", "STROOPC", "STROOPCW",
    "CFluency",
    "CVLTIIA_tot",
    "CVLTII_A1", "CVLTII_A2", "CVLTII_A3", "CVLTII_A4", "CVLTII_A5",
    "CVLTII_AR", "CVLTII_SF", "CVLTII_DF", "CVLTII_LF",
    "CVLTII_R",
    "VOSPSIL", "VOSPCUBE",

    # ── Motor function: higher = more independent ─────────────────────────
    "SEON",   # Schwab & England ON state (100% = fully independent)
    # Note: HYON (Hoehn & Yahr) is NOT inverted — higher stage = worse

    # ── Sleep: higher PDSS = better sleep quality ─────────────────────────
    "pdss_tot",
    "pdss01", "pdss02", "pdss03", "pdss04", "pdss05",
    "pdss06", "pdss07", "pdss08", "pdss09", "pdss10",
    "pdss11", "pdss12", "pdss13", "pdss14", "pdss15",
    # Note: Epworth (epwo) is NOT inverted — higher = more sleepy = worse

    # ── Quality of Life: higher SF-36 = better health ─────────────────────
    "sf36pf", "sf36rp", "sf36bp", "sf36gh",
    "sf36vt", "sf36sf", "sf36re", "sf36mh",
    "sf36pcs", "sf36mcs",
    "sf36aggp", "sf36aggm",
    "PPHC", "MMHC", "GGHC",
    # SF-36 PF (item 3a-j): "Does your health limit you?"
    # 1 = limits a lot, 2 = limits a bit, 3 = not at all  → higher = better
    "hl3a", "hl3b", "hl3c", "hl3d", "hl3e",
    "hl3f", "hl3g", "hl3h", "hl3i", "hl3j",
    # SF-36 PF aggregates (raw + transformed)
    "PHYFUN", "TPF",
    # Satisfaction With Life Scale: higher = more satisfied
    "SWLS.1_ideal", "SWLS.2_livsforhold", "SWLS.3_tilfredshet",
    "SWLS.4_viktigeting", "SWLS.5_forandretnoe", "SWLS_tot",
    # Note: hl1, hl2, hl6, hl7, hl8 use "lower = better" coding — DO NOT invert.
    # Note: hl4*, hl5*, hl9*, hl11* have mixed/ambiguous direction — leave for now.
}


def apply_feature_inversions(long_df, features: list[str]) -> tuple:
    """
    Negate features in INVERT_FEATURES so that higher always means more impaired.

    Applied BEFORE standardisation.  Returns the modified DataFrame and a list
    of which features were actually inverted (for the results table).
    """
    import pandas as pd
    inverted = []
    for feat in features:
        if feat in INVERT_FEATURES and feat in long_df.columns:
            long_df[feat] = -long_df[feat]
            inverted.append(feat)
    if inverted:
        print(f"[Inversion] Negated {len(inverted)} features (higher = worse): "
              f"{inverted}")
    return long_df, inverted


def load_column_names(path: str) -> list[str]:
    """Read ONLY the header row – zero patient data rows are loaded."""
    return pd.read_excel(path, nrows=0).columns.tolist()


def find_variants(base_name: str, all_columns: list[str]) -> list[str]:
    """
    Return all column names that match <PREFIX>_<base_name> or <base_name>
    for any known visit prefix.
    """
    candidates = []
    for col in all_columns:
        for prefix in VISIT_PREFIXES:
            if col == f"{prefix}_{base_name}":
                candidates.append(col)
                break
        else:
            if col == base_name:
                candidates.append(col)
    return candidates


def build_feature_map(all_columns: list[str]) -> pd.DataFrame:
    """
    Build a flat table: one row per (domain, feature) with availability info.
    """
    rows = []
    for domain, features in DOMAINS.items():
        for base_name, is_summary in features.items():
            variants = find_variants(base_name, all_columns)
            rows.append({
                "domain":           domain,
                "base_feature":     base_name,
                "is_summary_score": is_summary,
                "found_in_data":    len(variants) > 0,
                "n_visit_variants": len(variants),
                "visit_variants":   ", ".join(variants) if variants else "NOT FOUND",
            })
    return pd.DataFrame(rows)


def build_domain_summary(feature_map: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per domain.
    """
    rows = []
    for domain in DOMAINS:
        subset = feature_map[feature_map["domain"] == domain]
        summary = subset[subset["is_summary_score"]]
        items   = subset[~subset["is_summary_score"]]
        rows.append({
            "domain":                    domain,
            "n_features_defined":        len(subset),
            "n_summary_scores_defined":  len(summary),
            "n_items_defined":           len(items),
            "n_found_in_data":           subset["found_in_data"].sum(),
            "n_summary_found":           summary["found_in_data"].sum(),
            "n_items_found":             items["found_in_data"].sum(),
            "missing_features":          ", ".join(
                subset.loc[~subset["found_in_data"], "base_feature"].tolist()
            ) or "none",
        })
    return pd.DataFrame(rows)


def main():
    print("[Domain mapping] Loading column headers only (no patient data)...")
    all_columns = load_column_names(DATA_PATH)
    print(f"[Domain mapping] {len(all_columns)} columns found in dataset.")

    feature_map = build_feature_map(all_columns)
    domain_summary = build_domain_summary(feature_map)

    # Save outputs
    map_path     = OUTPUT_DIR / "domain_feature_map.csv"
    summary_path = OUTPUT_DIR / "domain_summary.csv"
    feature_map.to_csv(map_path, index=False)
    domain_summary.to_csv(summary_path, index=False)

    print(f"\n[Output] Feature map saved  -> {map_path}")
    print(f"[Output] Domain summary saved -> {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("DOMAIN SUMMARY")
    print("=" * 70)
    print(domain_summary.to_string(index=False))
    print("\nNote: 'NOT FOUND' features may use a different naming convention")
    print("      in the dataset. Check domain_feature_map.csv for details.")


if __name__ == "__main__":
    main()
