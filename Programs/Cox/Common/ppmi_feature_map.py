"""
ppmi_feature_map.py
-------------------
Domain feature lists for the PPMI dataset, mirroring the ParkWest
domain_feature_lists.py structure.

Mapping notes
-------------
Motor_Examination  : MDS-UPDRS Part III items + computed PIGD score
ADL                : MDS-UPDRS Part II items
Motor_Complications: MDS-UPDRS Part IV items
Non_Motor_Symptoms : MDS-UPDRS Part I (rater + patient questionnaire)
Cognition          : MoCA items
Mood_Psychiatric   : Geriatric Depression Scale (GDS) items
Sleep              : PDSS-2 online items + Epworth Sleepiness Scale
Autonomic          : SCOPA-AUT items
Apathy_Fatigue     : NP1FATG (UPDRS fatigue item) — no full FSS in PPMI
Quality_of_Life    : Modified Schwab & England ADL (MSEADLG)
Comorbidities      : not available in downloaded PPMI files

ParkWest -> PPMI feature correspondence (Cox-validated features)
    PIGD       -> PIGD (computed: mean of NP2WALK, NP2FREZ, NP3GAIT, NP3FRZGT, NP3PSTBL)
    UPO12      -> NP2TURN  (turning in bed)
    npi8hi     -> NP1HALL  (hallucinations — UPDRS Part I rater item)
    URINDYS    -> NP1URIN  (urinary problems — UPDRS Part I patient item)
    pdss02     -> PDSS_DFCLTY_STAY_ASLEEP_OL
    fss2       -> NP1FATG  (single fatigue item — proxy only)
    STROOPW    -> MCATOT   (MoCA total — different test, same domain)
    madrs10    -> GDS total (sum of 15 GDS items — different scale)
    sf36pf     -> MSEADLG  (Schwab & England — different scale)
"""

# ---------------------------------------------------------------------------
# Domain -> feature list
# ---------------------------------------------------------------------------
DOMAINS: dict[str, list[str]] = {

    "Motor_Examination": [
        # Computed composite (must be created in data loader)
        # PIGD excluded — computed from NP2WALK/NP2FREZ which are in event definition
        # Gait / postural instability components
        "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3RISNG",
        # Rigidity
        "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL",
        # Tremor
        "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
        "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON",
        # Dexterity
        "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
        "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
        "NP3LGAGR", "NP3LGAGL",
        # Face / speech
        "NP3SPCH", "NP3FACXP",
        # Summary scores
        "NP3TOT", "NHY",
    ],

    "ADL_Motor_Daily_Living": [
        "NP2TURN",   # turning in bed  <- ParkWest UPO12
        # NP2WALK and NP2FREZ excluded — used in event definition
        "NP2RISE",   # getting out of chair
        "NP2DRES",   # dressing
        "NP2HYGN",   # hygiene
        "NP2HWRT",   # handwriting
        "NP2HOBB",   # hobbies
        "NP2EAT",    # eating
        "NP2TRMR",   # tremor
        "NP2SPCH",   # speech
        "NP2SALV",   # saliva
        "NP2SWAL",   # swallowing
        # NP2PTOT excluded — includes NP2WALK/NP2FREZ which are in event definition
        # MSEADLG moved to Quality_of_Life (proxy for PW's SF-36)
    ],

    "Motor_Complications": [
        "NP4WDYSK",  # duration of dyskinesias
        "NP4DYSKI",  # functional impact of dyskinesias
        "NP4OFF",    # time in off state
        "NP4FLCTI",  # complexity of motor fluctuations (internal)
        "NP4FLCTX",  # complexity of motor fluctuations (external)
        "NP4DYSTN",  # painful off-state dystonias
        "NP4TOT",    # Part IV total
    ],

    "Non_Motor_Symptoms": [
        # Rater-completed (Part I)
        "NP1HALL",   # hallucinations  <- ParkWest npi8hi
        "NP1COG",    # cognitive impairment
        "NP1DPRS",   # depressed mood
        "NP1ANXS",   # anxiety
        # NP1APAT moved to Apathy_Fatigue (avoid cross-domain duplication)
        "NP1DDS",    # dopamine dysregulation
        "NP1RTOT",   # Part I rater total
        # Patient questionnaire (Part I)
        "NP1URIN",   # urinary problems  <- ParkWest URINDYS
        "NP1CNST",   # constipation
        "NP1LTHD",   # lightheadedness
        "NP1PAIN",   # pain
        "NP1SLPN",   # sleep (night)
        "NP1SLPD",   # daytime sleepiness
        # NP1FATG moved to Apathy_Fatigue (avoid cross-domain duplication)
        "NP1PTOT",   # Part I patient total
    ],

    "Cognition": [
        "MCATOT",    # MoCA total  <- ParkWest STROOPW proxy
        "MCAALTTM",  # alternating trail making
        "MCACUBE",   # cube copy
        "MCACLCKC",  # clock contour
        "MCACLCKN",  # clock numbers
        "MCACLCKH",  # clock hands
        "MCAVF",     # verbal fluency (letter F)
        "MCAVFNUM",  # verbal fluency count
        "MCAABSTR",  # abstraction
        "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5",  # delayed recall
        "MCAFDS",    # forward digit span
        "MCABDS",    # backward digit span
        "MCAVIGIL",  # vigilance
        "MCASER7",   # serial 7s
        "MCASNTNC",  # sentence repetition
        "MCALION", "MCARHINO", "MCACAMEL",  # naming
    ],

    "Mood_Psychiatric": [
        # GDS-15 items  <- ParkWest MADRS proxy
        "GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSGSPIR",
        "GDSAFRAD", "GDSHAPPY", "GDSHLPLS", "GDSHOME",  "GDSMEMRY",
        "GDSALIVE", "GDSWRTLS", "GDSENRGY", "GDSHOPLS", "GDSBETER",
        "GDS_TOT",   # computed total (sum of 15 items)
    ],

    "Sleep": [
        # PDSS-2 online items  <- ParkWest pdss items
        "PDSS_SLEPT_WELL_OL",
        "PDSS_DFCLTY_FALL_ASLEEP_OL",
        "PDSS_DFCLTY_STAY_ASLEEP_OL",   # <- ParkWest pdss02 direct match
        "PDSS_RSTLSS_LEG_OL",
        "PDSS_URGE_MOVE_LIMBS_OL",
        "PDSS_DSTRSSING_DREAMS_OL",
        "PDSS_DSTRSSING_HALLUC_OL",
        "PDSS_PASS_URINE_OL",
        "PDSS_IMMOBILE_OL",
        "PDSS_PAIN_IN_LIMBS_OL",
        "PDSS_CRAMPS_IN_LIMBS_OL",
        "PDSS_PAIN_POST_OF_LIMBS_OL",
        "PDSS_TREMOR_ON_WAKE_OL",
        "PDSS_TIRED_ON_WAKE_OL",
        "PDSS_SNORING_WOKE_OL",
        # Epworth Sleepiness Scale
        "ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8",
    ],

    "Autonomic": [
        # SCOPA-AUT (urinary = items 9-13, GI = 1-7, cardiovascular = 14-16)
        "SCAU1",  "SCAU2",  "SCAU3",  "SCAU4",  "SCAU5",  "SCAU6",  "SCAU7",
        "SCAU8",  "SCAU9",  "SCAU10", "SCAU11", "SCAU12", "SCAU13",
        "SCAU14", "SCAU15", "SCAU16", "SCAU17", "SCAU18", "SCAU19",
        "SCAU20", "SCAU21", "SCAU22", "SCAU23",
    ],

    "Apathy_Fatigue": [
        "NP1FATG",   # UPDRS Part I fatigue  <- ParkWest FSS proxy
        "NP1APAT",   # UPDRS Part I apathy
    ],

    "Quality_of_Life": [
        "MSEADLG",   # Modified Schwab & England  <- ParkWest sf36pf proxy
    ],

    # No comorbidities file available in downloaded PPMI data
}

# ---------------------------------------------------------------------------
# Features where higher = better (need inversion before Cox fitting)
# Same logic as ParkWest: negate so higher = more impaired = HR > 1
# ---------------------------------------------------------------------------
INVERT_FEATURES: set[str] = {
    # MoCA: higher = better cognitive function
    "MCATOT",
    "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH",
    "MCAVF", "MCAVFNUM", "MCAABSTR",
    "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5",
    "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7", "MCASNTNC",
    "MCALION", "MCARHINO", "MCACAMEL",

    # GDS: items are reverse-scored (1 = bad for most items) — computed GDS_TOT:
    # higher GDS_TOT = more depressive symptoms = worse -> do NOT invert total
    # Individual items: for most items 1=depressed/bad, 0=ok -> higher=worse -> no inversion
    # Exception: positive items (GDSSATIS, GDSGSPIR, GDSHAPPY, GDSALIVE, GDSENRGY)
    # where answer=0 means worse. GDS_TOT already accounts for this in standard scoring.
    # -> do not invert any GDS items (standard total = higher = more depressed = worse)

    # PDSS-2: higher score = worse sleep (0=always, 4=never problem) — CHECK DIRECTION
    # PDSS-2 items: 0=severe, 4=none -> higher = better sleep -> INVERT
    "PDSS_SLEPT_WELL_OL",           # higher = slept well = better -> invert
    # Other PDSS items: higher = worse symptom -> do NOT invert
    # (PDSS_DFCLTY_STAY_ASLEEP_OL etc: higher = more difficulty = worse)

    # Schwab & England: higher = more independent = better -> invert
    "MSEADLG",
}


def apply_feature_inversions(long_df, features: list[str]) -> tuple:
    """
    Negate features in INVERT_FEATURES so that higher always means more impaired.
    Returns (modified_df, list_of_inverted_features).
    """
    import pandas as pd
    inverted = []
    for feat in features:
        if feat in INVERT_FEATURES and feat in long_df.columns:
            long_df[feat] = -long_df[feat]
            inverted.append(feat)
    if inverted:
        print(f"[Inversion] Negated {len(inverted)} features (higher=worse): {inverted}")
    return long_df, inverted
