"""Single source of truth for the analysis pipeline.

Every module reads from CONFIG. Nothing analytical is hard-coded elsewhere.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT_TABLES = OUT / "tables"
OUT_FIGURES = OUT / "figures"
OUT_DATA = OUT / "data"

CONFIG = {
    "SEED": 42,
    "MIN_PARAGRAPHS_POOLED": 30,   # country floor, pooled analysis
    "MIN_PARAGRAPHS_PHASE": 15,    # country floor, per phase
    "PSEUDOCOUNT": 0.5,            # added to counts before CLR
    "BOOTSTRAP_B": 200,
    "K_RANGE": list(range(2, 9)),
    "CONFIDENCE_SENS": 0.7,        # sensitivity threshold

    # Editable. Auto-merge any phase with < 8 meetings or < 1,500 paragraphs
    # into its neighbor and log the merge.
    "PHASES": {
        "P1_war_onset":   ("2023-10-01", "2024-01-25"),
        "P2_post_icj":    ("2024-01-26", "2025-01-18"),
        "P3_ceasefire_1": ("2025-01-19", "2025-03-17"),
        "P4_resumed_war": ("2025-03-18", "2025-10-09"),
        "P5_ceasefire_2": ("2025-10-10", "2025-12-31"),
    },
    "PHASE_MIN_MEETINGS": 8,
    "PHASE_MIN_PARAGRAPHS": 1500,

    # Event studies. Editable.
    "EVENTS": {
        "E1_icj_orders":  "2024-01-26",   # ICJ provisional measures
        "E2_ceasefire_1": "2025-01-19",   # first ceasefire takes effect
    },
    "EVENT_WINDOW_DAYS": 180,  # pre and post; post truncates at next phase boundary

    # Outcomes for trends and event studies.
    # VERIFY transformative_framing against the codebook before trusting results.
    "OUTCOMES": {
        "attribution_israel": {
            "dim": "attribution", "codes": ["ATR-ISR"],
            "denominator": "paragraphs with any attribution code"},
        "structural_violence": {
            "dim": "violence_type", "codes": ["VS"],
            "denominator": "paragraphs with any violence_type code"},
        "transformative_framing": {
            "dim": "transcend", "codes": ["T2", "T3", "T4"],
            "denominator": "all paragraphs", "verify": True},
    },
    # Robustness variant for the contested outcome (see report.md): Galtung's
    # TRANSCEND method treats compromise (T2) as distinct from transcendence.
    "OUTCOME_VARIANTS": {
        "transformative_framing": {
            "name": "transformative_framing_strict",
            "dim": "transcend", "codes": ["T3", "T4"],
            "denominator": "all paragraphs"},
    },

    # --- data source -------------------------------------------------------
    "DB_PATH": ROOT / "data" / "db" / "annotations.db",
    "CODEBOOK_DIR": ROOT / "codebooks",
    "EXCLUDE_RUN_IDS": ["pilot-main"],
    "KEEP_RUN_ID": "main",

    # --- dimension bookkeeping --------------------------------------------
    # Order fixes the column order of the fingerprint / CLR matrix.
    "DIMENSIONS": [
        "violence_type", "attribution", "positive_negative",
        "transcend", "imperialism", "relation_type",
    ],
    # Synthetic category: the paragraph carries no annotation in this dimension.
    "NONE_TOKEN": "NONE",
    # The transcend codebook was annotated by two models; time comparisons are
    # restricted to the largest single-model span (computed in discovery.py).
    "MODEL_SPLIT_DIM": "transcend",

    # --- figure style ------------------------------------------------------
    "DPI": 300,
    # Okabe-Ito colourblind-safe palette.
    "PALETTE": [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000",
    ],
    "GREY": "#8C8C8C",
}
