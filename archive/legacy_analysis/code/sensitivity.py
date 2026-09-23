"""Sensitivity: re-run Steps 1-3 under alternative filters and compare typologies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .clustering import ward_labels
from .config import CONFIG
from .fingerprints import build_fingerprints, clr
from .io_utils import LOG, write_data, write_table


def variants(corpus) -> dict:
    """Each variant returns (units, annotations, dim_codes, description)."""
    ann, ms = corpus.ann, corpus.member_states
    thr = CONFIG["CONFIDENCE_SENS"]
    span = corpus.model_span

    v = {}
    v["(a) confidence >= %.2f" % thr] = dict(
        units=ms, ann=ann[ann.confidence.notna() & (ann.confidence >= thr)],
        dim_codes=corpus.dim_codes,
        desc=f"annotations with confidence >= {thr} "
             f"({int((ann.confidence >= thr).sum())} of {len(ann)}; "
             f"{int(ann.confidence.isna().sum())} rows have no confidence and are dropped)")

    v["(b) evidence verified only"] = dict(
        units=ms, ann=ann[ann.evidence_verified == 1], dim_codes=corpus.dim_codes,
        desc=f"annotations whose evidence span verified against the paragraph "
             f"({int((ann.evidence_verified == 1).sum())} of {len(ann)})")

    if span["split"]:
        sub = ms[ms.in_model_span]
        keep_units = set(sub.unit_id)
        a = ann[~((ann.dimension == CONFIG["MODEL_SPLIT_DIM"])
                  & (~ann.unit_id.isin(keep_units)))]
        v["(c) transcend single-model span"] = dict(
            units=ms, ann=a, dim_codes=corpus.dim_codes,
            desc=f"transcend annotations restricted to the "
                 f"{span['model'].split('/')[-1]} span "
                 f"({span['start'].date()}..{span['end'].date()}); all other dimensions "
                 "unchanged, so the country set is preserved")

    # (d) is not in the brief: it tests a redundancy found in the data, namely that
    # attribution's absence category is identical to violence_type's because
    # attribution is conditional_on violence_type.
    dc = {k: list(x) for k, x in corpus.dim_codes.items()}
    absent = CONFIG["NONE_TOKEN"] + "_ABSENT"
    if absent in dc.get("attribution", []):
        dc["attribution"] = [c for c in dc["attribution"] if c != absent]
        v["(d) drop duplicated attribution absence"] = dict(
            units=ms, ann=ann, dim_codes=dc,
            desc="attribution's synthetic absence column removed; it is identical to "
                 "violence_type's in every paragraph because attribution is "
                 "conditional_on violence_type, so keeping both double-weights that "
                 "single feature in the Euclidean distance")
    return v


def run(corpus, fp, cl) -> dict:
    print("\n[Sensitivity]")
    base_countries = list(fp["clr"].index)
    base_labels = cl["labels"]
    k = cl["k"]
    rows = []

    for name, spec in variants(corpus).items():
        shares, counts, n = build_fingerprints(
            spec["units"], spec["ann"], spec["dim_codes"],
            CONFIG["MIN_PARAGRAPHS_POOLED"])
        X = clr(counts, spec["dim_codes"])
        lab, _ = ward_labels(X, k)
        common = [c for c in base_countries if c in set(X.index)]
        a = [base_labels[base_countries.index(c)] for c in common]
        b = [lab[list(X.index).index(c)] for c in common]
        ari = adjusted_rand_score(a, b)
        agree = float(np.mean(np.array(a) == np.array(b)))
        # Labels are arbitrary: report the best-matching agreement too.
        best_agree = max(agree, 1 - agree) if k == 2 else agree
        rows.append({
            "Variant": name,
            "Countries": len(X),
            "Shared with main": len(common),
            "ARI vs main": ari,
            "Same partition": best_agree,
            "Verdict": ("typology holds" if ari >= 0.8 else
                        "largely holds" if ari >= 0.6 else
                        "MATERIALLY DIFFERENT"),
            "_desc": spec["desc"],
        })
        LOG.log("Sensitivity",
                f"{name}: {len(X)} countries, ARI vs main = {ari:.3f} "
                f"({best_agree:.0%} of shared countries in the matching cluster) — "
                f"{rows[-1]['Verdict']}.",
                spec["desc"])

    tab = pd.DataFrame(rows)
    write_data(tab.rename(columns={"_desc": "description"}), "sensitivity.csv")
    notes = ("Each variant re-runs Steps 1-3 end to end: fingerprints, per-block CLR, "
             "Ward linkage and a cut at the main solution's k"
             f"={k}. ARI is computed on the countries present in both the variant and "
             "the main analysis. Variant definitions: "
             + " ".join(f"{r['Variant']} = {r['_desc']}." for _, r in tab.iterrows()))
    write_table(tab.drop(columns=["_desc"]), "table_sensitivity.tex",
                caption="Sensitivity of the typology to annotation-quality filters and "
                        "to the annotation-model split.",
                label="tab:sensitivity", notes=notes,
                align="l" + "r" * 4 + "l")
    return {"table": tab}
