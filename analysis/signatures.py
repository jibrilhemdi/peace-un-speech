"""Step 4: cluster signatures, DRAFT labels, and the stratified reading sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG
from .fingerprints import ABSENT, code_label, unit_code_matrix
from .io_utils import LOG, write_data, write_table

N_OVER, N_UNDER = 6, 3
SAMPLE_PER_CELL = 20


def signatures(shares: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster over/under-representation: ratio to corpus mean plus a v-test.

    The v-test is the standardised difference of a cluster mean from the overall
    mean under sampling without replacement, i.e. the criterion FactoMineR uses
    to describe a partition.
    """
    N = len(shares)
    overall_mean = shares.mean(axis=0)
    overall_var = shares.var(axis=0, ddof=0)
    rows = []
    for g in np.unique(labels):
        m = labels == g
        ng = int(m.sum())
        gm = shares[m].mean(axis=0)
        denom = np.sqrt(((N - ng) / (N - 1)) * (overall_var / ng))
        v = (gm - overall_mean) / denom.replace(0, np.nan)
        for col in shares.columns:
            rows.append({
                "cluster": int(g), "code": col,
                "cluster_share": float(gm[col]),
                "corpus_share": float(overall_mean[col]),
                "ratio": float(gm[col] / overall_mean[col]) if overall_mean[col] else np.nan,
                "v_test": float(v[col]) if pd.notna(v[col]) else np.nan,
                "n_countries": ng,
            })
    return pd.DataFrame(rows)


def draft_label(sig_g: pd.DataFrame, code_labels: dict) -> str:
    """One-line DRAFT description assembled from the strongest signature codes."""
    over = sig_g.nlargest(4, "v_test")
    under = sig_g.nsmallest(2, "v_test")
    fmt = lambda d: ", ".join(
        (f"{c.split('__')[0].split('_')[0]}:none" if c.split("__", 1)[1] == ABSENT
         else c.split("__", 1)[1])
        for c in d.code)
    return f"high {fmt(over)}; low {fmt(under)}"


def reading_sample(corpus, assign, sig, seed: int) -> pd.DataFrame:
    """Stratified quote sample: cluster x phase, drawn from signature codes only."""
    rng = np.random.default_rng(seed)
    ms = corpus.member_states
    ucm = unit_code_matrix(ms, corpus.ann, corpus.dim_codes)
    cl_of = assign.set_index("country").cluster

    ann = corpus.ann
    # Per-paragraph code strings for every dimension, plus evidence/confidence.
    per_dim = {}
    for dim in CONFIG["DIMENSIONS"]:
        s = (ann[ann.dimension == dim].groupby("unit_id").value
             .agg(lambda v: "|".join(sorted(set(v)))))
        per_dim[dim] = s
    conf = ann.groupby("unit_id").confidence.mean()
    ver = ann.groupby("unit_id").evidence_verified.min()
    evid = (ann.dropna(subset=["evidence"]).sort_values(["unit_id", "annotation_index"])
            .groupby("unit_id").evidence.agg(lambda v: " || ".join(list(v)[:3])))

    out = []
    for g in sorted(sig.cluster.unique()):
        top_codes = list(sig[sig.cluster == g].nlargest(N_OVER, "v_test").code)
        top_codes = [c for c in top_codes if not c.endswith(ABSENT)]
        countries = [c for c in cl_of.index if cl_of[c] == g]
        pool_all = ms[ms.country.isin(countries)]
        for ph in list(corpus.phases) + ["outside"]:
            pool = pool_all[pool_all.phase == ph]
            if pool.empty:
                continue
            hit = ucm.loc[pool.unit_id, top_codes].sum(axis=1) > 0 if top_codes else None
            eligible = pool[hit.to_numpy()] if hit is not None else pool
            if eligible.empty:
                continue
            take = min(SAMPLE_PER_CELL, len(eligible))
            pick = eligible.iloc[rng.choice(len(eligible), size=take, replace=False)]
            for r in pick.itertuples():
                row = {"cluster": f"C{g}", "phase": ph, "country": r.country,
                       "iso3": r.iso3, "date": r.date.date(), "meeting_id": r.meeting_id,
                       "speech_id": r.speech_id, "unit_id": r.unit_id,
                       "speaker_name": r.speaker_name}
                for dim in CONFIG["DIMENSIONS"]:
                    row[dim] = per_dim[dim].get(r.unit_id, "")
                row["confidence_mean"] = round(float(conf.get(r.unit_id, np.nan)), 3)
                row["evidence_verified_all"] = int(ver.get(r.unit_id, 1))
                row["evidence_quote"] = evid.get(r.unit_id, "")
                row["paragraph_text"] = r.text
                row["signature_codes_matched"] = "|".join(
                    c for c in top_codes if ucm.at[r.unit_id, c] == 1)
                out.append(row)
    return pd.DataFrame(out)


def run(corpus, fp, cl) -> dict:
    print("\n[Step 4] signatures and reading samples")
    shares, labels = fp["shares"], cl["labels"]
    sig = signatures(shares, labels)
    sig["label"] = [code_label(corpus.code_labels, c) for c in sig.code]
    write_data(sig.sort_values(["cluster", "v_test"], ascending=[True, False]),
               "cluster_signatures.csv")

    drafts = {}
    rows = []
    for g in sorted(sig.cluster.unique()):
        s = sig[sig.cluster == g]
        drafts[g] = draft_label(s, corpus.code_labels)
        for kind, d in (("over", s.nlargest(N_OVER, "v_test")),
                        ("under", s.nsmallest(N_UNDER, "v_test"))):
            for r in d.itertuples():
                short = (f"{r.code.split('__')[0]}: none"
                         if r.code.split("__", 1)[1] == ABSENT else r.code.split("__", 1)[1])
                rows.append({
                    "Cluster": f"C{g}",
                    "Direction": "over" if kind == "over" else "under",
                    "Code": short,
                    "Label": corpus.code_labels.get(
                        (r.code.split("__")[0], r.code.split("__", 1)[1]), ""),
                    "Cluster share": r.cluster_share,
                    "Corpus share": r.corpus_share,
                    "Ratio": r.ratio,
                    "v-test": r.v_test,
                })
    tab = pd.DataFrame(rows)
    write_table(tab, "table_cluster_signatures.tex",
                caption="Cluster signatures: the "
                        f"{N_OVER} most over-represented and {N_UNDER} most "
                        "under-represented codes per cluster, against the mean across all "
                        "member states in the analysis population.",
                label="tab:cluster_signatures",
                notes="Shares are the mean across countries of the share of a country's "
                      "paragraphs carrying the code. Ratio is cluster mean / corpus mean. "
                      "The v-test is the standardised difference of the cluster mean from "
                      "the corpus mean under sampling without replacement; |v| > 1.96 "
                      "corresponds to the conventional 5% threshold. 'none' rows are the "
                      "synthetic absence category for that dimension.",
                align="llll" + "r" * 4)

    dl = pd.DataFrame({
        "Cluster": [f"C{g}" for g in drafts],
        "Countries": [int((labels == g).sum()) for g in drafts],
        "DRAFT label": [f"DRAFT: {v}" for v in drafts.values()],
    })
    write_table(dl, "table_cluster_draft_labels.tex",
                caption="Draft interpretive labels for each cluster, generated "
                        "mechanically from the signature codes.",
                label="tab:cluster_draft_labels",
                notes="THESE LABELS ARE DRAFTS. They are assembled from the strongest "
                      "signature codes and have not been checked against the text. They "
                      "must be revised after human close reading of "
                      "outputs/data/quotes_reading_sample.csv before use in any writeup.",
                align="lrp{9.5cm}", float_fmt="{:.0f}")
    LOG.log("Step 4",
            "DRAFT cluster labels: " + "; ".join(f"C{g}: {v}" for g, v in drafts.items()),
            "Generated mechanically from signature codes and explicitly marked DRAFT; they "
            "are placeholders for human close reading, not findings.")

    qs = reading_sample(corpus, cl["assign"], sig, CONFIG["SEED"])
    write_data(qs, "quotes_reading_sample.csv")
    cells = qs.groupby(["cluster", "phase"]).size()
    LOG.log("Step 4",
            f"Reading sample: {len(qs)} paragraphs across {len(cells)} cluster-by-phase "
            f"cells (target {SAMPLE_PER_CELL} per cell, drawn with seed "
            f"{CONFIG['SEED']} from paragraphs carrying at least one of the cluster's top "
            f"{N_OVER} signature codes).",
            "This file is for human close reading only; nothing downstream computes on it.")
    return {"signatures": sig, "draft_labels": drafts, "reading_sample": qs}
