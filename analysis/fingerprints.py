"""Step 1: country fingerprints, and the CLR transform used from Step 2 onward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG
from .io_utils import LOG, iso3, write_data, write_table

ABSENT = CONFIG["NONE_TOKEN"] + "_ABSENT"


def code_columns(dim_codes: dict) -> list[str]:
    """Ordered `dimension__code` column names spanning all six blocks."""
    return [f"{d}__{c}" for d in CONFIG["DIMENSIONS"] for c in dim_codes[d]]


def unit_code_matrix(units: pd.DataFrame, ann: pd.DataFrame,
                     dim_codes: dict) -> pd.DataFrame:
    """One binary row per paragraph: does it carry this code?

    Multiple annotations on one paragraph collapse to presence. The synthetic
    `*__NONE_ABSENT` column marks a paragraph with no annotation in that block.
    """
    idx = pd.Index(units.unit_id, name="unit_id")
    a = ann[ann.unit_id.isin(set(units.unit_id))]
    out = {}
    for dim in CONFIG["DIMENSIONS"]:
        sub = a[a.dimension == dim]
        present = pd.Series(False, index=idx)
        for code in dim_codes[dim]:
            if code == ABSENT:
                continue
            hit = idx.isin(set(sub.loc[sub.value == code, "unit_id"]))
            out[f"{dim}__{code}"] = hit.astype(np.int8)
            present |= hit
        out[f"{dim}__{ABSENT}"] = (~present.to_numpy()).astype(np.int8)
    return pd.DataFrame(out, index=idx)


def country_counts(units: pd.DataFrame, ucm: pd.DataFrame,
                   group_col: str = "country") -> pd.DataFrame:
    """Aggregate the unit-code matrix to counts per country (or country-phase)."""
    key = units.set_index("unit_id")[group_col]
    return ucm.groupby(key.reindex(ucm.index)).sum()


def build_fingerprints(units: pd.DataFrame, ann: pd.DataFrame, dim_codes: dict,
                       floor: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (shares, counts, n_paragraphs) for countries at or above `floor`."""
    ucm = unit_code_matrix(units, ann, dim_codes)
    counts = country_counts(units, ucm)
    n = units.groupby("country").size().rename("n_paragraphs")
    counts = counts.reindex(n.index).fillna(0).astype(int)
    keep = n[n >= floor].index
    counts, n = counts.loc[keep], n.loc[keep]
    shares = counts.div(n, axis=0)
    cols = [c for c in code_columns(dim_codes) if c in counts.columns]
    return shares[cols], counts[cols], n


def clr(counts: pd.DataFrame, dim_codes: dict,
        pseudocount: float | None = None) -> pd.DataFrame:
    """Centered log-ratio *within each dimension block*, blocks concatenated.

    Closing each block separately keeps the six Galtung dimensions on their own
    simplex, so a country that talks a lot about violence does not mechanically
    depress its imperialism profile.
    """
    p = CONFIG["PSEUDOCOUNT"] if pseudocount is None else pseudocount
    out = []
    for dim in CONFIG["DIMENSIONS"]:
        cols = [f"{dim}__{c}" for c in dim_codes[dim] if f"{dim}__{c}" in counts.columns]
        x = np.log(counts[cols].to_numpy(dtype=float) + p)
        out.append(pd.DataFrame(x - x.mean(axis=1, keepdims=True),
                                index=counts.index, columns=cols))
    return pd.concat(out, axis=1)


def outcome_flags(units: pd.DataFrame, ann: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Paragraph-level outcome and denominator membership for a CONFIG outcome.

    Returns `y` (carries any of the outcome codes) and `in_denom` (belongs in the
    denominator defined by the spec).
    """
    dim = spec["dim"]
    sub = ann[(ann.dimension == dim) & (ann.unit_id.isin(set(units.unit_id)))]
    hit = set(sub.loc[sub.value.isin(spec["codes"]), "unit_id"])
    any_code = set(sub.unit_id)
    d = pd.DataFrame({"unit_id": units.unit_id.to_numpy()})
    d["y"] = d.unit_id.isin(hit).astype(int)
    denom = str(spec.get("denominator", "all paragraphs")).lower()
    d["in_denom"] = (d.unit_id.isin(any_code) if "with any" in denom
                     else pd.Series(True, index=d.index))
    return d


# ------------------------------------------------------------------ outputs --

def code_label(labels: dict, col: str) -> str:
    dim, code = col.split("__", 1)
    if code == ABSENT:
        return f"{dim}: none"
    return f"{code} {labels.get((dim, code), '')}".strip()


def write_fingerprint_outputs(corpus, shares, counts, n) -> None:
    d = shares.copy()
    d.columns = [f"share__{c}" for c in d.columns]
    k = counts.copy()
    k.columns = [f"count__{c}" for c in k.columns]
    out = pd.concat([n.rename("n_paragraphs"), d, k], axis=1).reset_index()
    out.insert(1, "iso3", out.country.map(iso3))
    out = out.sort_values("n_paragraphs", ascending=False)
    write_data(out, "country_fingerprints.csv")

    # --- corpus overview by phase ----------------------------------------
    u = corpus.units
    ms = corpus.member_states
    rows = []
    for name in list(corpus.phases) + (["outside"] if (u.phase == "outside").any() else []):
        sub, msub = u[u.phase == name], ms[ms.phase == name]
        s, e = (corpus.phases[name] if name in corpus.phases
                else (sub.date.min(), sub.date.max()))
        rows.append({
            "Phase": name.replace("_", " "),
            "Start": str(s.date()), "End": str(e.date()),
            "Meetings": sub.meeting_id.nunique(),
            "Speeches": sub.speech_id.nunique(),
            "Paragraphs": len(sub),
            "MS paragraphs": len(msub),
            "Speaking countries": msub.country.nunique(),
            "Countries above floor": int(
                (msub.groupby("country").size() >= CONFIG["MIN_PARAGRAPHS_PHASE"]).sum()),
        })
    rows.append({
        "Phase": "All", "Start": str(u.date.min().date()), "End": str(u.date.max().date()),
        "Meetings": u.meeting_id.nunique(), "Speeches": u.speech_id.nunique(),
        "Paragraphs": len(u), "MS paragraphs": len(ms),
        "Speaking countries": ms.country.nunique(),
        "Countries above floor": int(
            (ms.groupby("country").size() >= CONFIG["MIN_PARAGRAPHS_POOLED"]).sum()),
    })
    write_table(
        pd.DataFrame(rows), "table_corpus_overview.tex",
        caption="Corpus composition by phase. Paragraphs are non-procedural paragraphs "
                "from the main annotation run; MS paragraphs are those spoken by member "
                "states in national capacity.",
        label="tab:corpus_overview",
        notes=f"Country floor: {CONFIG['MIN_PARAGRAPHS_POOLED']} paragraphs pooled, "
              f"{CONFIG['MIN_PARAGRAPHS_PHASE']} per phase. The 'outside' row covers "
              "meetings predating the first phase window; they are retained in pooled "
              "analyses only.",
        float_fmt="{:.0f}")

    # --- headline shares for the ten highest-volume states ----------------
    head = [
        ("violence_type__VD", "VD"), ("violence_type__VS", "VS"),
        ("attribution__ATR-ISR", "ATR-ISR"), ("attribution__ATR-PAL", "ATR-PAL"),
        ("positive_negative__PN3", "PN3"), ("transcend__T3", "T3"),
        ("imperialism__I1", "I1"), ("relation_type__SYM", "SYM"),
    ]
    top = n.sort_values(ascending=False).head(10).index
    t = pd.DataFrame({"Country": [c for c in top], "ISO3": [iso3(c) for c in top],
                      "Paragraphs": [int(n[c]) for c in top]})
    for col, short in head:
        t[short] = [shares.loc[c, col] for c in top]
    legend = "; ".join(
        f"{s} = {corpus.code_labels.get((c.split('__')[0], s), s)}" for c, s in head)
    write_table(
        t, "table_fingerprints_summary.tex",
        caption="Headline code shares for the ten highest-volume member states. Each "
                "cell is the share of that country's paragraphs carrying the code.",
        label="tab:fingerprints_summary",
        notes=f"Codes: {legend}. Shares within a dimension can exceed one in sum because a "
              "paragraph may carry several codes; the full fingerprint including the "
              "explicit none category is in outputs/data/country_fingerprints.csv.",
        float_fmt="{:.2f}")


def run(corpus) -> dict:
    print("\n[Step 1] country fingerprints")
    ms = corpus.member_states
    floor = CONFIG["MIN_PARAGRAPHS_POOLED"]
    shares, counts, n = build_fingerprints(ms, corpus.ann, corpus.dim_codes, floor)
    dropped = ms.country.nunique() - len(n)
    LOG.log("Step 1",
            f"{len(n)} of {ms.country.nunique()} member states clear the pooled floor of "
            f"{floor} paragraphs ({dropped} dropped, covering "
            f"{len(ms) - int(n.sum())} paragraphs).",
            "Countries below the floor have fingerprints too noisy to place on the map.")

    # Parties: reference points, computed but never clustered.
    parties = corpus.parties
    pshares, pcounts, pn = build_fingerprints(parties, corpus.ann, corpus.dim_codes, 1)
    LOG.log("Step 1",
            f"Party reference fingerprints computed for {list(pn.index)} "
            "and held out of every clustering step.")

    write_fingerprint_outputs(corpus, shares, counts, n)
    X = clr(counts, corpus.dim_codes)
    pX = clr(pcounts, corpus.dim_codes)
    return {"shares": shares, "counts": counts, "n": n, "clr": X,
            "party_shares": pshares, "party_counts": pcounts, "party_n": pn,
            "party_clr": pX}
