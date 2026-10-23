"""Ranked RQ1 displays and source-verified appendix excerpts."""
from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations, product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def generate_ranked_rq1(
    root: Path, paper: Path, data: pd.DataFrame, *,
    blocks: list[str], block_codes: dict[str, list[str]],
    short: dict[str, str], triplets: list[tuple[str, str, str]],
    excerpts: list[tuple[str, str, str]],
    feature: Callable, tex: Callable,
) -> tuple[list[Path], dict]:
    """Write top-ten table, complete ranking, old stats, and appendix table."""
    ranked = []
    for selected_blocks in combinations(blocks, 3):
        for codes in product(*(block_codes[block] for block in selected_blocks)):
            columns = [feature(code, data.columns) for code in codes]
            evaluable = data[columns].notna().all(axis=1)
            hits = evaluable & data[columns].eq(1).all(axis=1)
            count, denominator = int(hits.sum()), int(evaluable.sum())
            ranked.append({
                "configuration": "+".join(codes),
                "constructs": " + ".join(short[code] for code in codes),
                "paragraphs": count,
                "evaluable_paragraphs": denominator,
                "share": count / denominator if denominator else np.nan,
                "meetings": int(data.loc[hits, "meeting_id"].nunique()),
            })
    ranked = (pd.DataFrame(ranked)
              .sort_values(["paragraphs", "configuration"],
                           ascending=[False, True], kind="stable")
              .reset_index(drop=True))
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    if len(ranked) != 710:
        raise ValueError("RQ1 triplet candidate set changed")
    audit = paper / "rq1_triplet_rankings.csv"
    ranked.to_csv(audit, index=False)
    top = ranked.head(10)

    rows = []
    for _, row in top.iterrows():
        label = (tex(row.constructs) + r"\newline\texttt{"
                 + tex(row.configuration) + "}")
        rows.append([str(int(row["rank"])), label, f"{int(row.paragraphs):,}",
                     f"{100 * row.share:.2f}\\%", str(int(row.meetings))])
    table = paper / "rq1_configurations.tex"
    lines = ["% Generated from exhaustive scoped triplet ranking; do not hand-edit.",
             r"\begin{table*}[!tbp]", r"\centering\small",
             rf"\caption{{Ten most frequent cross-block three-code configurations. Shares use jointly evaluable paragraphs (up to {len(data):,}); exact denominators are saved with the ranking. Configurations overlap; shares do not sum to 100\%. Meetings counts distinct meeting records.}}",
             r"\label{tab:rq1-configurations}",
             r"\begin{tabularx}{\textwidth}{@{}rXrrr@{}}", r"\toprule",
             "Rank & Configuration & Paragraphs & Share & Meetings \\\\",
             r"\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    table.write_text("\n".join(lines), encoding="utf-8")

    # Preserve the former three prespecified configuration estimates in
    # supplementary data; selected top-ten ranks are descriptive.
    old_stats = []
    for i, codes in enumerate(triplets):
        columns = [feature(code, data.columns) for code in codes]
        evaluable = data[columns].notna().all(axis=1)
        subset = data.loc[evaluable].copy()
        subset["hit"] = subset[columns].eq(1).all(axis=1).astype(int)
        per_speech = subset.groupby("speech_id").hit.agg(["sum", "count"]).to_numpy()
        seed = 20260913 + i
        rng = np.random.default_rng(seed)
        draws = per_speech[rng.integers(0, len(per_speech),
                                      size=(500, len(per_speech)))].sum(axis=1)
        ci = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])
        hits = subset[subset.hit.eq(1)]
        old_stats.append({
            "configuration": "+".join(codes),
            "positive_paragraphs": len(hits),
            "evaluable_paragraphs": len(subset),
            "share": len(hits) / len(subset),
            "ci_low": ci[0], "ci_high": ci[1],
            "meetings": int(hits.meeting_id.nunique()),
            "bootstrap_replicates": 500, "seed": seed,
        })
    supplementary = paper / "rq1_configuration_stats.csv"
    pd.DataFrame(old_stats).to_csv(supplementary, index=False)

    import fitz
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    evidence = pd.read_parquet(root / "prepared_data/annotations_evidence.parquet")
    # Reuse verified passages by configuration, independently of their historical rank.
    # New configurations receive a deterministic, PDF-checked evidence excerpt.
    from .galtung_research import ROOT
    existing = {name: (unit, quote) for name, unit, quote in excerpts}
    refreshed = []
    for name in top.configuration:
        if name in existing:
            refreshed.append((name, *existing[name]))
            continue
        codes = name.split("+")
        candidates = data[data[[feature(c, data.columns) for c in codes]].eq(1).all(axis=1)].sort_values("unit_id")
        found = None
        for row in candidates.itertuples():
            ev = evidence[evidence.unit_id.eq(row.unit_id) & evidence.evidence_verified.eq(1)]
            if not set(codes).issubset(set(ev.value)):
                continue
            source = Path(str(row.source_pdf))
            pdf = source if source.is_absolute() else ROOT / (source if source.parts[0] == "data" else Path("data/raw") / source)
            with fitz.open(pdf) as document:
                text = normalize(" ".join(page.get_text() for page in document))
            for quote in ev.loc[ev.value.eq(codes[0]), "evidence"]:
                if isinstance(quote, str) and normalize(quote) and normalize(quote) in normalize(row.text) and normalize(quote) in text:
                    found = (name, row.unit_id, quote)
                    break
            if found:
                break
        if found is None:
            raise ValueError(f"No PDF-verified excerpt for {name}")
        refreshed.append(found)
    excerpts = refreshed
    provenance, appendix_rows = [], []
    for rank, (name, unit_id, quote) in enumerate(excerpts, start=1):
        match = data[data.unit_id.eq(unit_id)]
        if len(match) != 1:
            raise ValueError(f"Excerpt {unit_id} absent or duplicated in RQ1 scope")
        row = match.iloc[0]
        codes = name.split("+")
        if not row[[feature(code, data.columns) for code in codes]].eq(1).all():
            raise ValueError(f"Excerpt {unit_id} lost configuration {name}")
        verified = evidence[(evidence.unit_id.eq(unit_id))
                            & evidence.evidence_verified.eq(1)].value.unique().tolist()
        if not set(codes).issubset(verified):
            raise ValueError(f"Excerpt {unit_id} lacks all three verified code evidence rows")
        source = Path(str(row.source_pdf))
        if source.is_absolute():
            pdf = source
        elif source.parts[0] == "data":
            pdf = ROOT / source
        else:
            pdf = ROOT / "data/raw" / source
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        with fitz.open(pdf) as document:
            pdf_text = " ".join(page.get_text() for page in document)
        if normalize(quote) not in normalize(row.text) or normalize(quote) not in normalize(pdf_text):
            raise ValueError(f"Excerpt {unit_id} does not match paragraph and source PDF")
        provenance.append({
            "rank": rank, "configuration": name, "unit_id": unit_id,
            "country": row.country, "date": pd.Timestamp(row.date).date().isoformat(),
            "doc_symbol": row.doc_symbol,
            "source_pdf": str(pdf.relative_to(ROOT)),
            "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "excerpt": quote, "verified_codes": codes,
            "verification": "all three selected annotation evidence rows verified; normalized exact excerpt substring in scoped paragraph and source PDF",
        })
        source_label = (tex(row.country) + r"\newline "
                        + tex(pd.Timestamp(row.date).date()) + r"\newline \texttt{"
                        + tex(row.doc_symbol) + "}")
        appendix_rows.append([
            str(rank) + r" \texttt{" + tex(name) + "}",
            source_label,
            r"\emph{\textquotedblleft " + tex(quote) + r"\textquotedblright}",
        ])
    provenance_path = paper / "rq1_excerpt_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    appendix = paper / "rq1_excerpts.tex"
    lines = ["% Generated; verified against all three code evidence rows and source PDFs.",
             r"\begin{table*}[!tbp]", r"\centering\small",
             r"\renewcommand{\arraystretch}{1.25}",
             r"\caption{Source-speech excerpts for Table~\ref{tab:rq1-configurations} ranks 1--10, checked against the original UN records. Codes apply to the full paragraph; excerpts provide reading context.}",
             r"\label{tab:rq1-excerpts}",
             r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.9in}>{\raggedright\arraybackslash}p{1.62in}X@{}}", r"\toprule",
             "Rank and codes & Country, date, UN symbol & Excerpt \\\\",
             r"\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in appendix_rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    appendix.write_text("\n".join(lines), encoding="utf-8")
    return [table, appendix, audit, supplementary, provenance_path], {
        "ranked_candidates": len(ranked),
        "observed_candidates": int(ranked.paragraphs.gt(0).sum()),
        "top_ten": top.to_dict(orient="records"),
    }
