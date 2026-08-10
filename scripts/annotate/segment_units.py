#!/usr/bin/env python3
"""Turn the speech corpus into stable annotation units.

    python scripts/annotate/segment_units.py
    python scripts/annotate/segment_units.py --dry-run          # report, write nothing
    python scripts/annotate/segment_units.py --allow-id-change  # accept that existing unit_ids will change

Unit IDs are sha256(speech_id + para_index + normalised_text) and must survive
re-segmentation, so annotations produced under codebook 1 are not orphaned when segmentation
is re-run before codebook 4. If a re-run would change or drop IDs that already carry
annotations, this refuses to write without --allow-id-change.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config import REPO_ROOT, Config, ProceduralRule  # noqa: E402
from lib.db import Store  # noqa: E402

OUTPUT_COLUMNS = [
    "unit_id", "speech_id", "para_index", "text", "country", "ambassador_name",
    "role", "language", "meeting_date", "bloc", "is_party", "para_is_procedural", "n_words",
]


# --------------------------------------------------------------------------------------
# Normalisation and IDs
# --------------------------------------------------------------------------------------
def normalise_for_hash(text: str, opts: dict[str, Any]) -> str:
    t = text
    if opts.get("normalise_quotes", True):
        t = t.replace("’", "'").replace("‘", "'")
        t = t.replace("“", '"').replace("”", '"')
    if opts.get("collapse_whitespace", True):
        t = " ".join(t.split())
    if opts.get("strip", True):
        t = t.strip()
    return t


def unit_id(speech_id: str, para_index: int, normalised: str, length: int) -> str:
    h = hashlib.sha256()
    h.update(speech_id.encode())
    h.update(b"\x00")
    h.update(str(para_index).encode())
    h.update(b"\x00")
    h.update(normalised.encode())
    return h.hexdigest()[:length]


def classify_paragraphs(paragraphs: list[str], rules: list[ProceduralRule]) -> list[tuple[bool, str]]:
    """(is_procedural, rule_name) per paragraph. Rules see the previous verdict, so a quoted
    passage can inherit the procedural introduction that preceded it."""
    out: list[tuple[bool, str]] = []
    prev = False
    for p in paragraphs:
        hit = ""
        for rule in rules:
            if rule.fires(p, prev):
                hit = rule.name
                break
        prev = bool(hit)
        out.append((prev, hit))
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def histogram(counts: dict[str, int], *, width: int = 44, top: int | None = None,
              label: str = "country") -> list[str]:
    if not counts:
        return ["  (none)"]
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = items[:top] if top else items
    hi = max(v for _, v in shown) or 1
    pad = max(len(k) for k, _ in shown)
    lines = []
    for k, v in shown:
        bar = "█" * max(1, round(v / hi * width))
        lines.append(f"  {k:<{pad}}  {v:>5}  {bar}")
    if top and len(items) > top:
        rest = items[top:]
        plural = label if label.endswith("s") else (
            label[:-1] + "ies" if label.endswith("y") else label + "s"
        )
        lines.append(f"  ... {len(rest)} more {plural}, {sum(v for _, v in rest)} speeches "
                     f"(min {rest[-1][1]}, median {sorted(v for _, v in rest)[len(rest)//2]})")
    return lines


def bucket_histogram(values: list[int], edges: list[int], *, width: int = 44) -> list[str]:
    if not values:
        return ["  (none)"]
    labels, counts = [], []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
        labels.append(f"{lo}-{hi}" if hi is not None else f"{lo}+")
        counts.append(sum(1 for v in values if v >= lo and (hi is None or v <= hi)))
    top = max(counts) or 1
    pad = max(len(x) for x in labels)
    return [
        f"  {lab:>{pad}}  {c:>5}  {'█' * max(0, round(c / top * width))}"
        for lab, c in zip(labels, counts)
    ]


def describe(series: pd.Series, name: str) -> str:
    q = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return (
        f"  {name}: n={len(series)}  mean={series.mean():.2f}  min={int(series.min())}  "
        f"p10={int(q[0.1])}  p25={int(q[0.25])}  median={int(q[0.5])}  p75={int(q[0.75])}  "
        f"p90={int(q[0.9])}  p99={int(q[0.99])}  max={int(series.max())}"
    )


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--allow-id-change", action="store_true",
                    help="proceed even though existing unit_ids would change")
    ap.add_argument("--top-countries", type=int, default=30,
                    help="rows in the speeches-per-country histogram (0 = all)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    corpus_path = cfg.path_for("corpus")
    units_path = cfg.path_for("units")
    db_path = cfg.path_for("db")

    if not corpus_path.exists():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 1

    cols = cfg.get("corpus.columns", {})
    seg = cfg.section("segmentation")
    rules = cfg.procedural_rules()
    blocs = cfg.bloc_map()
    min_words = int(seg.get("min_words", 15))
    id_len = int(seg.get("unit_id_length", 16))
    splitter = re.compile(str(seg.get("paragraph_split", r"\n\s*\n")))
    norm_opts = seg.get("normalise", {}) or {}
    party = set(cfg.get("corpus.party_countries", []) or [])

    df = pd.read_csv(corpus_path)
    n_all = len(df)

    # -- row filter ---------------------------------------------------------------------
    for col, want in (cfg.get("corpus.include_when", {}) or {}).items():
        if col not in df.columns:
            print(f"corpus.include_when references missing column '{col}'", file=sys.stderr)
            return 1
        col_vals = df[col].astype(bool) if df[col].dtype == bool else df[col]
        df = df[col_vals == want]
    excluded_roles = set(cfg.get("corpus.exclude_roles", []) or [])
    if excluded_roles:
        df = df[~df[cols.get("role", "role")].isin(excluded_roles)]

    print(f"corpus            {corpus_path.relative_to(REPO_ROOT)}")
    print(f"rows              {n_all:,} total -> {len(df):,} after row filter")
    if excluded_roles:
        print(f"                  (excluded roles: {', '.join(sorted(excluded_roles))})")

    # -- segment ------------------------------------------------------------------------
    text_col = cols.get("text", "speech")
    rows: list[dict[str, Any]] = []
    rule_hits: Counter[str] = Counter()
    paras_per_speech: list[int] = []
    dropped_short = 0
    dropped_empty = 0

    for rec in df.to_dict("records"):
        speech_id = f"{rec[cols.get('doc_symbol', 'doc_symbol')]}#{rec[cols.get('turn_index', 'turn_index')]}"
        raw_paras = [p for p in splitter.split(str(rec[text_col])) if p.strip()]
        verdicts = classify_paragraphs([p.strip() for p in raw_paras], rules)

        kept = 0
        for idx, (raw, (is_proc, rule)) in enumerate(zip(raw_paras, verdicts)):
            text = raw.strip()
            if not text:
                dropped_empty += 1
                continue
            n_words = len(text.split())
            if n_words < min_words:
                dropped_short += 1
                continue
            if is_proc:
                rule_hits[rule] += 1
            country = rec.get(cols.get("country", "country"))
            country = None if pd.isna(country) else str(country)
            rows.append(
                {
                    "unit_id": unit_id(speech_id, idx, normalise_for_hash(text, norm_opts), id_len),
                    "speech_id": speech_id,
                    "para_index": idx,
                    "text": text,
                    "country": country,
                    "ambassador_name": _s(rec.get(cols.get("speaker", "ambassador_name"))),
                    "role": _s(rec.get(cols.get("role", "role"))),
                    "language": _s(rec.get(cols.get("language", "language"))),
                    "meeting_date": _s(rec.get(cols.get("date", "meeting_date"))),
                    "bloc": blocs.get(country) if country else None,
                    "is_party": int(country in party) if country else 0,
                    "para_is_procedural": int(is_proc),
                    "n_words": n_words,
                }
            )
            kept += 1
        paras_per_speech.append(kept)

    units = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if units.empty:
        print("no units produced — check corpus.include_when and segmentation.min_words",
              file=sys.stderr)
        return 1

    # -- ID stability -------------------------------------------------------------------
    new_ids = set(units.unit_id)
    if len(new_ids) != len(units):
        dupes = units[units.unit_id.duplicated(keep=False)]
        print(f"\nFATAL: {len(units) - len(new_ids)} duplicate unit_ids "
              f"({dupes.speech_id.nunique()} speeches affected).", file=sys.stderr)
        print("Two paragraphs hashed identically. This should be impossible given that the "
              "hash includes speech_id and para_index — inspect the corpus.", file=sys.stderr)
        return 1

    old_ids: set[str] = set()
    annotated_ids: set[str] = set()
    if units_path.exists():
        old_ids = set(pd.read_parquet(units_path, columns=["unit_id"]).unit_id)
    if db_path.exists():
        with Store(db_path) as st:
            old_ids |= st.existing_unit_ids()
            annotated_ids = {
                r[0] for r in st.conn.execute("SELECT DISTINCT unit_id FROM jobs WHERE status='ok'")
            }

    lost = old_ids - new_ids
    gained = new_ids - old_ids
    orphaned = lost & annotated_ids

    n_speeches_in = len(df)
    n_speeches_out = units.speech_id.nunique()
    print(f"units             {len(units):,} kept")
    print(f"                  {dropped_short:,} dropped below {min_words} words, "
          f"{dropped_empty:,} empty")
    print(f"speeches          {n_speeches_out:,}", end="")
    if n_speeches_out < n_speeches_in:
        print(f"  ({n_speeches_in - n_speeches_out} lost entirely — every paragraph fell "
              f"below the {min_words}-word floor)")
    else:
        print()
    print(f"countries         {units.country.nunique():,} "
          f"({int(units.country.isna().sum()):,} units with no country)")
    print(f"procedural paras  {int(units.para_is_procedural.sum()):,} flagged "
          f"({units.para_is_procedural.mean() * 100:.1f}%)")

    if old_ids:
        print(f"\nID stability      {len(old_ids & new_ids):,} unchanged, {len(lost):,} lost, "
              f"{len(gained):,} new")
        if orphaned:
            print()
            print("!" * 78)
            print(f"! {len(orphaned):,} unit_ids that ALREADY CARRY ANNOTATIONS would disappear.")
            print("! Those annotations would be orphaned: still in the DB, joined to nothing.")
            print("!")
            by_cb = {}
            if db_path.exists():
                with Store(db_path) as st:
                    qmarks = ",".join("?" * min(len(orphaned), 900))
                    sample = list(orphaned)[:900]
                    for r in st.conn.execute(
                        f"""SELECT codebook_id, codebook_version, COUNT(DISTINCT unit_id)
                            FROM jobs WHERE status='ok' AND unit_id IN ({qmarks})
                            GROUP BY codebook_id, codebook_version""",
                        sample,
                    ):
                        by_cb[f"{r[0]} v{r[1]}"] = int(r[2])
            for k, v in sorted(by_cb.items()):
                print(f"!   {k}: {v:,} annotated units affected")
            print("!")
            print("! Cause is almost always a change to segmentation.min_words,")
            print("! segmentation.normalise, or the paragraph_split regex in config.yaml.")
            print("! Revert that change, or re-run with --allow-id-change to accept the loss.")
            print("!" * 78)
            if not args.allow_id_change:
                print("\nRefusing to write. Nothing has been changed.", file=sys.stderr)
                return 2
            print("\n--allow-id-change given: proceeding despite orphaned annotations.")
        elif lost:
            print(f"                  {len(lost):,} lost IDs carry no annotations — harmless")

    # -- distributions ------------------------------------------------------------------
    per_speech = units.groupby("speech_id").size()
    print()
    print("Paragraphs per speech (after filtering)")
    print(describe(per_speech, "units/speech"))
    print()
    print(*bucket_histogram(per_speech.tolist(), [1, 3, 5, 8, 11, 15, 21, 31]), sep="\n")

    print()
    print("Words per unit")
    print(describe(units.n_words, "words/unit"))

    speeches_per_country = (
        units[units.country.notna()].groupby("country").speech_id.nunique().to_dict()
    )
    print()
    print(f"Speeches per country ({len(speeches_per_country)} countries)")
    counts = sorted(speeches_per_country.values())
    if counts:
        med = counts[len(counts) // 2]
        print(f"  total {sum(counts):,}  median {med}  "
              f"min {counts[0]}  max {counts[-1]}  "
              f"countries with <5 speeches: {sum(1 for c in counts if c < 5)}  "
              f"with <10: {sum(1 for c in counts if c < 10)}")
        print()
        print(*histogram(speeches_per_country, top=args.top_countries or None), sep="\n")

    if units.bloc.notna().any():
        print()
        print("Units per bloc")
        print(*histogram(units.groupby("bloc").size().to_dict(), label="bloc"), sep="\n")

    if rule_hits:
        print()
        print("Paragraph-level procedural rule hits")
        for name, n in rule_hits.most_common():
            print(f"  {name:<28} {n:>6}")
        print("  Audit these in data/interim/units.parquet: units[units.para_is_procedural == 1]")

    unmapped = sorted(set(units[units.bloc.isna() & units.country.notna()].country))
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} country/countries not in the bloc map "
              f"(bloc will be null): {', '.join(unmapped)}")

    # -- write --------------------------------------------------------------------------
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    units_path.parent.mkdir(parents=True, exist_ok=True)
    units.to_parquet(units_path, index=False)
    with Store(db_path) as st:
        st.replace_units(units.to_dict("records"))

    print(f"\nwrote {units_path}  ({len(units):,} rows)")
    print(f"wrote units table in {db_path}")
    return 0


def _s(v: Any) -> str | None:
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
