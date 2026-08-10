#!/usr/bin/env python3
"""Tag each speaker turn with whether it was present in the pre-rebuild corpus.

    python scripts/collect/flag_provenance.py
    python scripts/collect/flag_provenance.py --new data/interim/un_speeches_extracted_v2.csv \
        --legacy data/interim/un_speeches_extracted_legacy.csv --out data/interim/corpus.csv

The corpus was rebuilt after the Selenium collector was found to have captured 500 of 2,070
catalogue records, and to have filtered on `creation_date` (the catalogue date) rather than
the meeting date. The rebuild fetches meeting records by symbol from ODS, which has no
pagination ceiling, so it is complete by construction.

The rebuilt corpus REPRODUCES the old one exactly on every meeting the two share: 87 shared
meetings, 2,845 turns, zero differences in either direction. So this is not a merge — the new
data already contains all in-scope old data, and nothing needs to be carried across. What
this adds is provenance, so any result can be checked for dependence on newly recovered
material:

    in_legacy_corpus    this exact turn (same meeting, speaker and text) was in the old corpus
    meeting_provenance  "both"     the meeting was already held
                        "new_only" the meeting was recovered by the rebuild

Turns are matched on (doc_symbol, speaker, normalised speech text) rather than on row order,
because the rebuild changed which meetings are present and therefore every row index.

Old meetings deliberately dropped by the corpus selection rule — off-agenda meetings such as
"Protection of civilians in armed conflict" and the two pre-7-October ones — are not
represented here at all. They are absent by design, not by accident; corpus_manifest.csv
records what was selected.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW = REPO_ROOT / "data" / "interim" / "un_speeches_extracted_v2.csv"
DEFAULT_LEGACY = REPO_ROOT / "data" / "interim" / "un_speeches_extracted.csv"


def norm_symbol(s: object) -> str:
    """S/PV.9451 (Resumption 1) and S/PV.9451(Resumption1) are the same meeting."""
    return re.sub(r"[\s()]", "", str(s))


def turn_key(symbol: object, speaker: object, speech: object) -> tuple[str, str, str]:
    text = re.sub(r"\s+", " ", str(speech)).strip()
    return (norm_symbol(symbol), str(speaker).strip(),
            hashlib.sha1(text.encode()).hexdigest()[:16])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    ap.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: overwrite --new in place")
    args = ap.parse_args()

    for p in (args.new, args.legacy):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    new = pd.read_csv(args.new)
    legacy = pd.read_csv(args.legacy)

    legacy_turns = {turn_key(r.doc_symbol, r.ambassador_name, r.speech)
                    for r in legacy.itertuples()}
    legacy_meetings = {norm_symbol(s) for s in legacy.doc_symbol}

    keys = [turn_key(r.doc_symbol, r.ambassador_name, r.speech) for r in new.itertuples()]
    new["in_legacy_corpus"] = [k in legacy_turns for k in keys]
    new["meeting_provenance"] = [
        "both" if norm_symbol(s) in legacy_meetings else "new_only" for s in new.doc_symbol
    ]

    out = args.out or args.new
    new.to_csv(out, index=False)

    n_meet = new.groupby("meeting_provenance").doc_symbol.nunique()
    n_turn = new.meeting_provenance.value_counts()
    print(f"{len(new)} turns / {new.doc_symbol.nunique()} meetings -> {out}")
    for k in ("both", "new_only"):
        print(f"  {k:9s} {n_meet.get(k, 0):4d} meetings  {n_turn.get(k, 0):5d} turns")
    print(f"  turns present in the legacy corpus: {int(new.in_legacy_corpus.sum())}")

    dropped = legacy_meetings - {norm_symbol(s) for s in new.doc_symbol}
    if dropped:
        print(f"\nlegacy meetings not in the rebuilt corpus ({len(dropped)}), "
              f"excluded by the selection rule:")
        print("  " + ", ".join(sorted(dropped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
