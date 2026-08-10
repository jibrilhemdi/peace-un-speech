#!/usr/bin/env python3
"""Select the analysis corpus from the fetched meeting records.

    python scripts/collect/select_corpus.py
    python scripts/collect/select_corpus.py --rule agenda --from-date 2023-10-07 --to-date 2025-12-31
    python scripts/collect/select_corpus.py --rule agenda-or-council-terms --dry-run

fetch_meeting_records.py downloads EVERY meeting in the requested range — Ukraine, Libya,
Haiti and all — because which meetings are on the Palestine agenda is only knowable from the
documents themselves. This is the step that narrows that download to the corpus.

Selection is deliberately a separate, cheap, re-runnable step. Changing the rule costs
seconds and never re-downloads anything, so the corpus definition stays an explicit decision
rather than something buried in the collector.

RULES

  agenda (default)
      Keep a meeting only if its agenda item names Palestine — "The situation in the Middle
      East, including the Palestinian question", "Illegal Israeli actions in occupied East
      Jerusalem ...", "Question of Palestine". This is the strict, defensible rule: relevance
      is decided by what the Council/Assembly convened to discuss, not by word counts.

      Cost of the strictness, measured: it also drops S/PV.9560 and S/PV.9781, which sit
      under a "Protection of civilians in armed conflict" agenda but are overwhelmingly about
      Gaza (246 and 242 mentions). Use --rule agenda-or-council-terms to keep them.

  agenda-or-council-terms
      The above, plus Security Council meetings with >= --min-terms Palestine mentions. Picks
      up the Gaza-dominated thematic debates; also picks up Syria and Red Sea meetings where
      Gaza comes up in passing.

Selected PDFs are symlinked (not copied) into --out-dir, so the corpus costs no extra disk
and re-selecting is instant. Point extract_speech_turns.py at that directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "raw" / "meeting_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "corpus_pdfs"
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "data" / "raw" / "corpus_manifest.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    ap.add_argument("--rule", choices=["agenda", "agenda-or-council-terms"], default="agenda")
    ap.add_argument("--min-terms", type=int, default=20,
                    help="Palestine mentions required by agenda-or-council-terms")
    ap.add_argument("--from-date", default="2023-10-07")
    ap.add_argument("--to-date", default="2025-12-31")
    ap.add_argument("--dry-run", action="store_true", help="report, create no symlinks")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"no manifest at {args.manifest} — run fetch_meeting_records.py first",
              file=sys.stderr)
        return 1

    m = pd.read_csv(args.manifest)
    m = m[(m.exists == True) & m.meeting_date.notna()].copy()  # noqa: E712
    m["d"] = pd.to_datetime(m.meeting_date)
    n_fetched = len(m)

    w = m[(m.d >= args.from_date) & (m.d <= args.to_date)]
    keep = w.agenda_palestine == 1
    if args.rule == "agenda-or-council-terms":
        keep = keep | ((w.series == "S/PV") & (w.n_palestine >= args.min_terms))
    sel = w[keep].sort_values(["d", "series", "meeting_number", "resumption"])

    print(f"fetched            {n_fetched}")
    print(f"in date window     {len(w)}   ({args.from_date} .. {args.to_date})")
    print(f"selected           {len(sel)}   rule={args.rule}")
    print("\nby series:")
    for s, n in sel.groupby("series").size().items():
        print(f"  {s:14s} {n}")
    print("\nby quarter:")
    for q, n in sel.groupby(sel.d.dt.to_period("Q")).size().items():
        print(f"  {q}  {n}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()

    linked = missing = 0
    for _, r in sel.iterrows():
        src = Path(r.path)
        if not src.is_absolute():
            src = REPO_ROOT / src
        if not src.exists():
            missing += 1
            continue
        (args.out_dir / src.name).symlink_to(src)
        linked += 1

    cols = ["symbol", "series", "meeting_number", "resumption", "meeting_date", "agenda",
            "n_pages", "n_words", "n_palestine", "n_gaza", "sha256", "path"]
    sel[cols].to_csv(args.corpus_manifest, index=False)

    print(f"\nlinked {linked} PDFs into {args.out_dir}"
          + (f"  ({missing} missing from disk)" if missing else ""))
    print(f"corpus manifest: {args.corpus_manifest}")
    print(f"\nnext: python scripts/collect/extract_speech_turns.py --pdf-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
