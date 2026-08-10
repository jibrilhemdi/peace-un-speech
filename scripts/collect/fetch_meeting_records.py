#!/usr/bin/env python3
"""Fetch Security Council verbatim meeting records by symbol, straight from ODS.

    python scripts/collect/fetch_meeting_records.py --from-date 2023-10-07 --to-date 2025-12-31
    python scripts/collect/fetch_meeting_records.py --start 9431 --end 10070
    python scripts/collect/fetch_meeting_records.py --from-date 2023-10-07 --dry-run

WHY THIS EXISTS. The previous collector drove digitallibrary.un.org/search with Selenium.
That path has three defects that cannot be worked around from the client side:

  1. Its pagination stops updating past jrec~500, so no query returning more than ~500
     records can ever be walked to the end. The first run collected 500 of 2,070.
  2. /search sits behind an AWS WAF JS challenge, which is the only reason a real browser
     was needed at all.
  3. Its date filter is `creation_date`, the CATALOGUE date, not the meeting date. Records
     lag their meeting by months (S/PV.9400 met 2023-08-21, catalogued 2023-12-20), so the
     filter silently shifts the corpus window earlier by roughly a quarter.

None of that applies here. Security Council meetings are numbered consecutively, and ODS
serves any document by symbol:

    https://documents.un.org/api/symbol/access?s=S/PV.9439&l=E

That endpoint is not behind the WAF, needs no browser, and has no result-set ceiling. So
instead of asking a search engine which meetings exist and paging through its answer, this
enumerates the meeting numbers directly and asks for each one. Coverage is complete by
construction: a meeting can only be missed if its number is outside the requested range.

Verified byte-identical: S/PV.9400 fetched here has the same sha256 as the copy the old
collector downloaded through the browser.

A nonexistent symbol returns 200 with an HTML stub rather than 404, so existence is decided
by Content-Type, not status code. Resumed meetings use the unspaced form the API accepts,
`S/PV.9963(Resumption2)` — the spaced form the catalogue displays is rejected.

The date and agenda in the manifest are read from the PDF itself, so selection never depends
on catalogue metadata again. Meetings are filtered on the TRUE meeting date.

Resumable: an existing manifest is reloaded and already-fetched symbols are skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from threading import Lock

import fitz  # PyMuPDF
import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "meetings"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "raw" / "meeting_manifest.csv"

ODS = "https://documents.un.org/api/symbol/access"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# The Security Council agenda item this project is about. Meetings are also tagged with the
# PALESTINE QUESTION subject heading in the catalogue, which is broader than the agenda item
# — it also covers "The situation in the Middle East" and a handful of thematic debates. The
# manifest therefore records the agenda verbatim and the Palestine/Gaza term counts, and
# leaves the final inclusion call to classify_meetings.py rather than hard-coding it here.
AGENDA_PALESTINE = re.compile(r"Palestin", re.I)
AGENDA_MIDDLE_EAST = re.compile(r"situation in the Middle East", re.I)
TERM_PALESTINE = re.compile(r"\bPalestin\w*", re.I)
TERM_GAZA = re.compile(r"\bGaza\w*", re.I)

MAX_RESUMPTIONS = 6
# A run of consecutive absent meeting numbers means we are past the last meeting held.
STOP_AFTER_CONSECUTIVE_MISSES = 12


@dataclass
class Meeting:
    symbol: str
    series: str
    meeting_number: int
    resumption: int
    exists: bool
    meeting_date: str = ""
    agenda: str = ""
    n_pages: int = 0
    n_words: int = 0
    n_palestine: int = 0
    n_gaza: int = 0
    agenda_palestine: int = 0
    agenda_middle_east: int = 0
    sha256: str = ""
    n_bytes: int = 0
    path: str = ""
    error: str = ""


def symbol_for(series: str, number: int, resumption: int) -> str:
    """ODS accepts the unspaced resumption form only.

    `series` is the symbol stem without the meeting number: "S/PV" for the Security Council,
    "A/ES-10/PV" for the Emergency Special Session on Palestine, "A/78/PV" for a regular
    General Assembly session. All are served by the same endpoint.
    """
    base = f"{series}.{number}"
    return base if resumption == 0 else f"{base}(Resumption{resumption})"


def filename_for(series: str, number: int, resumption: int) -> str:
    stem = series.replace("/", "_")
    base = f"{stem}.{number}"
    return f"{base}.pdf" if resumption == 0 else f"{base}_Resumption{resumption}.pdf"


def fetch(client: httpx.Client, symbol: str, tries: int = 4) -> tuple[bytes | None, str]:
    """Return (pdf_bytes, error). pdf_bytes is None when the symbol does not exist.

    ODS answers a nonexistent symbol with 200 + text/html, so the content type decides.
    """
    last = ""
    for attempt in range(tries):
        try:
            r = client.get(ODS, params={"s": symbol, "l": "E"}, timeout=60.0)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and "application/pdf" in ctype:
                return r.content, ""
            if r.status_code == 200:
                return None, ""  # genuine "no such symbol"
            last = f"http {r.status_code}"
        except Exception as e:  # network flake, not a verdict on existence
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    return None, last or "exhausted retries"


DATE_RE = re.compile(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+20\d\d)\b")


def parse_pdf(data: bytes) -> tuple[str, str, int, int, int, int]:
    """Return (meeting_date, agenda, n_pages, n_words, n_palestine, n_gaza)."""
    doc = fitz.open(stream=data, filetype="pdf")
    # str() is a no-op at runtime — PyMuPDF's stub types get_text() as a union whatever the
    # mode, so this is only here to keep the type checker quiet.
    first: str = str(doc[0].get_text("text"))
    full: str = "\n".join(str(p.get_text("text")) for p in doc)

    m = DATE_RE.search(first)
    meeting_date = ""
    if m:
        try:
            meeting_date = datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
        except ValueError:
            meeting_date = ""

    # Two layouts. Security Council verbatims head the block with a bare "Agenda"; General
    # Assembly ones use "Agenda item 5 (continued)". Both put the item title on the next line.
    a = re.search(r"Agenda(?:\s+item\s+\d+)?(?:\s*\([^)]*\))?\s*\n(.{0,400})", first, re.S)
    agenda = ""
    if a:
        agenda = re.sub(r"\s+", " ", a.group(1))
        # Whatever follows the item title is boilerplate, the draft-resolution list, or the
        # start of debate — none of it is part of the agenda item.
        agenda = re.split(r"This record contains|\*Reissued|Draft resolution|Draft amendment|"
                          r"The President|Report of the Secretary-General|Letter dated",
                          agenda)[0].strip(" .")

    return (meeting_date, agenda, doc.page_count, len(full.split()),
            len(TERM_PALESTINE.findall(full)), len(TERM_GAZA.findall(full)))


def process(client: httpx.Client, series: str, number: int, resumption: int, out_dir: Path,
            dry_run: bool) -> Meeting:
    symbol = symbol_for(series, number, resumption)
    data, err = fetch(client, symbol)
    if data is None:
        return Meeting(symbol, series, number, resumption, exists=False, error=err)

    rec = Meeting(symbol, series, number, resumption, exists=True,
                  sha256=hashlib.sha256(data).hexdigest(), n_bytes=len(data))
    try:
        (rec.meeting_date, rec.agenda, rec.n_pages, rec.n_words,
         rec.n_palestine, rec.n_gaza) = parse_pdf(data)
    except Exception as e:
        rec.error = f"parse failed: {type(e).__name__}: {e}"

    rec.agenda_palestine = int(bool(AGENDA_PALESTINE.search(rec.agenda)))
    rec.agenda_middle_east = int(bool(AGENDA_MIDDLE_EAST.search(rec.agenda)))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / filename_for(series, number, resumption)
        p.write_bytes(data)
        # Repo-relative when it can be, so the manifest stays portable; absolute otherwise
        # (a --out-dir outside the repo is legitimate for test runs).
        rec.path = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
    return rec


def load_manifest(path: Path) -> dict[str, Meeting]:
    if not path.exists():
        return {}
    out: dict[str, Meeting] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row.setdefault("series", "S/PV")
            row["meeting_number"] = int(row["meeting_number"])
            row["resumption"] = int(row["resumption"])
            row["exists"] = row["exists"] in ("True", "true", "1")
            for k in ("n_pages", "n_words", "n_palestine", "n_gaza",
                      "agenda_palestine", "agenda_middle_east", "n_bytes"):
                row[k] = int(row[k] or 0)
            out[row["symbol"]] = Meeting(**row)
    return out


def write_manifest(path: Path, records: dict[str, Meeting]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records.values(), key=lambda r: (r.series, r.meeting_number, r.resumption))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    tmp.replace(path)


def reparse(manifest_path: Path) -> int:
    """Re-derive the parsed columns from the PDFs already on disk.

    Fetching is the expensive, rate-limited half; parsing rules change far more often than
    the documents do. This re-reads every stored PDF and rewrites the manifest in place, so
    a fix to the agenda regex costs seconds rather than another full download.
    """
    records = load_manifest(manifest_path)
    if not records:
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 1

    changed = missing = 0
    for rec in records.values():
        if not rec.exists or not rec.path:
            continue
        p = Path(rec.path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            missing += 1
            continue
        before = (rec.agenda, rec.meeting_date)
        try:
            (rec.meeting_date, rec.agenda, rec.n_pages, rec.n_words,
             rec.n_palestine, rec.n_gaza) = parse_pdf(p.read_bytes())
            rec.error = ""
        except Exception as e:
            rec.error = f"parse failed: {type(e).__name__}: {e}"
            continue
        rec.agenda_palestine = int(bool(AGENDA_PALESTINE.search(rec.agenda)))
        rec.agenda_middle_east = int(bool(AGENDA_MIDDLE_EAST.search(rec.agenda)))
        if (rec.agenda, rec.meeting_date) != before:
            changed += 1

    write_manifest(manifest_path, records)
    got = [r for r in records.values() if r.exists]
    print(f"reparsed {len(got)} documents, {changed} changed"
          + (f", {missing} PDFs missing from disk" if missing else ""))
    print(f"  agenda names the Palestinian question: "
          f"{sum(r.agenda_palestine for r in got)}")
    print(f"  agenda is 'situation in the Middle East': "
          f"{sum(r.agenda_middle_east and not r.agenda_palestine for r in got)}")
    print(f"  no agenda parsed: {sum(1 for r in got if not r.agenda)}")
    print(f"manifest: {manifest_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="S/PV",
                    help="symbol stem without the meeting number, e.g. S/PV, A/ES-10/PV, A/78/PV")
    ap.add_argument("--start", type=int, default=9431,
                    help="first S/PV meeting number (9431 = 9 Oct 2023)")
    ap.add_argument("--end", type=int, default=None,
                    help="last meeting number; default: run until --to-date is passed")
    ap.add_argument("--from-date", default=None, help="keep meetings on/after this ISO date")
    ap.add_argument("--to-date", default=None, help="stop once meetings pass this ISO date")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests (be polite)")
    ap.add_argument("--batch", type=int, default=40, help="meeting numbers per batch")
    ap.add_argument("--dry-run", action="store_true", help="probe and report, write no PDFs")
    ap.add_argument("--force", action="store_true", help="refetch symbols already in manifest")
    ap.add_argument("--reparse", action="store_true",
                    help="re-read the PDFs already on disk and rewrite the manifest, "
                         "downloading nothing. Use after changing the parsing rules.")
    args = ap.parse_args()

    if args.reparse:
        return reparse(args.manifest)

    to_date = date.fromisoformat(args.to_date) if args.to_date else None
    records = {} if args.force else load_manifest(args.manifest)
    if records:
        print(f"resuming: {len(records)} symbols already in {args.manifest}")

    lock = Lock()
    client = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True)

    number = args.start
    consecutive_misses = 0
    stop = False

    while not stop:
        batch = list(range(number, number + args.batch))
        if args.end is not None:
            batch = [n for n in batch if n <= args.end]
            if not batch:
                break

        def do(n: int) -> list[Meeting]:
            out = []
            sym = symbol_for(args.series, n, 0)
            if sym in records and not args.force:
                out.append(records[sym])
                base_exists = records[sym].exists
            else:
                rec = process(client, args.series, n, 0, args.out_dir, args.dry_run)
                out.append(rec)
                base_exists = rec.exists
            if base_exists:
                for k in range(1, MAX_RESUMPTIONS + 1):
                    s = symbol_for(args.series, n, k)
                    if s in records and not args.force:
                        r = records[s]
                    else:
                        r = process(client, args.series, n, k, args.out_dir, args.dry_run)
                    out.append(r)
                    if not r.exists:
                        break
            return out

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for group in ex.map(do, batch):
                for rec in group:
                    with lock:
                        records[rec.symbol] = rec

        for n in batch:
            base = records.get(symbol_for(args.series, n, 0))
            if base is None:
                continue
            if not base.exists:
                consecutive_misses += 1
                if consecutive_misses >= STOP_AFTER_CONSECUTIVE_MISSES:
                    print(f"stopping: {consecutive_misses} consecutive absent meeting numbers "
                          f"at {args.series}.{n} — past the last meeting held")
                    stop = True
                    break
            else:
                consecutive_misses = 0
                if to_date and base.meeting_date and date.fromisoformat(base.meeting_date) > to_date:
                    print(f"stopping: {args.series}.{n} is {base.meeting_date}, past --to-date {to_date}")
                    stop = True
                    break

        if records:
            write_manifest(args.manifest, records)
        got = [r for r in records.values() if r.exists]
        print(f"  through {args.series}.{batch[-1]}: {len(got)} documents, "
              f"{sum(r.agenda_palestine or r.agenda_middle_east for r in got)} on the ME/Palestine agenda")
        number = batch[-1] + 1

    client.close()
    if not records:
        print("nothing fetched", file=sys.stderr)
        return 1

    write_manifest(args.manifest, records)
    got = [r for r in records.values() if r.exists]
    dated = [r for r in got if r.meeting_date]
    in_window = dated
    if args.from_date:
        fd = date.fromisoformat(args.from_date)
        in_window = [r for r in in_window if date.fromisoformat(r.meeting_date) >= fd]
    if to_date:
        in_window = [r for r in in_window if date.fromisoformat(r.meeting_date) <= to_date]

    print("\n" + "=" * 78)
    print(f"documents fetched      {len(got)}")
    print(f"  with a parsed date   {len(dated)}")
    if dated:
        print(f"  meeting date range   {min(r.meeting_date for r in dated)} -> "
              f"{max(r.meeting_date for r in dated)}")
    print(f"in requested window    {len(in_window)}")
    print(f"  ME/Palestine agenda  {sum(r.agenda_palestine or r.agenda_middle_east for r in in_window)}")
    errs = [r for r in records.values() if r.error]
    if errs:
        print(f"\n! {len(errs)} symbols errored (network, not absence) — re-run to retry:")
        for r in errs[:10]:
            print(f"    {r.symbol}: {r.error}")
    print(f"\nmanifest: {args.manifest}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
