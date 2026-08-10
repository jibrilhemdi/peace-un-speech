#!/usr/bin/env python3
"""Dump annotations.db to CSV and Excel, so the job store can be opened and read.

    python scripts/export/dump_database.py
    python scripts/export/dump_database.py --table annotations --table units
    python scripts/export/dump_database.py --formats csv --with-responses

Writes into data/exports/db/:
  csv/<table>.csv          one file per table in the database, verbatim
  excel/annotations_db.xlsx  the same tables, one sheet each

and one derived sheet that is not in the database at all:
  annotations_with_text    every annotation next to the paragraph it labels, with the
                           speaker's country, the meeting date, the model's evidence quote
                           and its confidence

That derived sheet is the one to open when the question is "what did the model actually say
about this paragraph". The verbatim dumps are for auditing and for re-running counts outside
Python.

Two columns are dropped by default because they are machine plumbing, not data, and together
they are most of the file size: jobs.response_json (the raw model output, ~95 MB) and
jobs.error. Pass --with-responses to keep them.

Read-only. This never writes to the database.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config import Config  # noqa: E402

# Every table in the schema, in the order a reader would want to meet them.
TABLES = ["units", "annotations", "jobs", "pilots", "quota", "provider_caps"]

# Excel's own limits, not the writer's.
XLSX_MAX_ROWS = 1_048_576 - 1
XLSX_MAX_CELL = 32_767

# Dropped from the jobs dump unless --with-responses. response_json is ~95 MB of raw model
# output; error is a stack-trace-ish string that only matters when debugging a failed run.
HEAVY_JOB_COLS = ["response_json", "error"]

READABLE_SQL = """
SELECT a.codebook_id,
       a.dimension,
       a.value,
       u.country,
       u.meeting_date,
       u.speech_id,
       u.para_index,
       u.text            AS paragraph,
       a.evidence,
       a.confidence,
       a.evidence_verified,
       a.annotation_index,
       a.unit_id,
       a.run_id,
       a.codebook_version,
       u.bloc,
       u.is_party,
       u.ambassador_name,
       u.role,
       u.language,
       u.n_words
FROM annotations a
JOIN units u ON u.unit_id = a.unit_id
WHERE a.run_id = ?
ORDER BY u.meeting_date, u.speech_id, u.para_index, a.codebook_id, a.annotation_index
"""


def read_table(conn: sqlite3.Connection, table: str, with_responses: bool) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608 — table is from TABLES
    if table == "jobs" and not with_responses:
        df = df.drop(columns=[c for c in HEAVY_JOB_COLS if c in df.columns])
    return df


def write_csvs(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        p = out_dir / f"{name}.csv"
        # utf-8-sig, or Excel on Windows reads the file as cp1252 and mangles every accented
        # country name and every non-ASCII quotation mark in the evidence.
        df.to_csv(p, index=False, encoding="utf-8-sig")
        written.append(p)
    return written


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> tuple[Path, list[str]]:
    """One workbook, one sheet per table, frozen header and autofilter on each."""
    notes: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            out = df
            if len(out) > XLSX_MAX_ROWS:
                notes.append(f"{name}: {len(df):,} rows exceeds the .xlsx row limit; the sheet "
                             f"holds the first {XLSX_MAX_ROWS:,}. The CSV has all of them.")
                out = out.iloc[:XLSX_MAX_ROWS]
            for c in out.columns:
                if out[c].dtype == object:
                    long = out[c].map(lambda v: isinstance(v, str) and len(v) > XLSX_MAX_CELL)
                    if long.any():
                        notes.append(f"{name}.{c}: {int(long.sum())} cell(s) truncated to "
                                     f"{XLSX_MAX_CELL:,} chars")
                        out = out.copy()
                        out[c] = out[c].map(
                            lambda v: v[:XLSX_MAX_CELL] if isinstance(v, str) else v)
            sheet_name = name[:31]
            out.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            if len(out.columns):
                sheet.autofilter(0, 0, max(len(out), 1), len(out.columns) - 1)
            # Paragraph and evidence are long prose; give them room and let them wrap rather
            # than spilling across forty columns of the reader's screen.
            for i, c in enumerate(out.columns):
                if c in ("paragraph", "text", "evidence"):
                    sheet.set_column(i, i, 80, writer.book.add_format({"text_wrap": True}))
                elif c.endswith("_id") or c in ("ambassador_name", "country"):
                    sheet.set_column(i, i, 22)
    return path, notes


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", default=None,
                    help=f"restrict to these tables (repeatable). Known: {', '.join(TABLES)}")
    ap.add_argument("--run-id", default="main",
                    help="run_id for the annotations_with_text sheet (default: main)")
    ap.add_argument("--formats", default="csv,excel",
                    help="comma-separated subset of csv,excel (default: both)")
    ap.add_argument("--with-responses", action="store_true",
                    help="keep jobs.response_json and jobs.error — adds ~95 MB")
    ap.add_argument("--no-readable", action="store_true",
                    help="skip the annotations_with_text sheet")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in ("csv", "excel")]
    if bad or not formats:
        print(f"--formats: unknown {bad or 'empty'}; pick from csv, excel", file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    db_path = cfg.path_for("db")
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1

    wanted = list(args.table) if args.table else list(TABLES)
    unknown = [t for t in wanted if t not in TABLES]
    if unknown:
        print(f"--table: unknown {unknown}; known tables are {', '.join(TABLES)}",
              file=sys.stderr)
        return 2

    # Read-only so a dump can never be the thing that corrupts a store that took days to fill.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [t for t in wanted if t not in present]
        if missing:
            print(f"not in this database, skipping: {', '.join(missing)}", file=sys.stderr)
        tables = {t: read_table(conn, t, args.with_responses) for t in wanted if t in present}
        if not tables:
            print("nothing to dump", file=sys.stderr)
            return 1

        readable = None
        if not args.no_readable and "annotations" in tables and "units" in present:
            readable = pd.read_sql_query(READABLE_SQL, conn, params=(args.run_id,))
            readable["evidence_verified"] = readable.evidence_verified.astype(bool)
            if readable.empty:
                print(f"note: no annotations for run_id '{args.run_id}' — "
                      "annotations_with_text is empty", file=sys.stderr)
    finally:
        conn.close()

    sheets = dict(tables)
    if readable is not None:
        # First, because it is the sheet a reader actually wants.
        sheets = {"annotations_with_text": readable, **tables}

    out_dir = cfg.path_for("exports") / "db"
    written: list[Path] = []
    notes: list[str] = []
    if "csv" in formats:
        written += write_csvs(sheets, out_dir / "csv")
    if "excel" in formats:
        p, notes = write_workbook(sheets, out_dir / "excel" / "annotations_db.xlsx")
        written.append(p)

    root = cfg.path_for("exports").parent
    for p in written:
        rows = "" if p.suffix == ".xlsx" else f"{len(sheets[p.stem]):,} rows"
        label = f"{len(sheets)} sheets" if p.suffix == ".xlsx" else rows
        print(f"wrote {p.relative_to(root)}".ljust(50) + label)
    for n in notes:
        print(f"  ! {n}", file=sys.stderr)
    if not args.with_responses and "jobs" in sheets:
        print("\n  jobs: response_json and error omitted — pass --with-responses to keep them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
