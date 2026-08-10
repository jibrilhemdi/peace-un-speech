#!/usr/bin/env python3
"""Flatten the job store into analysis-ready tables.

    python scripts/export/build_tables.py
    python scripts/export/build_tables.py --codebook violence --run-id main
    python scripts/export/build_tables.py --formats parquet          # skip the csv/excel copies

Writes three tables into data/exports/:
  annotations_long    unit_id, codebook_id, annotation_index, dimension, value,
                      evidence, confidence  (+ provenance)
  unit_code_matrix    one row per unit, one binary column per (dimension, code)
  country_profiles    per country, the share of its non-procedural paragraphs
                      carrying each code — the salience measure for MCA and LCA

plus export_manifest.json — codebook versions, models, prompt hashes, row counts.

Each table is written once per requested format, into its own subdirectory:

  data/exports/parquet/   canonical. Preserves Int8/Float32 and the NA-vs-0 distinction.
  data/exports/csv/       same rows, openable anywhere. NA becomes an empty field.
  data/exports/excel/     one workbook, one sheet per table, frozen headers + autofilter.

Parquet is the one to analyse from; csv and excel exist to be opened and read. Round-tripping
a csv back into pandas loses the nullable-integer types, so treat them as views, not sources.

Everything is derived from the database alone. Nothing calls the API, and re-running with the
same DB produces byte-identical tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from collections.abc import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import codebook as cbmod  # noqa: E402
from lib.config import Config, model_tag  # noqa: E402
from lib.db import Store  # noqa: E402


def _put(matrix, name, col, dtype, multi_run, rid):
    """Write a column, scoped to one run when several runs share the frame.

    With one run the frame is one row per unit and the column is written whole. With several,
    the frame is one row per (unit, run) and each run fills only its own rows — otherwise the
    last codebook processed would overwrite the others' values.
    """
    if not multi_run:
        matrix[name] = col
        return
    if name not in matrix.columns:
        matrix[name] = pd.Series(pd.NA, index=matrix.index, dtype=dtype)
    matrix.loc[matrix.run_id == rid, name] = col[matrix.run_id == rid]


def col_name(codebook_id: str, dimension: str, code: str) -> str:
    return f"{codebook_id}__{dimension}__{code}"


# Hard limits of the .xlsx format itself, not of the writer.
XLSX_MAX_ROWS = 1_048_576 - 1  # one row goes to the header
XLSX_MAX_CELL = 32_767


def write_table(df: pd.DataFrame, out_dir: Path, name: str, formats: Sequence[str]) -> list[Path]:
    """Write one table as parquet and/or csv. Returns the paths written.

    Parquet is written from the frame as-is, so a nullable Int8 column stays Int8 and an
    un-annotated unit stays NA. CSV renders NA as an empty field — the same distinction, but
    only by convention, which is why parquet stays the canonical form.
    """
    written = []
    for fmt in formats:
        if fmt == "parquet":
            p = out_dir / "parquet" / f"{name}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            written.append(p)
        elif fmt == "csv":
            p = out_dir / "csv" / f"{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig: Excel on Windows reads a plain utf-8 CSV as cp1252 and mangles every
            # non-ASCII country name and quoted evidence string. The BOM is what stops that.
            df.to_csv(p, index=False, encoding="utf-8-sig")
            written.append(p)
    return written


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> tuple[Path, list[str]]:
    """Write every table into one workbook, a sheet each. Returns the path and any warnings.

    Frozen header row and an autofilter on each sheet, because the point of this file is that
    it opens and can be scrolled and filtered without writing any code.
    """
    warnings: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            out = df
            if len(out) > XLSX_MAX_ROWS:
                warnings.append(
                    f"{name}: {len(df):,} rows exceeds the .xlsx limit; the sheet holds the "
                    f"first {XLSX_MAX_ROWS:,}. Use the parquet or csv copy for the full table."
                )
                out = out.iloc[:XLSX_MAX_ROWS]
            # A single cell cannot hold more than 32,767 characters. Evidence quotes are far
            # shorter, but truncating loudly beats a corrupt workbook if one ever isn't.
            for c in out.columns:
                if out[c].dtype == object:
                    long = out[c].map(lambda v: isinstance(v, str) and len(v) > XLSX_MAX_CELL)
                    if long.any():
                        warnings.append(f"{name}.{c}: {int(long.sum())} cell(s) truncated to "
                                        f"{XLSX_MAX_CELL:,} chars")
                        out = out.copy()
                        out[c] = out[c].map(
                            lambda v: v[:XLSX_MAX_CELL] if isinstance(v, str) else v)
            out.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes(1, 0)
            if len(out.columns):
                sheet.autofilter(0, 0, max(len(out), 1), len(out.columns) - 1)
    return path, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codebook", action="append", default=None,
                    help="restrict to these codebook ids (repeatable)")
    ap.add_argument("--run-id", action="append", default=None,
                    help="repeatable. One run_id keeps the current table shape; give it twice "
                         "to export two runs side by side (e.g. --run-id main --run-id laguna), "
                         "which adds a run_id column and keys the matrix on (unit_id, run_id).")
    ap.add_argument("--include-procedural", action="store_true",
                    help="keep paragraph-level procedural units in the denominators")
    ap.add_argument("--formats", default="parquet,csv,excel",
                    help="comma-separated subset of parquet,csv,excel. Default writes all "
                         "three: parquet is canonical, csv and excel are openable copies of "
                         "the same rows.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    known = ["parquet", "csv", "excel"]
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in known]
    if bad or not formats:
        print(f"--formats: unknown {bad or 'empty'}; pick from {', '.join(known)}",
              file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    run_ids = args.run_id or ["main"]
    multi_run = len(run_ids) > 1
    db_path = cfg.path_for("db")
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1

    store = Store(db_path)
    out_dir = cfg.path_for("exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cb_dir = cfg.path_for("codebooks")

    runs = [r for r in store.codebook_runs() if r[2] in run_ids]
    if args.codebook:
        keep = set(args.codebook)
        runs = [r for r in runs if r[0] in keep]
    if not runs:
        print(f"nothing to export for run_id(s) {run_ids}", file=sys.stderr)
        store.close()
        return 1

    # Latest version per (codebook, run_id) present in the DB. Keyed by run too, so a
    # re-run under a new run_id is exported alongside the original rather than shadowing it.
    latest: dict[tuple[str, str], int] = {}
    for cb_id, version, rid in runs:
        k = (cb_id, rid)
        latest[k] = max(latest.get(k, 0), version)
    selected = [(cb_id, rid, v) for (cb_id, rid), v in latest.items()]

    units = pd.read_sql_query("SELECT * FROM units", store.conn)
    if not args.include_procedural:
        units = units[units.para_is_procedural == 0]
    units = units.reset_index(drop=True)

    manifest: dict[str, object] = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_ids": run_ids,
        "database": str(db_path),
        "include_procedural": args.include_procedural,
        "codebooks": {},
    }

    long_frames: list[pd.DataFrame] = []
    matrix = units[["unit_id"]].copy()
    matrix_cols: list[str] = []
    coverage: dict[str, pd.Series] = {}

    key_cols = ["unit_id", "run_id"] if multi_run else ["unit_id"]
    if multi_run:
        matrix = (units[["unit_id"]].merge(pd.DataFrame({"run_id": run_ids}), how="cross")
                  .reset_index(drop=True))

    for cb_id, rid, version in sorted(selected):
        try:
            cb = cbmod.require_codebook(cb_dir, cb_id)
        except cbmod.CodebookError as exc:
            print(f"skipping {cb_id}: {exc}", file=sys.stderr)
            continue

        params = (cb_id, version, rid)
        ann = pd.read_sql_query(
            """SELECT unit_id, codebook_id, codebook_version, run_id, annotation_index,
                      dimension, value, evidence, confidence, evidence_verified
               FROM annotations
               WHERE codebook_id=? AND codebook_version=? AND run_id=?""",
            store.conn, params=params,
        )
        done = pd.read_sql_query(
            """SELECT unit_id, model, prompt_hash FROM jobs
               WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'""",
            store.conn, params=params,
        )
        done_ids = set(done.unit_id)
        done["model_tag"] = done.model.map(model_tag)
        # The model is known at request time and recorded per job — never inferred from the
        # response. Carrying it into every export is what lets a later re-run on a different
        # model sit beside this one and be told apart.
        unit_model = done.set_index("unit_id").model
        unit_tag = done.set_index("unit_id").model_tag
        ann["model"] = ann.unit_id.map(unit_model)
        ann["model_tag"] = ann.unit_id.map(unit_tag)

        if version != cb.version:
            print(
                f"NOTE: exporting {cb_id} v{version} from the DB; codebooks/{cb_id}.yaml is "
                f"now v{cb.version}. The export reflects what was actually run.",
                file=sys.stderr,
            )

        long_frames.append(ann)

        # -- binary matrix --------------------------------------------------------------
        for dim in cb.dimensions:
            codes = list(dim.codes) + (["NONE"] if dim.allows_none() else [])
            sub = ann[ann.dimension == dim.name]

            # An ordinal dimension also gets its numeric score, so the scale can enter a
            # regression directly while the same annotation still enters an MCA as categories.
            # Taken from the lowest annotation_index when a unit somehow carries several.
            if dim.is_ordinal:
                score_map = {c.code: c.value for c in dim.categories}
                scored = sub[sub.value != "NONE"].copy()
                scored["_score"] = scored.value.map(score_map)
                # A paragraph may carry several positions on the scale. How they collapse to
                # one number is a declared choice (dimension.score_aggregation), not a silent
                # default — these are "how far along" scales, so max is the default.
                agg = {"max": "max", "min": "min", "mean": "mean",
                       "first": "first"}[dim.score_aggregation]
                if dim.score_aggregation == "first":
                    per_unit = (scored.sort_values("annotation_index")
                                .drop_duplicates("unit_id").set_index("unit_id")._score)
                else:
                    per_unit = scored.groupby("unit_id")._score.agg(agg)
                name = f"{cb_id}__{dim.name}__score"
                col = matrix.unit_id.map(per_unit)
                col = col.where(matrix.unit_id.isin(done_ids), other=pd.NA)
                dtype = "Float32" if dim.score_aggregation == "mean" else "Int8"
                _put(matrix, name, col.astype(dtype), dtype, multi_run, rid)
                if name not in matrix_cols:
                    matrix_cols.append(name)

            for code in codes:
                name = col_name(cb_id, dim.name, code)
                hit = set(sub[sub.value == code].unit_id)
                col = matrix.unit_id.isin(hit).astype("int8")
                # A unit that was never successfully annotated is NA, not zero — a missing
                # label and an absent code are different things and must not be conflated.
                col = col.where(matrix.unit_id.isin(done_ids), other=pd.NA)
                _put(matrix, name, col.astype("Int8"), "Int8", multi_run, rid)
                if name not in matrix_cols:
                    matrix_cols.append(name)

        tag_col = f"{cb_id}__model_tag"
        if multi_run:
            sel = matrix.run_id == rid
            if tag_col not in matrix.columns:
                matrix[tag_col] = pd.NA
            matrix.loc[sel, tag_col] = matrix.loc[sel, "unit_id"].map(unit_tag)
        else:
            matrix[tag_col] = matrix.unit_id.map(unit_tag)
        if tag_col not in matrix_cols:
            matrix_cols.append(tag_col)

        mkey = f"{cb_id}@{rid}" if multi_run else cb_id
        scope = (matrix.run_id == rid) if multi_run else pd.Series(True, index=matrix.index)
        coverage[mkey] = matrix.unit_id.isin(done_ids) & scope

        manifest["codebooks"][f"{cb_id}@{rid}" if multi_run else cb_id] = {  # type: ignore[index]
            "codebook_id": cb_id,
            "run_id": rid,
            "version_run": version,
            "version_on_disk": cb.version,
            "prompt_hash_on_disk": cbmod.prefix_hash(cb),
            "prompt_hashes_used": sorted({h for h in done.prompt_hash.dropna().unique()}),
            "models_used": sorted({m for m in done.model.dropna().unique()}),
            "model_tags": sorted({t for t in done.model_tag.dropna().unique()}),
            "units_annotated": int(len(done_ids)),
            "annotations": int(len(ann.groupby(["unit_id", "annotation_index"])) if len(ann) else 0),
            "dimensions": {d.name: list(d.codes) for d in cb.dimensions},
        }

    if not long_frames:
        print("nothing exported", file=sys.stderr)
        store.close()
        return 1

    # -- annotations_long -------------------------------------------------------------
    long = pd.concat(long_frames, ignore_index=True)
    long = long.merge(
        units[["unit_id", "speech_id", "country", "bloc", "is_party", "meeting_date"]],
        on="unit_id", how="left",
    )
    long["evidence_verified"] = long.evidence_verified.astype(bool)
    long = long.sort_values(["codebook_id", "unit_id", "annotation_index", "dimension"])

    # -- unit_code_matrix ---------------------------------------------------------------
    meta_cols = ["speech_id", "country", "bloc", "is_party", "meeting_date", "role",
                 "language", "n_words"]
    matrix = matrix.merge(units[["unit_id", *meta_cols]], on="unit_id", how="left")
    lead = ["unit_id"] + (["run_id"] if multi_run else [])
    matrix = matrix[[*lead, *meta_cols, *matrix_cols]]

    # -- country_profiles ---------------------------------------------------------------
    sparse_min = int(cfg.get("export.sparse_country_min_units", 30))
    have_country = matrix[matrix.country.notna()].copy()
    profiles = []
    # With several runs exported, group by (country, run_id). Grouping by country alone would
    # average two models into one profile row — the precise thing the model columns exist to
    # prevent.
    group_keys = ["country", "run_id"] if multi_run else ["country"]
    for gk, grp in have_country.groupby(group_keys, sort=True):
        gk = gk if isinstance(gk, tuple) else (gk,)
        row: dict[str, object] = {
            "country": gk[0],
            "bloc": grp.bloc.dropna().iloc[0] if grp.bloc.notna().any() else None,
            "is_party": int(grp.is_party.max()),
            **({"run_id": gk[1]} if multi_run else {}),
            "n_speeches": int(grp.speech_id.nunique()),
            "n_units": int(len(grp)),
        }
        for name in matrix_cols:
            col = grp[name]
            if name.endswith("__model_tag"):
                # Which model produced this country's labels for that codebook. Usually one;
                # "a+b" when a codebook was run across a model switch, which is exactly the
                # case this column exists to make visible.
                tags = sorted({t for t in col.dropna().unique()})
                row[name.replace("__model_tag", "__models")] = "+".join(tags) if tags else None
                continue
            if name.endswith("__score"):
                # Mean score among the country's paragraphs that were actually scored.
                scored = col.dropna()
                row[f"n_scored__{name}"] = int(len(scored))
                row[name] = float(scored.mean()) if len(scored) else None
                continue
            denom = int(col.notna().sum())
            # Share of the country's ANNOTATED paragraphs carrying the code. Units that were
            # never annotated are excluded from both numerator and denominator, so a
            # half-finished run gives an honest rate rather than a diluted one.
            row[f"n_annotated__{name}"] = denom
            row[name] = float(col.sum() / denom) if denom else None
        # NB: not named "sparse" — that collides with pandas' .sparse accessor.
        row["is_sparse"] = bool(row["n_units"] < sparse_min)
        profiles.append(row)

    prof = pd.DataFrame(profiles)

    # -- write every table in every requested format -------------------------------------
    tables = {
        "annotations_long": long,
        "unit_code_matrix": matrix,
        "country_profiles": prof,
    }
    written: list[Path] = []
    for name, df in tables.items():
        written += write_table(df, out_dir, name, formats)
    xlsx_warnings: list[str] = []
    if "excel" in formats:
        xlsx_path, xlsx_warnings = write_workbook(
            tables, out_dir / "excel" / "un_speeches_annotations.xlsx")
        written.append(xlsx_path)

    manifest["formats"] = formats
    manifest["rows"] = {
        "annotations_long": int(len(long)),
        "unit_code_matrix": int(len(matrix)),
        "country_profiles": int(len(prof)),
        "matrix_columns": matrix_cols,
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    shape = {
        "annotations_long": f"{len(long):,} rows",
        "unit_code_matrix": f"{len(matrix):,} rows x {len(matrix_cols)} code columns",
        "country_profiles": f"{len(prof):,} countries",
    }
    root = out_dir.parent
    for p in written:
        note = shape.get(p.stem, "3 sheets" if p.suffix == ".xlsx" else "")
        print(f"wrote {p.relative_to(root)}".ljust(52) + note)
    print(f"wrote {manifest_path.relative_to(root)}")
    for w in xlsx_warnings:
        print(f"  ! {w}", file=sys.stderr)
    print()
    for cb_id, info in manifest["codebooks"].items():  # type: ignore[union-attr]
        cov = coverage.get(cb_id)
        denom = int((matrix.run_id == info["run_id"]).sum()) if multi_run else len(matrix)
        pct = (cov.sum() / denom * 100) if cov is not None and denom else 0
        print(f"  {cb_id} v{info['version_run']}: {info['units_annotated']:,} units "
              f"({pct:.1f}% coverage), models {info['models_used']}, "
              f"prompt {info['prompt_hashes_used']}")
        if len(info["prompt_hashes_used"]) > 1 or len(info["models_used"]) > 1:
            print("    ! mixed prompt/model within this codebook — see scripts/annotate/report_status.py")
    if prof["is_sparse"].any():
        n = int(prof["is_sparse"].sum())
        print(f"\n  {n} country/countries flagged sparse (<{sparse_min} annotated units). "
              "Flagged, not dropped — the choice is yours.")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
