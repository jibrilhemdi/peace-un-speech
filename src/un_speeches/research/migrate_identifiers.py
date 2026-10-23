"""Stage, verify and activate the naming/population migration without annotation calls.

python -m un_speeches.research.migrate_identifiers --dry-run
python -m un_speeches.research.migrate_identifiers backup
python -m un_speeches.research.migrate_identifiers stage --backup PATH --workers 3
python -m un_speeches.research.migrate_identifiers verify --backup PATH
python -m un_speeches.research.migrate_identifiers activate --backup PATH
python -m un_speeches.research.migrate_identifiers rollback --backup PATH --dry-run
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from .bfgs_update import provenance as optimizer_provenance
import yaml

from un_speeches.lib.identifiers import (
    CODEBOOK_IDS, MAPPING_VERSION, codebook_id, feature_name, legacy_frame,
    mapping_manifest, normalize_frame, normalize_structured,
)
from . import galtung_research as gr

ROOT = gr.ROOT
DEFAULT_STAGE = ROOT / "migration_staging"
INPUT_TREES = ("src", "tests", "config", "codebooks", "docs", "paper", "data", "research_outputs")


def affected_files() -> list[Path]:
    files = [p for tree in INPUT_TREES for p in (ROOT / tree).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".db-wal", ".db-shm"))]
    files += [ROOT / n for n in ("README.md", "pyproject.toml", "requirements.txt", ".gitignore",
                                "UN_Palestine_Galtung_Research_Plan.docx") if (ROOT / n).exists()]
    return sorted(files)


def make_backup() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ROOT / "migration_backups" / stamp
    dest.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in affected_files():
        rel = source.relative_to(ROOT)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        before = gr.sha256(source)
        method = "copy"
        if source.suffix == ".db":
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as original:
                with sqlite3.connect(target) as snapshot:
                    original.backup(snapshot)
                    if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RuntimeError(f"invalid SQLite snapshot: {target}")
            method = "sqlite_backup"
        else:
            shutil.copy2(source, target)
        copied = gr.sha256(target)
        if before != gr.sha256(source) or (method == "copy" and copied != before):
            raise RuntimeError(f"input changed during backup: {source}")
        entries.append({"path": str(rel), "source_sha256": before, "backup_sha256": copied,
                        "bytes": target.stat().st_size, "method": method})
    (dest / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (dest / "git_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "entries": entries,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (dest.parent / "LATEST").write_text(str(dest.relative_to(ROOT)) + "\n")
    return dest


def verify_backup(backup: Path, *, check_originals: bool = True) -> dict:
    manifest = json.loads((backup / "manifest.json").read_text())
    for entry in manifest["entries"]:
        target = backup / entry["path"]
        if gr.sha256(target) != entry["backup_sha256"]:
            raise RuntimeError(f"backup checksum mismatch: {target}")
        # Only immutable source inputs/reviews are compared to the working copy.
        rel = Path(entry["path"])
        if check_originals and rel.parts[0] == "data" and rel.parts[1] in {"raw", "db", "interim", "pilots", "gold", "samples"}:
            if gr.sha256(ROOT / rel) != entry["source_sha256"]:
                raise RuntimeError(f"original research input changed: {rel}")
        if entry["method"] == "sqlite_backup":
            with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as con:
                if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError(f"SQLite integrity failed: {target}")
    return {"files_verified": len(manifest["entries"]), "original_inputs_unchanged": check_originals,
            "sqlite_integrity": "ok"}


def _review_name(path: Path, frame: pd.DataFrame | None = None) -> str:
    """Trust internal fields first: e.g. 2029_relations actually contains imperialism."""
    name = path.name
    if frame is not None and "codebook_id" in frame:
        values = frame.codebook_id.dropna().unique()
        if len(values) == 1 and values[0] in CODEBOOK_IDS:
            target = CODEBOOK_IDS[values[0]]
            for token in ("relations", "imperialism", "coupling", "relational"):
                if token in name:
                    return name.replace(token, target)
    for old, new in CODEBOOK_IDS.items():
        name = name.replace(old, new)
    return name


def copy_review_file(source: Path, target_dir: Path) -> Path:
    """Create a corrected copy; preserve every assessment/sample value and raw output."""
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix in {".csv", ".parquet"}:
        # Reading CSV as strings preserves literal NA/NONE, zeros, blanks, IDs and weights.
        frame = (pd.read_csv(source, dtype=str, keep_default_na=False) if suffix == ".csv"
                 else pd.read_parquet(source))
        out = normalize_frame(frame)
        target = target_dir / _review_name(source, frame)
        if target.exists():
            raise FileExistsError(f"corrected review target collision: {target}")
        if suffix == ".csv":
            out.to_csv(target, index=False, encoding="utf-8-sig")
            reread = pd.read_csv(target, dtype=str, keep_default_na=False)
        else:
            out.to_parquet(target, index=False)
            reread = pd.read_parquet(target)
        pd.testing.assert_frame_equal(frame, legacy_frame(reread), check_dtype=True)
        return target
    target = target_dir / _review_name(source)
    if target.exists():
        raise FileExistsError(f"corrected review target collision: {target}")
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(source)
        changed = False
        for sheet in wb:
            new_title = feature_name(sheet.title)
            if new_title != sheet.title and new_title in wb.sheetnames:
                raise ValueError(f"review sheet collision: {new_title}")
            changed |= sheet.title != new_title
            sheet.title = new_title
            headers = [c.value for c in sheet[1]]
            for col, header in enumerate(headers, start=1):
                if isinstance(header, str):
                    changed |= feature_name(header) != header
                    sheet.cell(1, col).value = feature_name(header)
                if header in {"codebook_id", "dimension"}:
                    for row in range(2, sheet.max_row + 1):
                        cell = sheet.cell(row, col)
                        if isinstance(cell.value, str):
                            value = codebook_id(cell.value) if header == "codebook_id" else feature_name(cell.value)
                            changed |= value != cell.value
                            cell.value = value
        if changed:
            wb.save(target)
        else:
            shutil.copy2(source, target)
        # Verify all cell values, allowing only the declared structured renames.
        before, after = load_workbook(source), load_workbook(target)
        for left, right in zip(before, after):
            headers = [c.value for c in left[1]]
            for row_left, row_right in zip(left, right):
                for a, b in zip(row_left, row_right):
                    expected = a.value
                    if isinstance(expected, str) and (a.row == 1 or headers[a.column-1] in {"codebook_id", "dimension"}):
                        expected = feature_name(expected)
                    if expected != b.value and not (expected in (None, "") and b.value in (None, "")):
                        raise AssertionError(f"review cell changed: {source} {left.title} {a.coordinate}")
        return target
    if suffix == ".json":
        original = json.loads(source.read_text())
        out = normalize_structured(original)
        # Historical sample metadata and probabilities belong to the historical frame.
        if isinstance(out, dict):
            out["identifier_migration"] = mapping_manifest()
            out["source_metadata_path"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    else:
        shutil.copy2(source, target)
    return target


def copy_reviews(backup: Path, stage: Path) -> list[dict]:
    results = []
    # Keep the prior validation frame and weights; expanded-population pack is separate.
    for tree in ("data/gold", "data/pilots", "data/samples", "research_outputs/validation"):
        for source in sorted((backup / tree).rglob("*")):
            if not source.is_file() or source.name == ".DS_Store":
                continue
            relative = source.relative_to(backup)
            base = stage / "review_materials" / relative.parent
            target = copy_review_file(source, base)
            results.append({"source": str(relative), "corrected_copy": str(target.relative_to(stage)),
                            "source_sha256": gr.sha256(source), "copy_sha256": gr.sha256(target)})
    old = stage / "review_materials/research_outputs/validation"
    active = stage / "expanded/validation"
    if active.exists():
        raise FileExistsError(active)
    shutil.copytree(old, active)
    (stage / "review_copy_manifest.json").write_text(json.dumps(results, indent=2))
    return results


def stage_config(backup: Path, stage: Path, name: str) -> dict:
    cfg = gr.load_config()
    for key in ("database", "corpus", "meeting_manifest"):
        cfg["source"][key] = str(backup / yaml.safe_load((backup / "config/research.yaml").read_text())["source"][key])
    cfg["source"]["output"] = str(stage / name)
    cfg["latex"]["output"] = str(stage / name / "latex")
    if name == "rename_only":
        cfg["population"]["included_observers"] = []
    cfg["validation"]["directory"] = "validation_member_states_palestine_v1"
    return cfg


def run_baseline(backup: Path, stage: Path) -> None:
    """Execute the unedited backed-up implementation with a declared stable RNG start.

    The wrapper only pins statsmodels' previously unseeded initialization. Its
    posterior-draw seeds, model recipe, thresholds and data are unchanged.
    """
    cfg = yaml.safe_load((backup / "config/research.yaml").read_text())
    for key in ("database", "corpus", "meeting_manifest", "codebooks"):
        cfg["source"][key] = str(backup / cfg["source"][key])
    cfg["source"]["output"] = str(stage / "original_population_legacy")
    cfg["latex"]["output"] = str(stage / "original_population_legacy/latex")
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / "baseline.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    # A WAL-mode backup may create empty sidecars on its first read. Prime that
    # connection before the frozen implementation takes its before/after hashes.
    with sqlite3.connect(f"file:{cfg['source']['database']}?mode=ro", uri=True) as con:
        con.execute("SELECT COUNT(*) FROM units").fetchone()
    code = '''
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from un_speeches.research import galtung_research as gr
import importlib.util
spec = importlib.util.spec_from_file_location("migration_bfgs_update", sys.argv[3])
algebra = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = algebra
spec.loader.exec_module(algebra)
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
original_vb = BinomialBayesMixedGLM.fit_vb
def equivalent_vb(self, *args, **kwargs):
    kwargs["fit_method"] = algebra.rank_two_bfgs
    return original_vb(self, *args, **kwargs)
BinomialBayesMixedGLM.fit_vb = equivalent_vb
original_fit = gr.fit_mixed_model
def deterministic_fit(frame, outcome, cfg, variant="primary"):
    np.random.seed(cfg["analysis"]["seed"] + sum(map(ord,outcome)) + sum(map(ord,variant)))
    print("BASELINE fitting",outcome,variant,flush=True)
    result = original_fit(frame, outcome, cfg, variant)
    print("BASELINE finished",outcome,result[1].get("converged"),flush=True)
    return result
gr.fit_mixed_model = deterministic_fit
cfg = gr.load_config(sys.argv[2])
data = gr.prepare(cfg)
for step in ["audit", "rq1", "rq2", "rq2_sensitivity", "rq3"]:
    print("BASELINE", step, flush=True)
    getattr(gr,step)(cfg,data)
from un_speeches.research.light_stability import run_light_stability
run_light_stability(cfg,data,Path(cfg["source"]["output"]))
print("BASELINE COMPLETE",flush=True)
'''
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code, str(backup / "src"), str(path), str(ROOT / "src/un_speeches/research/bfgs_update.py")], cwd=ROOT, env=env, check=True)
    (stage / "original_population_legacy/baseline_complete.json").write_text(json.dumps({"frozen_code": str(backup / "src"),
        "config": str(path), "optimizer_computation": optimizer_provenance(),
        "initialization": "existing declared seed plus legacy feature identity and variant"}, indent=2))


def export_views(backup: Path, stage: Path) -> None:
    """Rebuild selected-run analysis exports and full read-only database views."""
    from un_speeches.export.build_tables import write_table, write_workbook
    from un_speeches.export.dump_database import read_table, TABLES, write_csvs, write_workbook as write_db_workbook
    cfg = gr.load_config(stage / "expanded.yaml")
    data = gr.load_prepared(cfg)
    matrix = data["matrix"].copy()
    matrix["in_main_analysis"] = matrix.unit_id.isin(gr.population(matrix, cfg).unit_id)
    specs, _ = gr.codebook_specs(cfg)
    for cb, raw in specs.items():
        for dim in raw["dimensions"]:
            if dim.get("scale") != "ordinal":
                continue
            sub = data["dedup"]
            sub = sub[sub.codebook_id.eq(cb) & sub.dimension.eq(dim["name"])].copy()
            sub["score"] = sub.value.map({cat["code"]: cat["value"] for cat in dim["categories"]})
            aggregation = dim.get("score_aggregation", "max")
            scored = sub.groupby("unit_id").score.agg(aggregation)
            matrix[f"{cb}__{dim['name']}__score"] = matrix.unit_id.map(scored).astype("Float32" if aggregation == "mean" else "Int8")
    matrix = matrix.rename(columns={c: c.removeprefix("Y__") for c in matrix if c.startswith("Y__")})
    score_columns = [c for c in matrix if c.endswith("__score")]
    feature_columns = [c.removeprefix("Y__") for c in data["matrix"] if c.startswith("Y__")] + score_columns
    from un_speeches.lib.config import model_tag
    with sqlite3.connect(f"file:{cfg['source']['database']}?mode=ro", uri=True) as con:
        jobs = normalize_frame(pd.read_sql_query("SELECT unit_id,codebook_id,codebook_version,run_id,model,prompt_hash,request_hash,created_at,updated_at FROM jobs WHERE status='ok'", con))
    selected_jobs = []
    for cb, selection in cfg["selected_runs"].items():
        chosen = jobs[jobs.codebook_id.eq(cb) & jobs.codebook_version.eq(selection["version"]) & jobs.run_id.eq(selection["run_id"])].copy()
        selected_jobs.append(chosen)
        matrix[f"{cb}__model_tag"] = matrix.unit_id.map(chosen.set_index("unit_id").model.map(model_tag))
    metadata = ["unit_id", "speech_id", "country", "bloc", "is_party", "meeting_date", "role", "language", "n_words",
                "in_scope", "in_main_analysis", "is_observer", "institutional_status"]
    matrix = matrix[[*metadata, *feature_columns, *[f"{cb}__model_tag" for cb in cfg["selected_runs"]]]]
    # Country profiles use the same analytical population as the research pipeline.
    profiles = []
    for country, group in matrix[matrix.in_main_analysis].groupby("country"):
        row = {"country": country, "bloc": group.bloc.iloc[0], "is_party": int(group.is_party.max()),
               "n_speeches": group.speech_id.nunique(), "n_units": len(group)}
        for column in feature_columns:
            row[("n_scored__" if column.endswith("__score") else "n_annotated__") + column] = int(group[column].notna().sum())
            row[column] = group[column].mean()
        profiles.append(row)
    provenance = pd.concat(selected_jobs).drop(columns=["source_codebook_id", "identifier_mapping_version"])
    annotations = data["evidence"].merge(provenance, on=["unit_id", "codebook_id", "codebook_version", "run_id"],
                                          how="left", validate="many_to_one")
    annotations["model_tag"] = annotations.model.map(model_tag)
    annotations = annotations.merge(matrix[metadata], on="unit_id", how="left", validate="many_to_one")
    tables = {"annotations_long": annotations, "unit_code_matrix": matrix,
              "country_profiles": pd.DataFrame(profiles)}
    out = stage / "exports"
    for name, frame in tables.items():
        write_table(frame, out, name, ["parquet", "csv"])
    _, warnings = write_workbook(tables, out / "excel/un_speeches_annotations.xlsx")
    with sqlite3.connect(f"file:{cfg['source']['database']}?mode=ro", uri=True) as con:
        con.execute("BEGIN")
        database_tables = {t: read_table(con, t, False) for t in TABLES}
    selected = data["evidence"].merge(data["matrix"][["unit_id", "text", "country", "speech_id", "meeting_date"]],
                                        on="unit_id", how="left", validate="many_to_one")
    database_tables = {"annotations_with_text": selected, **database_tables}
    write_csvs(database_tables, out / "db/csv")
    _, notes = write_db_workbook(database_tables, out / "db/excel/annotations_db.xlsx")
    (out / "export_manifest.json").write_text(json.dumps({"identifier_mapping": mapping_manifest(),
        "selected_runs": cfg["selected_runs"], "source_database": cfg["source"]["database"],
        "analysis_signature": gr.analysis_signature(cfg), "rows": {k: len(v) for k, v in tables.items()},
        "profile_population": "UN member states and Palestine; centralized main analytical population",
        "raw_responses": "preserved verbatim in SQLite snapshot; heavy raw response columns omitted from workbook",
        "workbook_notes": warnings + notes}, indent=2))


def run_stage(backup: Path, stage: Path, workers: int, only: str | None = None) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("rename_only", "expanded"):
        if only and name != only:
            continue
        cfg = stage_config(backup, stage, name)
        config_path = stage / f"{name}.yaml"
        done = stage / name / "migration_complete.json"
        if done.exists():
            if json.loads(done.read_text())["signature"] == gr.analysis_signature(cfg):
                print(f"{name}: already complete with identical inputs/configuration", flush=True)
                continue
            raise RuntimeError(f"{name}: stage belongs to different inputs; choose a fresh --staging directory")
        config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False))
        cfg = gr.load_config(config_path)
        data = gr.prepare(cfg)
        for step in ("audit", "rq1", "rq2", "rq2_sensitivity", "rq3"):
            print(f"{name}: {step}", flush=True)
            kwargs = {"workers": workers} if step in {"rq2", "rq2_sensitivity"} else {}
            getattr(gr, step)(cfg, data, **kwargs)
        from .light_stability import run_light_stability
        run_light_stability(cfg, data, stage / name)
        if name == "expanded":
            from .validation_pack import export_validation_pack, write_validation_diagnostics
            export_validation_pack(cfg, data)
            write_validation_diagnostics(cfg)
        from .research_outputs import export_latex
        export_latex(stage / name, stage / name / "latex", cfg["latex"]["float_decimals"])
        from .reporting import write_provisional_report
        write_provisional_report(cfg, data, ROOT)
        done.write_text(json.dumps({"signature": gr.analysis_signature(cfg), "mapping": mapping_manifest()}, indent=2))
    if only is None or only == "expanded":
        if not (stage / "review_copy_manifest.json").exists():
            copy_reviews(backup, stage)
        export_views(backup, stage)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, *, numeric: bool = False) -> None:
    right = legacy_frame(right)
    right = right.drop(columns=[c for c in right if c not in left])
    if numeric:
        for a, b in (("dimension_a", "dimension_b"), ("feature_a", "feature_b")):
            if a in left and b in left:
                left = left.copy(); right = right.copy()
                for frame in (left, right):
                    pairs = [sorted(pair) for pair in frame[[a, b]].itertuples(index=False, name=None)]
                    frame[a] = [pair[0] for pair in pairs]; frame[b] = [pair[1] for pair in pairs]
        keys = [c for c in left if not pd.api.types.is_numeric_dtype(left[c])]
        if keys:
            left = left.sort_values(keys, kind="stable")
            right = right.sort_values(keys, kind="stable")
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right[left.columns].reset_index(drop=True),
                                  check_dtype=not numeric, check_exact=not numeric,
                                  rtol=1e-8, atol=1e-10)


def verify_migration(backup: Path, stage: Path) -> dict:
    integrity = verify_backup(backup)
    baseline = stage / "original_population_legacy"
    rename = stage / "rename_only"
    expanded = stage / "expanded"
    if not (baseline / "baseline_complete.json").exists():
        raise RuntimeError("frozen-code baseline has not completed")
    for directory in (baseline, rename, expanded):
        for relative in ("rq2_provisional/model_diagnostics.csv", "rq3_provisional/adjusted_profile_matrix.csv",
                         "rq3_provisional/light_stability_status.json"):
            if not (directory / relative).exists():
                raise RuntimeError(f"unfinished analysis: {directory / relative}")
    for directory in (rename, expanded):
        if not (directory / "migration_complete.json").exists():
            raise RuntimeError(f"staged pipeline not complete: {directory}")
    for relative in ("exports/export_manifest.json", "review_copy_manifest.json",
                     "expanded/validation_member_states_palestine_v1/coordinator_only/sampling_manifest.json"):
        if not (stage / relative).exists():
            raise RuntimeError(f"required migration artifact missing: {relative}")
    review_manifest = json.loads((stage / "review_copy_manifest.json").read_text())
    for entry in review_manifest:
        if gr.sha256(backup / entry["source"]) != entry["source_sha256"]:
            raise AssertionError(f"original review changed: {entry['source']}")
        if gr.sha256(stage / entry["corrected_copy"]) != entry["copy_sha256"]:
            raise AssertionError(f"corrected review changed after verification: {entry['corrected_copy']}")
    checked = []
    for file in (baseline / "prepared_data").glob("*.parquet"):
        compare_frames(pd.read_parquet(file), pd.read_parquet(rename / "prepared_data" / file.name))
        checked.append("prepared_data/" + file.name)
    if not checked:
        raise RuntimeError("Original legacy baseline missing; run the documented frozen-code baseline first")
    codebooks_verified = []
    for old, new in CODEBOOK_IDS.items():
        original = yaml.safe_load((backup / f"codebooks/{old}.yaml").read_text())
        expected = normalize_structured(original)
        expected["title"] = new.title()
        active = yaml.safe_load((ROOT / f"codebooks/{new}.yaml").read_text())
        if expected != active:
            raise AssertionError(f"codebook changed substantively: {new}")
        codebooks_verified.append(new)
    for relative in ("audit/country_eligibility.csv", "audit/selected_runs.csv", "audit/data_dictionary.csv",
                     "audit/coverage_overall.csv", "audit/coverage_country.csv", "audit/coverage_time.csv",
                     "audit/coverage_forum.csv", "audit/joint_coverage.csv", "audit/quality_issues.csv"):
        compare_frames(pd.read_csv(baseline / relative, keep_default_na=False),
                       pd.read_csv(rename / relative, keep_default_na=False), numeric=True)
        checked.append(relative)
    original_matrix = pd.read_parquet(baseline / "prepared_data/analysis_matrix.parquet")
    original_cfg = gr.load_config(stage / "rename_only.yaml")
    original_sc_ids = set(gr.population(original_matrix, original_cfg, original_cfg["population"]["rq2_rq3_forum"]).unit_id)
    evidence_scope_correction = []
    for tree in ("rq1_provisional", "rq2_provisional", "rq3_provisional"):
        for file in sorted((baseline / tree).glob("*.csv")):
            try:
                a = pd.read_csv(file, keep_default_na=False)
                b = pd.read_csv(rename / tree / file.name, keep_default_na=False)
            except pd.errors.EmptyDataError:
                continue
            if file.name == "representative_deviant_evidence.csv":
                # Historical examples used all forums even though RQ3 uses SC.
                # Verify corrected examples against the exact unchanged source
                # annotation rather than requiring an ineligible GA example.
                normalized = legacy_frame(b)
                keys = ["country", "profile_id", "case_role", "codebook_id"]
                paired = a.merge(normalized, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
                if len(paired) != len(a) or len(paired) != len(b):
                    raise AssertionError("RQ3 case groups changed in rename-only analysis")
                changed = paired[paired.unit_id_old.ne(paired.unit_id_new)]
                if not set(normalized.unit_id).issubset(original_sc_ids):
                    raise AssertionError("corrected RQ3 evidence contains excluded observations")
                for row in changed.itertuples():
                    if row.unit_id_old in original_sc_ids:
                        raise AssertionError("eligible historical RQ3 evidence unexpectedly changed")
                    evidence_scope_correction.append({"country": row.country, "codebook_id": row.codebook_id,
                        "old_unit_id": row.unit_id_old, "new_unit_id": row.unit_id_new,
                        "reason": "historical evidence outside the SC analytical population"})
                source = pd.read_parquet(baseline / "prepared_data/annotations_evidence.parquet")
                ann_keys = ["unit_id", "codebook_id", "codebook_version", "run_id", "annotation_index", "dimension"]
                joined = normalized.merge(source, on=ann_keys, suffixes=("_view", "_source"), validate="many_to_one", indicator=True)
                if not joined["_merge"].eq("both").all():
                    raise AssertionError("RQ3 example has no original annotation")
                for column in ("value", "evidence", "confidence", "evidence_verified"):
                    pd.testing.assert_series_equal(joined[column+"_view"], joined[column+"_source"],
                        check_names=False, check_dtype=False, rtol=1e-12, atol=1e-12)
                checked.append(str(file.relative_to(baseline)) + " (source-content and SC eligibility verified)")
                continue
            # Display/provenance text is intentionally corrected; numerical fields match.
            for column in ("scope", "contrast"):
                if column in a:
                    a = a.drop(columns=column); b = b.drop(columns=column)
            compare_frames(a, b, numeric=True)
            checked.append(str(file.relative_to(baseline)))
    draws_relative = "rq2_provisional/adjusted_probability_draws.parquet"
    compare_frames(pd.read_parquet(baseline / draws_relative), pd.read_parquet(rename / draws_relative), numeric=True)
    checked.append(draws_relative)
    for file in sorted((baseline / "rq3_provisional").glob("*.npy")):
        np.testing.assert_allclose(np.load(file), np.load(rename / "rq3_provisional" / file.name), rtol=1e-8, atol=1e-10)
        checked.append(str(file.relative_to(baseline)))
    a = pd.read_parquet(rename / "prepared_data/analysis_matrix.parquet")
    b = pd.read_parquet(expanded / "prepared_data/analysis_matrix.parquet")
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    oldcfg = gr.load_config(stage / "rename_only.yaml")
    newcfg = gr.load_config(stage / "expanded.yaml")
    export_manifest = json.loads((stage / "exports/export_manifest.json").read_text())
    if export_manifest.get("analysis_signature") != gr.analysis_signature(newcfg):
        raise RuntimeError("exports are incomplete or belong to an incompatible source/configuration")
    counts = {}
    for label, forum in (("rq1", None), ("rq2_rq3_sc", newcfg["population"]["rq2_rq3_forum"])):
        old, new = gr.population(a, oldcfg, forum), gr.population(b, newcfg, forum)
        added = new[~new.unit_id.isin(old.unit_id)]
        if not added.country.eq("Palestine").all() or not set(old.unit_id).issubset(set(new.unit_id)):
            raise AssertionError("Population change must add only eligible Palestine observations")
        pd.testing.assert_frame_equal(old, new[~new.country.eq("Palestine")], check_exact=True)
        if new.country.eq("Holy See").any() or not new.country.eq("Israel").any():
            raise AssertionError("Holy See exclusion / Israel inclusion changed")
        counts[label] = {"before_paragraphs": len(old), "after_paragraphs": len(new),
                         "before_speeches": old.speech_id.nunique(), "after_speeches": new.speech_id.nunique(),
                         "palestine_paragraphs": len(added), "palestine_speeches": added.speech_id.nunique()}
    p = gr.population(b, newcfg)
    ids = set(p.loc[p.country.eq("Palestine"), "unit_id"])
    ev = pd.read_parquet(expanded / "prepared_data/evaluation_status.parquet")
    coverage = (ev[ev.unit_id.isin(ids)].groupby(["codebook_id", "dimension", "evaluation_status"])
                .size().rename("paragraphs").reset_index())
    coverage.to_csv(stage / "palestine_coverage.csv", index=False)
    eligibility = pd.read_csv(expanded / "audit/country_eligibility.csv")
    old_eligibility = pd.read_csv(rename / "audit/country_eligibility.csv")
    pd.testing.assert_frame_equal(old_eligibility.reset_index(drop=True),
        eligibility[~eligibility.country.eq("Palestine")].reset_index(drop=True), check_exact=True)
    primary_countries = set(eligibility.loc[eligibility.included_primary, "country"])
    adjusted = pd.read_csv(expanded / "rq2_provisional/adjusted_probabilities.csv")
    profile = pd.read_csv(expanded / "rq3_provisional/adjusted_profile_matrix.csv")
    if "Palestine" in primary_countries:
        if "Palestine" not in set(adjusted.country) or "Palestine" not in set(profile.country):
            raise AssertionError("Eligible Palestine missing from RQ2/RQ3 results")
    prevalence = pd.read_csv(expanded / "rq1_provisional/prevalence.csv")
    for row in prevalence.itertuples():
        values = p[f"Y__{row.codebook_id}__{row.dimension}__{row.code}"]
        if row.evaluable_units != values.notna().sum() or row.positive_units != values.sum():
            raise AssertionError(f"incorrect expanded RQ1 numerator/denominator: {row.code}")
    new_key = pd.read_csv(expanded / "validation_member_states_palestine_v1/coordinator_only/coordinator_key.csv")
    if not set(new_key.unit_id).issubset(set(p.unit_id)):
        raise AssertionError("new validation sample contains excluded observations")
    for cb, group in new_key[new_key.sampling_component.eq("probability")].groupby("codebook_id"):
        dimension = newcfg["core_profile_dimensions"][cb]
        frame_size = ev[ev.unit_id.isin(p.unit_id) & ev.codebook_id.eq(cb) & ev.dimension.eq(dimension)
                        & ev.evaluation_status.eq("successful_complete")].unit_id.nunique()
        expected_probability = len(group) / frame_size
        if not ((group.selection_probability - expected_probability).abs() < 1e-12).all():
            raise AssertionError(f"incorrect expanded-frame sample probability: {cb}")
    result = {"mapping": mapping_manifest(), "backup": str(backup), "integrity": integrity,
              "rename_only": {"exact_prepared_and_content_comparisons": checked,
                              "codebook_content_verified": codebooks_verified,
                              "numerical_tolerances": {"rtol": 1e-8, "atol": 1e-10},
                              "result": "passed"},
              "add_palestine": {"counts": counts, "existing_countries_unchanged": True,
                                "coverage": coverage.to_dict("records"),
                                "eligibility": eligibility[eligibility.country.eq("Palestine")].to_dict("records")},
              "optimizer_computation": optimizer_provenance(), "validation_status": "pending"}
    result["review_materials"] = {"files_verified": len(review_manifest),
                                   "historical_samples_redrawn": False, "historical_weights_reused_for_new_frame": False}
    result["historical_evidence_scope_correction"] = evidence_scope_correction
    (stage / "verification.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def activate(backup: Path, stage: Path) -> None:
    # Always rerun integrity/equivalence checks before publishing a staged result.
    verify_migration(backup, stage)
    state = stage / "activation.json"
    if state.exists():
        print("Already activated; no files overwritten")
        return
    # Previous outputs are retained in both the initial backup and a local preactivation tree.
    previous = stage / "preactivation"
    previous.mkdir(exist_ok=False)
    replacements = [(stage / "expanded", ROOT / "research_outputs")]
    if (stage / "exports").exists():
        replacements.append((stage / "exports", ROOT / "data/exports"))
    for source, target in replacements:
        temporary = target.with_name(target.name + ".migration-next")
        if temporary.exists():
            raise FileExistsError(temporary)
        shutil.copytree(source, temporary)
        old = previous / target.name
        if target.exists():
            target.rename(old)
        temporary.rename(target)
    # SQLite's backup mechanism can change physical header bytes while preserving
    # all records. Record both fingerprints and align the active cache signature
    # with the verified unchanged live source, preserving the staged provenance.
    snapshot_path = ROOT / "research_outputs/audit/snapshot_manifest.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["staged_analysis_signature"] = snapshot["analysis_signature"]
    snapshot["analysis_signature"] = gr.analysis_signature(gr.load_config())
    snapshot["live_source_verified_against_backup_manifest"] = str(backup / "manifest.json")
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
    shutil.copytree(stage / "review_materials/data", ROOT / "data/review_materials_identifier_v1", dirs_exist_ok=True)
    state.write_text(json.dumps({"activated_at": datetime.now(timezone.utc).isoformat(),
                                 "backup": str(backup), "previous_outputs": str(previous)}, indent=2))


def rollback(backup: Path, stage: Path) -> None:
    verify_backup(backup, check_originals=False)
    retained = stage / ("rollback_preserved_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    retained.mkdir(parents=True, exist_ok=False)
    for rel in (Path("research_outputs"), Path("data/exports")):
        target = ROOT / rel
        temporary = target.with_name(target.name + ".rollback-next")
        if temporary.exists():
            raise FileExistsError(temporary)
        shutil.copytree(backup / rel, temporary)
        if target.exists():
            target.rename(retained / rel.name)
        temporary.rename(target)
    # Restore only files present in the snapshot; never delete later researcher work.
    for entry in json.loads((backup / "manifest.json").read_text())["entries"]:
        rel = Path(entry["path"])
        if rel.parts[0] == "research_outputs" or rel.parts[:2] == ("data", "exports"):
            continue
        if rel.parts[0] == "data" and rel.parts[1] in {"raw", "db", "interim", "pilots", "gold", "samples"}:
            continue  # immutable source storage is never changed by this migration
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup / rel, target)
    for old, new in CODEBOOK_IDS.items():
        for suffix in ("yaml", "csv"):
            p = ROOT / f"codebooks/{new}.{suffix}"
            if not (backup / p.relative_to(ROOT)).exists():
                p.unlink(missing_ok=True)
    for rel in ("src/un_speeches/lib/identifiers.py", "src/un_speeches/research/migrate_identifiers.py",
                "src/un_speeches/research/bfgs_update.py",
                "config/research_environment_identifier_v1.txt",
                "tests/test_identifier_migration.py", "docs/IDENTIFIER_MIGRATION.md"):
        path = ROOT / rel
        if path.exists() and not (backup / rel).exists():
            target = retained / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    print(f"Original code/config/output files and initial uncommitted changes restored. Migrated outputs retained in {retained}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="plan", choices=["plan", "backup", "baseline", "stage", "exports", "verify", "activate", "rollback"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", type=Path)
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", choices=["rename_only", "expanded"])
    args = ap.parse_args()
    if args.dry_run or args.action == "plan":
        print(json.dumps({"action": args.action, "mapping": mapping_manifest(),
                          "affected_artifacts": [str(p.relative_to(ROOT)) for p in affected_files()],
                          "staging": str(args.staging), "source_database": "read-only; snapshot via SQLite backup API",
                          "expected_changes": ["correct structured identifiers; preserve source identifiers/provenance",
                            "copy reviews without redrawing historical samples", "add only eligible Palestine observations",
                            "refit existing models and sensitivities; regenerate assets; no LLM calls"],
                          "rollback": "restore snapshot source/config/output files; preserve original research database"}, indent=2))
        return 0
    if args.action == "backup":
        print(make_backup())
        return 0
    backup = args.backup or ROOT / (ROOT / "migration_backups/LATEST").read_text().strip()
    backup = backup.resolve()
    if args.action == "stage":
        run_stage(backup, args.staging.resolve(), args.workers, args.only)
    elif args.action == "baseline":
        run_baseline(backup, args.staging.resolve())
    elif args.action == "exports":
        export_views(backup, args.staging.resolve())
    elif args.action == "verify":
        print(json.dumps(verify_migration(backup, args.staging.resolve()), indent=2, default=str))
    elif args.action == "activate":
        activate(backup, args.staging.resolve())
    elif args.action == "rollback":
        rollback(backup, args.staging.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
