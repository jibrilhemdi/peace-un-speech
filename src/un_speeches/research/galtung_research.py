"""Reproducible, read-only Galtung research pipeline.

No function in this module calls an annotation service or writes to source data. All
research products are explicitly provisional until returned human assessments are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from un_speeches.lib.identifiers import (MAPPING_VERSION, normalize_config, normalize_frame,
                                        feature_name, mapping_manifest)
from .bfgs_update import rank_two_bfgs, UPDATE_VERSION, provenance as optimizer_provenance

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config" / "research.yaml"
STATUS = "PROVISIONAL / UNVALIDATED"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = normalize_config(cfg)
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def outdir(cfg: dict[str, Any], *parts: str) -> Path:
    p = ROOT / cfg["source"]["output"]
    if parts and parts[0] == "validation":
        parts = (cfg.get("validation", {}).get("directory", "validation"), *parts[1:])
    for part in parts:
        p /= part
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprint(path: Path) -> str | None:
    # A first read of a WAL-mode SQLite snapshot can create an empty sidecar.
    # Missing and empty WAL both contain no transactions.
    if path.name.endswith("-wal") and (not path.exists() or path.stat().st_size == 0):
        return None
    return sha256(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_frame(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    write_csv(df, base.with_suffix(".csv"))


def read_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read all SQLite tables in one read transaction and verify the source stayed fixed."""
    db = (ROOT / cfg["source"]["database"]).resolve()
    corpus_path = (ROOT / cfg["source"]["corpus"]).resolve()
    meeting_manifest_path = (ROOT / cfg["source"]["meeting_manifest"]).resolve()
    before = {str(p): source_fingerprint(p) for p in (db, Path(str(db) + "-wal"), corpus_path, meeting_manifest_path)}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("BEGIN")
    try:
        tables = {
            n: pd.read_sql_query(f"SELECT * FROM {n}", con)
            for n in ("units", "jobs", "annotations", "pilots", "provider_caps")
        }
    finally:
        con.rollback()
        con.close()
    after = {str(p): source_fingerprint(p) for p in (db, Path(str(db) + "-wal"), corpus_path, meeting_manifest_path)}
    if before != after:
        raise RuntimeError("Source database/corpus changed during extraction; rerun for a consistent snapshot")
    tables["corpus"] = pd.read_csv(corpus_path, low_memory=False)
    tables["meeting_manifest"] = pd.read_csv(meeting_manifest_path, low_memory=False)
    tables["fingerprints"] = before
    for name in ("jobs", "annotations", "pilots"):
        tables[name] = normalize_frame(tables[name])
    return tables


def normalize_symbol(value: str) -> str:
    """Match catalogue-style resumption spacing to ODS manifest symbols."""
    return re.sub(r"\s+", "", str(value)).casefold()


def mark_scope(paragraphs: pd.DataFrame, manifest: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Keep source rows intact and mark which meetings enter statistical work."""
    if cfg["population"]["meeting_rule"] != "palestine_agenda":
        raise ValueError("unsupported meeting rule")
    selected = manifest.loc[manifest.exists.eq(True),
        ["symbol", "meeting_date", "agenda", "agenda_palestine"]].copy()
    selected["symbol_key"] = selected.symbol.map(normalize_symbol)
    if selected.symbol_key.duplicated().any():
        raise ValueError("meeting manifest has duplicate normalized symbols")
    out = paragraphs.copy()
    out["symbol_key"] = out.doc_symbol.map(normalize_symbol)
    out = out.merge(selected.rename(columns={"meeting_date": "manifest_meeting_date"}),
                    on="symbol_key", how="left", validate="many_to_one")
    matched = out.symbol.notna()
    start = pd.Timestamp(cfg["population"]["start_date"])
    end = pd.Timestamp(cfg["population"]["end_date"])
    in_dates = out.date.between(start, end, inclusive="both")
    if (~matched & in_dates).any():
        missing = sorted(out.loc[~matched & in_dates, "doc_symbol"].dropna().unique())
        raise ValueError(f"in-window meetings missing from manifest: {missing}")
    manifest_date = pd.to_datetime(out.manifest_meeting_date, errors="coerce")
    if (matched & in_dates & out.date.ne(manifest_date)).any():
        raise ValueError("meeting dates disagree with manifest")
    agenda_ok = pd.to_numeric(out.agenda_palestine, errors="coerce").eq(1)
    out["in_scope"] = in_dates & matched & agenda_ok
    out["scope_exclusion_reason"] = np.select(
        [out.date.lt(start), out.date.gt(end), ~matched, ~agenda_ok],
        ["before_start", "after_end", "missing_manifest", "agenda_not_palestine"],
        default="included")
    return out.drop(columns=["symbol_key"])


def codebook_specs(cfg: dict[str, Any]) -> tuple[dict[str, dict], list[dict]]:
    specs, rows = {}, []
    cbdir = ROOT / cfg["source"]["codebooks"]
    for cb_id, selection in cfg["selected_runs"].items():
        path = cbdir / f"{cb_id}.yaml"
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw["id"] != cb_id:
            raise ValueError(f"canonical codebook mismatch: {path}: {raw['id']}")
        if int(raw["version"]) != int(selection["version"]):
            raise ValueError(f"selected {cb_id} v{selection['version']} but on-disk codebook is v{raw['version']}")
        specs[cb_id] = raw
        for dim in raw["dimensions"]:
            for cat in dim["categories"]:
                rows.append({"codebook_id": cb_id, "codebook_version": raw["version"],
                             "dimension": dim["name"], "required": bool(dim.get("required", True)),
                             "conditional_on": dim.get("conditional_on"), "code": cat["code"],
                             "label": cat["label"], "definition": cat["definition"],
                             "codebook_theory": raw.get("theory", ""),
                             "codebook_instructions": raw.get("instructions", ""),
                             "includes": " | ".join(cat.get("includes", [])),
                             "excludes": " | ".join(cat.get("excludes", [])),
                             "positive_examples": " | ".join(x.get("text", "") for x in cat.get("positive_examples", [])),
                             "negative_examples": " | ".join(x.get("text", "") for x in cat.get("negative_examples", [])),
                             "allows_multiple": bool(dim.get("allow_multiple_values", raw.get("allow_multiple_annotations", False)))})
    actual = {(r["codebook_id"], r["dimension"], r["code"]) for r in rows}
    expected = set()
    for dim, codes in cfg["dimensions"].items():
        cb = next((c for c, raw in specs.items() if dim in [d["name"] for d in raw["dimensions"]]), None)
        if cb is None:
            raise ValueError(f"configured dimension absent from codebooks: {dim}")
        for code in codes:
            if code != "NONE": expected.add((cb, dim, code))
    if not expected.issubset(actual):
        raise ValueError("canonical Galtung identifiers do not match configuration")
    return specs, rows


def prepare(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    signature = analysis_signature(cfg)
    base = ROOT / cfg["source"]["output"]
    previous_manifest = base / "audit/snapshot_manifest.json"
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    if previous.get("analysis_signature") != signature:
        generated = [base / name for name in ("rq1_provisional", "rq2_provisional", "rq3_provisional", "latex", "reports")]
        if any(p.exists() for p in generated):
            archive = base / "incompatible_cache" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archive.mkdir(parents=True)
            for path in generated:
                if path.exists():
                    path.rename(archive / path.name)
            (archive / "invalidation.json").write_text(json.dumps({"old_signature": previous.get("analysis_signature"),
                "new_signature": signature, "reason": "source, codebook naming, configuration or population changed"}, indent=2))
    src = read_sources(cfg)
    specs, dictionary = codebook_specs(cfg)
    units, jobs, ann, corpus = src["units"].copy(), src["jobs"].copy(), src["annotations"].copy(), src["corpus"].copy()
    if units.unit_id.duplicated().any():
        raise ValueError("unit_id is not unique in units")
    # Verified repository convention: speech_id == doc_symbol + '#' + turn_index.
    corpus["speech_id"] = corpus.doc_symbol.astype(str) + "#" + corpus.turn_index.astype(int).astype(str)
    if corpus.speech_id.duplicated().any():
        raise ValueError("derived corpus speech_id is not unique")
    extra_cols = ["speech_id", "doc_symbol", "meeting_number", "speaker_header", "word_count",
                  "source_pdf", "record_ids", "meeting_provenance"]
    paragraphs = units.merge(corpus[extra_cols], on="speech_id", how="left", validate="many_to_one", indicator=True)
    paragraphs["join_status"] = paragraphs.pop("_merge").astype(str)
    paragraphs["date"] = pd.to_datetime(paragraphs.meeting_date, format="%d %B %Y", errors="coerce")
    paragraphs = mark_scope(paragraphs, src["meeting_manifest"], cfg)
    paragraphs["year_month"] = paragraphs.date.dt.to_period("M").astype("string")
    paragraphs["forum"] = np.select([paragraphs.doc_symbol.str.startswith("S/", na=False),
                                      paragraphs.doc_symbol.str.startswith("A/", na=False)], ["SC", "GA"], default="UNKNOWN")
    paragraphs["meeting_id"] = paragraphs.doc_symbol
    paragraphs["is_observer"] = paragraphs.country.isin(cfg["population"]["observer_cases"])
    paragraphs["institutional_status"] = np.where(paragraphs.is_observer, "non-member observer state",
        np.where(paragraphs.role.eq("representative") & paragraphs.country.notna(), "UN member state", "official/other"))
    paragraphs["text_missing"] = paragraphs.text.isna() | paragraphs.text.astype(str).str.strip().eq("")
    paragraphs["date_invalid"] = paragraphs.date.isna()
    paragraphs["speaker_ambiguous"] = paragraphs.ambassador_name.isna() | paragraphs.ambassador_name.astype(str).str.strip().eq("")
    # Select exactly one explicitly configured run/version per codebook; never union runs.
    selected_jobs, selected_ann = [], []
    selection_rows = []
    for cb, sel in cfg["selected_runs"].items():
        qj = jobs[(jobs.codebook_id == cb) & (jobs.codebook_version == sel["version"]) & (jobs.run_id == sel["run_id"])].copy()
        qa = ann[(ann.codebook_id == cb) & (ann.codebook_version == sel["version"]) & (ann.run_id == sel["run_id"])].copy()
        selected_jobs.append(qj); selected_ann.append(qa)
        selection_rows.append({"codebook_id": cb, **sel, "jobs": len(qj), "ok_jobs": int(qj.status.eq("ok").sum()),
                               "units": qj.unit_id.nunique(), "annotation_rows": len(qa),
                               "models": " | ".join(sorted(qj.model.dropna().unique())),
                               "prompt_hashes": " | ".join(sorted(qj.prompt_hash.dropna().unique()))})
    sj = pd.concat(selected_jobs, ignore_index=True)
    evidence = pd.concat(selected_ann, ignore_index=True)
    dedup = evidence.drop_duplicates(["unit_id", "codebook_id", "codebook_version", "run_id", "dimension", "value"]).copy()
    eval_rows, matrix = [], paragraphs.copy()
    for cb, raw in specs.items():
        sel = cfg["selected_runs"][cb]
        j = sj[(sj.codebook_id == cb) & (sj.codebook_version == sel["version"]) & (sj.run_id == sel["run_id"])].set_index("unit_id")
        a = dedup[(dedup.codebook_id == cb) & (dedup.codebook_version == sel["version"]) & (dedup.run_id == sel["run_id"])]
        ok = matrix.unit_id.map(j.status).eq("ok")
        for dim in raw["dimensions"]:
            dname = dim["name"]
            applicable = pd.Series(True, index=matrix.index)
            cond = dim.get("conditional_on")
            if cond:
                parent = cond if isinstance(cond, str) else cond["dimension"]
                parent_positive = set(a.loc[a.dimension.eq(parent), "unit_id"])
                applicable = matrix.unit_id.isin(parent_positive)
            status = np.where(~matrix.unit_id.isin(j.index), "not_evaluated",
                     np.where(~ok, matrix.unit_id.map(j.status).fillna("unknown"),
                     np.where(~applicable, "not_applicable", "successful_complete")))
            eval_rows.append(pd.DataFrame({"unit_id": matrix.unit_id, "codebook_id": cb,
                                           "codebook_version": sel["version"], "run_id": sel["run_id"],
                                           "dimension": dname, "evaluation_status": status}))
            complete = pd.Series(status).eq("successful_complete").to_numpy()
            for cat in dim["categories"]:
                code = cat["code"]
                hit = set(a.loc[a.dimension.eq(dname) & a.value.eq(code), "unit_id"])
                vals = pd.Series(pd.NA, index=matrix.index, dtype="Int8")
                vals.loc[complete] = matrix.loc[complete, "unit_id"].isin(hit).astype("int8")
                matrix[f"Y__{cb}__{dname}__{code}"] = vals
            if dname == "attribution":
                hit = set(a.loc[a.dimension.eq(dname) & a.value.eq("NONE"), "unit_id"])
                vals = pd.Series(pd.NA, index=matrix.index, dtype="Int8")
                vals.loc[complete] = matrix.loc[complete, "unit_id"].isin(hit).astype("int8")
                matrix[f"Y__{cb}__{dname}__NONE"] = vals
    evaluations = pd.concat(eval_rows, ignore_index=True)
    pdir = outdir(cfg, "prepared_data")
    write_frame(paragraphs, pdir / "paragraphs")
    write_frame(evidence, pdir / "annotations_evidence")
    write_frame(dedup, pdir / "annotations_long_deduplicated")
    write_frame(evaluations, pdir / "evaluation_status")
    write_frame(matrix, pdir / "analysis_matrix")
    write_csv(pd.DataFrame(dictionary), outdir(cfg, "audit") / "data_dictionary.csv")
    schema_rows=[]
    for table in ("units","jobs","annotations","pilots","provider_caps","corpus","meeting_manifest"):
        for col,dtype in src[table].dtypes.items(): schema_rows.append({"table":table,"column":col,"pandas_dtype":str(dtype)})
    write_csv(pd.DataFrame(schema_rows),outdir(cfg,"audit")/"source_schema.csv")
    write_csv(pd.DataFrame(selection_rows), outdir(cfg, "audit") / "selected_runs.csv")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "validation_status": "pending",
                "analysis_status": STATUS, "config": cfg["_path"], "source_fingerprints": src["fingerprints"],
                "identifier_mapping": mapping_manifest(), "analysis_signature": signature,
                "population": cfg["population"],
                "optimizer_computation": optimizer_provenance(),
                "rows": {"paragraphs": len(paragraphs), "evidence": len(evidence), "deduplicated": len(dedup),
                         "evaluation_status": len(evaluations)},
                "software": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__,
                             "numpy": np.__version__}}
    (outdir(cfg, "audit") / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"paragraphs": paragraphs, "evidence": evidence, "dedup": dedup,
            "evaluations": evaluations, "matrix": matrix, "dictionary": pd.DataFrame(dictionary)}


def load_prepared(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    p = outdir(cfg, "prepared_data")
    needed = {"paragraphs": "paragraphs.parquet", "evidence": "annotations_evidence.parquet",
              "dedup": "annotations_long_deduplicated.parquet", "evaluations": "evaluation_status.parquet",
              "matrix": "analysis_matrix.parquet"}
    manifest_path = outdir(cfg, "audit") / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if (not all((p / f).exists() for f in needed.values())
            or manifest.get("analysis_signature") != analysis_signature(cfg)):
        return prepare(cfg)
    out = {k: pd.read_parquet(p / f) for k, f in needed.items()}
    out["dictionary"] = pd.read_csv(outdir(cfg, "audit") / "data_dictionary.csv")
    return out


def analysis_signature(cfg: dict[str, Any]) -> str:
    """Reject prepared caches from another naming/population/source configuration."""
    relevant = {k: cfg[k] for k in ("selected_runs", "dimensions", "core_profile_dimensions", "population", "thresholds", "analysis")}
    relevant["mapping_version"] = MAPPING_VERSION
    relevant["optimizer_computation"] = UPDATE_VERSION
    relevant["inputs"] = {k: sha256(ROOT / cfg["source"][k]) for k in ("database", "corpus", "meeting_manifest")}
    wal = source_fingerprint(Path(str(ROOT / cfg["source"]["database"]) + "-wal"))
    if wal is not None:
        relevant["wal"] = wal
    relevant["codebooks"] = {cb: sha256(ROOT / cfg["source"]["codebooks"] / f"{cb}.yaml") for cb in cfg["selected_runs"]}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def population(matrix: pd.DataFrame, cfg: dict[str, Any], forum: str | None = None,
               include_observer: bool = False) -> pd.DataFrame:
    d = matrix[(matrix.role == cfg["population"]["role"]) & matrix.para_is_procedural.eq(0)
               & matrix.in_scope.eq(True)].copy()
    if not include_observer:
        excluded = set(cfg["population"]["observer_cases"]) - set(cfg["population"].get("included_observers", []))
        d = d[~d.country.isin(excluded)]
    if forum:
        d = d[d.forum.eq(forum)]
    else:
        d = d[d.forum.isin(cfg["population"]["rq1_forums"])]
    return d


def audit(cfg: dict[str, Any], data: dict[str, pd.DataFrame]) -> None:
    p, e = data["paragraphs"], data["evaluations"]
    adir = outdir(cfg, "audit")
    source_meetings = (p.groupby("doc_symbol", dropna=False)
                       .agg(source_date=("date", "first"), paragraphs=("unit_id", "size"),
                            speeches=("speech_id", "nunique")).reset_index())
    source_meetings["symbol_key"] = source_meetings.doc_symbol.map(normalize_symbol)
    meeting_manifest = pd.read_csv(ROOT / cfg["source"]["meeting_manifest"])
    meeting_manifest["symbol_key"] = meeting_manifest.symbol.map(normalize_symbol)
    meeting_scope = meeting_manifest.merge(source_meetings, on="symbol_key", how="outer",
                                           validate="one_to_one")
    meeting_scope["date"] = pd.to_datetime(meeting_scope.meeting_date, errors="coerce")
    meeting_scope["date"] = meeting_scope.date.fillna(meeting_scope.source_date)
    start, end = pd.Timestamp(cfg["population"]["start_date"]), pd.Timestamp(cfg["population"]["end_date"])
    in_dates = meeting_scope.date.between(start, end, inclusive="both")
    agenda_ok = pd.to_numeric(meeting_scope.agenda_palestine, errors="coerce").eq(1)
    has_source = meeting_scope.paragraphs.fillna(0).gt(0)
    meeting_scope["eligible_by_rule"] = in_dates & agenda_ok & meeting_scope.exists.eq(True)
    meeting_scope["in_scope"] = meeting_scope.eligible_by_rule & has_source
    meeting_scope["scope_exclusion_reason"] = np.select(
        [meeting_scope.date.lt(start), meeting_scope.date.gt(end), meeting_scope.exists.eq(False),
         meeting_scope.symbol.isna(), ~agenda_ok, ~has_source],
        ["before_start", "after_end", "missing_pdf", "missing_manifest",
         "agenda_not_palestine", "eligible_but_not_annotated"], default="included")
    meeting_scope = meeting_scope.sort_values(["date", "symbol_key"])
    write_csv(meeting_scope[["symbol", "doc_symbol", "date", "agenda", "agenda_palestine",
                             "eligible_by_rule", "in_scope", "scope_exclusion_reason",
                             "paragraphs", "speeches"]], adir / "meeting_scope.csv")
    eligible_count = int(meeting_scope.eligible_by_rule.sum())
    observed_count = int(meeting_scope.in_scope.sum())
    issues = pd.DataFrame([
        {"check": "unit_id_unique", "n": int(p.unit_id.duplicated().sum()), "severity": "error"},
        {"check": "speech_join_unmatched", "n": int(p.join_status.ne("both").sum()), "severity": "error"},
        {"check": "missing_text", "n": int(p.text_missing.sum()), "severity": "error"},
        {"check": "invalid_date", "n": int(p.date_invalid.sum()), "severity": "error"},
        {"check": "unknown_forum", "n": int(p.forum.eq("UNKNOWN").sum()), "severity": "warning"},
        {"check": "ambiguous_speaker", "n": int(p.speaker_ambiguous.sum()), "severity": "flag"},
    ])
    write_csv(issues, adir / "quality_issues.csv")
    cov = e.merge(p[["unit_id", "country", "year_month", "forum", "role", "in_scope"]],
                  on="unit_id", how="left")
    write_csv(cov.groupby(["codebook_id", "dimension", "evaluation_status"], dropna=False)
              .size().rename("units").reset_index(), adir / "source_coverage_overall.csv")
    cov = cov[cov.in_scope.eq(True)]
    for by, name in [(["codebook_id", "dimension", "evaluation_status"], "coverage_overall"),
                     (["country", "codebook_id", "dimension", "evaluation_status"], "coverage_country"),
                     (["year_month", "codebook_id", "dimension", "evaluation_status"], "coverage_time"),
                     (["forum", "codebook_id", "dimension", "evaluation_status"], "coverage_forum")]:
        write_csv(cov.groupby(by, dropna=False).size().rename("units").reset_index(), adir / f"{name}.csv")
    joint = (e[e.unit_id.isin(p.loc[p.in_scope.eq(True), "unit_id"])]
             .assign(ok=lambda x: x.evaluation_status.eq("successful_complete"))
             .pivot_table(index="unit_id", columns="dimension", values="ok", aggfunc="max", fill_value=False))
    jr = []
    for a in joint.columns:
        for b in joint.columns:
            if a < b: jr.append({"dimension_a": a, "dimension_b": b, "joint_successful_units": int((joint[a] & joint[b]).sum())})
    write_csv(pd.DataFrame(jr), adir / "joint_coverage.csv")
    main = population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])
    population_label = "UN member states and Palestine" if "Palestine" in cfg["population"].get("included_observers", []) else "UN member states"
    core_cbs = list(cfg["core_profile_dimensions"])
    eval_ok = e[e.evaluation_status.eq("successful_complete")].pivot_table(index="unit_id", columns="codebook_id", values="dimension", aggfunc="size", fill_value=0)
    rows = []
    for country, g in main.groupby("country"):
        row = {"country": country, "unique_paragraphs": g.unit_id.nunique(), "speeches": g.speech_id.nunique()}
        for cb in core_cbs: row[f"eval_{cb}"] = int(g.unit_id.isin(eval_ok.index[eval_ok.get(cb, 0).gt(0)]).sum())
        for tier in ("primary", "sensitivity"):
            t = cfg["thresholds"][tier]
            reasons = []
            if row["unique_paragraphs"] < t["unique_paragraphs"]: reasons.append("paragraphs")
            if row["speeches"] < t["speeches"]: reasons.append("speeches")
            if any(row[f"eval_{cb}"] < t["evaluated_per_core_codebook"] for cb in core_cbs): reasons.append("codebook_coverage")
            row[f"included_{tier}"] = not reasons; row[f"exclusion_reasons_{tier}"] = "|".join(reasons)
        rows.append(row)
    write_csv(pd.DataFrame(rows), adir / "country_eligibility.csv")
    full_main = population(data["matrix"], cfg)
    summaries = []
    for label, frame in (("rq1", full_main), ("rq2_rq3_sc_before_thresholds", main)):
        summaries.append({"population": label, "paragraphs": len(frame), "speeches": frame.speech_id.nunique(),
                          "countries": frame.country.nunique(), "palestine_paragraphs": int(frame.country.eq("Palestine").sum()),
                          "palestine_speeches": frame.loc[frame.country.eq("Palestine"), "speech_id"].nunique()})
    write_csv(pd.DataFrame(summaries), adir / "population_summary.csv")
    palestine_ids = set(full_main.loc[full_main.country.eq("Palestine"), "unit_id"])
    palestine_eval = e[e.unit_id.isin(palestine_ids)]
    write_csv(palestine_eval.groupby(["codebook_id", "dimension", "evaluation_status"])
              .size().rename("paragraphs").reset_index(), adir / "palestine_coverage.csv")
    palestine_source = p[p.country.eq("Palestine")].copy()
    palestine_source["in_main_analysis"] = palestine_source.unit_id.isin(palestine_ids)
    write_csv(palestine_source[["unit_id", "speech_id", "forum", "is_observer", "is_party",
                               "institutional_status", "in_scope", "para_is_procedural", "in_main_analysis",
                               "scope_exclusion_reason"]], adir / "palestine_source_population.csv")
    fallback=ROOT/"data/exports/db/csv/annotations_with_text.csv"
    if fallback.exists():
        exp=pd.read_csv(fallback,low_memory=False)
        rec=[{"source":"selected_database","rows":len(data["evidence"]),"unique_units":data["evidence"].unit_id.nunique(),
              "selected":True,"reason":"explicit ok job status and configured full current runs"},
             {"source":str(fallback.relative_to(ROOT)),"rows":len(exp),"unique_units":exp.unit_id.nunique() if "unit_id" in exp else np.nan,
              "selected":False,"reason":"fallback export lacks the selected consistent current run family; not merged"}]
        write_csv(pd.DataFrame(rec),adir/"source_reconciliation.csv")
    report = f"""# Source, schema, and dependency audit\n\nStatus: **{STATUS}**. Validation status: **pending**.\n\nThe selected source is the configured SQLite job store, read in one read-only transaction. It was preferred over exports because it contains explicit job status, run/version, prompt hash, model, raw response, and every eligible unit. The extended corpus was joined only to recover verified speech/meeting metadata. Existing exports were not merged into the selected data.\n\n`jobs.status = ok` denotes a successfully parsed complete response for a unit. Required dimensions are therefore evaluable even when the response contains an empty annotation array. The conditional violence `attribution` dimension is evaluable only where a violence annotation exists; elsewhere its outcomes are NA, not zero. Failed, missing, and unknown jobs remain NA. Repeated evidence was collapsed only within unit + selected run/version + dimension + value.\n\nThe date window is {cfg['population']['start_date']} through {cfg['population']['end_date']}, and the meeting agenda must name Palestine. The collected manifest contains {eligible_count} eligible meetings; {observed_count} are present in the existing annotated snapshot, leaving {eligible_count-observed_count} eligible but unannotated meetings. `meeting_scope.csv` records every collected meeting and the exclusion or missing-annotation reason. Results describe the annotated subset, not all eligible UN meetings.\n\nThe main population is `role=representative`; {population_label} are included on equal analytical criteria; Palestine retains its non-member observer and conflict-party status. Israel remains included; the Holy See and UN officials remain excluded. RQ2/RQ3 use verified `S/` document symbols (Security Council). Country eligibility is recalculated in `country_eligibility.csv`.\n\nDependencies: Python mixed modelling uses statsmodels' binomial Bayesian mixed GLM because no supported R mixed-model package is installed. Source data and database schemas are never modified. The ignored external kappa CSV and prior kappa values are not dependencies.\n"""
    (adir / "audit_report.md").write_text(report, encoding="utf-8")


def bh_adjust(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index); ok = p.notna(); x = p[ok].sort_values(); n = len(x)
    if n: out.loc[x.index] = np.minimum.accumulate((x.to_numpy() * n / np.arange(1, n + 1))[::-1])[::-1].clip(0, 1)
    return out


def cluster_boot_ci(g: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float]:
    per = g.groupby("speech_id")[col].agg(["sum", "count"]); rng = np.random.default_rng(seed)
    if len(per) < 2 or per["count"].sum() == 0: return (np.nan, np.nan)
    ix = rng.integers(0, len(per), size=(reps, len(per)))
    vals = per.to_numpy()[ix].sum(axis=1); est = vals[:, 0] / vals[:, 1]
    return tuple(np.quantile(est, [0.025, 0.975]))


def rq1(cfg: dict[str, Any], data: dict[str, pd.DataFrame]) -> None:
    d = population(data["matrix"], cfg)
    d = d[d.forum.isin(cfg["population"]["rq1_forums"])]
    cols = [c for c in d if c.startswith("Y__")]
    rows = []
    reps, seed = cfg["analysis"]["prevalence_bootstrap_replicates"], cfg["analysis"]["seed"]
    for i, c in enumerate(cols):
        y = d[c]; lo, hi = cluster_boot_ci(d, c, reps, seed + i)
        cb, dim, code = c.split("__")[1:]
        speech_props = d.groupby("speech_id")[c].mean()
        rows.append({"codebook_id": cb, "dimension": dim, "code": code, "positive_units": int(y.sum()),
                     "evaluable_units": int(y.notna().sum()), "missing_units": int(y.isna().sum()),
                     "paragraph_prevalence": y.mean(), "cluster_boot_ci_low": lo, "cluster_boot_ci_high": hi,
                     "speech_weighted_mean": speech_props.mean(), "speeches_evaluable": speech_props.notna().sum(),
                     "scope": ("UN member states and Palestine" if "Palestine" in cfg["population"].get("included_observers", []) else "UN member states") + " representative paragraphs; verified SC+GA forums", "status": STATUS})
    prev = pd.DataFrame(rows); rdir = outdir(cfg, "rq1_provisional"); write_csv(prev, rdir / "prevalence.csv")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            # within/cross dimensions; always restrict to same jointly observed subset.
            z = d[[a, b]].dropna(); n = len(z)
            if not n: continue
            pa, pb = z[a].mean(), z[b].mean(); joint = (z[a] * z[b]).mean()
            den = math.sqrt(pa * (1-pa) * pb * (1-pb))
            pairs.append({"feature_a": a, "feature_b": b, "joint_n": n, "joint_positive": int((z[a]*z[b]).sum()),
                          "joint_prevalence": joint, "phi": (joint-pa*pb)/den if den else np.nan,
                          "lift": joint/(pa*pb) if pa*pb else np.nan, "sparse": bool((z[a]*z[b]).sum()<10), "status": STATUS})
    write_csv(pd.DataFrame(pairs), rdir / "cooccurrence.csv")
    # Reproducible evidence-review cases: high confidence positives, diversified by country/date.
    ev = data["evidence"].merge(d[["unit_id","speech_id","country","meeting_date","text","role"]], on="unit_id")
    ev = ev.drop_duplicates(["codebook_id","dimension","value","country"])
    picked=[]
    for _,g in ev.groupby(["codebook_id","dimension","value"]):
        g=g.sort_values(["confidence","country","meeting_date"],ascending=[False,True,True])
        hi=g.head(2).copy(); hi["case_role"]="typical_high_confidence_positive"
        lo=g[~g.unit_id.isin(hi.unit_id)].sort_values(["confidence","country","meeting_date"],ascending=[True,True,True]).head(2).copy(); lo["case_role"]="contrasting_lower_confidence_boundary"
        picked += [hi,lo]
    cases=pd.concat(picked,ignore_index=True)
    cases["selection_note"] = "reproducible code-positive evidence review; researcher interpretation required"
    cases["status"] = STATUS
    write_csv(cases, rdir / "evidence_review.csv")
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR",str(outdir(cfg,"audit","matplotlib_cache")))
    os.environ.setdefault("XDG_CACHE_HOME",str(outdir(cfg,"audit","font_cache")))
    import matplotlib.pyplot as plt
    plot=prev[~prev.dimension.eq("attribution")].copy(); plot["label"]=plot.code+" ("+plot.codebook_id+")"
    fig,ax=plt.subplots(figsize=(10,8)); ax.barh(plot.label,plot.paragraph_prevalence,color="#aec4e3",ecolor="#394453",xerr=[plot.paragraph_prevalence-plot.cluster_boot_ci_low,plot.cluster_boot_ci_high-plot.paragraph_prevalence]); ax.set_xlabel("Positive / confirmed evaluable units"); ax.set_title("Galtung code prevalence"); fig.tight_layout(); fig.savefig(rdir/"prevalence.png",dpi=180); plt.close(fig)
    from matplotlib.colors import LinearSegmentedColormap
    core=[c for c in cols if "__attribution__" not in c]; corr=d[core].corr()
    fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap=LinearSegmentedColormap.from_list("pastel_phi",["#ebbeaa","#fbfaf8","#aec4e3"])); ax.set_xticks(range(len(core)),[x.split("__")[-1] for x in core],rotation=90); ax.set_yticks(range(len(core)),[x.split("__")[-1] for x in core]); fig.colorbar(im,ax=ax,label="Phi on jointly evaluable units"); fig.tight_layout(); fig.savefig(rdir/"cooccurrence_phi.png",dpi=180); plt.close(fig)


def country_descriptives(cfg: dict[str, Any], data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])
    elig = pd.read_csv(outdir(cfg,"audit")/"country_eligibility.csv")
    countries = set(elig.loc[elig.included_primary.astype(bool), "country"])
    d = d[d.country.isin(countries)]
    rows=[]
    for country,g in d.groupby("country"):
        for c in [x for x in d if x.startswith("Y__")]:
            y=g[c]; sp=g.groupby("speech_id")[c].mean()
            rows.append({"country":country,"feature":c,"positive_units":int(y.sum()),"evaluable_units":int(y.notna().sum()),
                         "paragraph_weighted":y.mean(),"speech_weighted":sp.mean(),"speeches_evaluable":int(sp.notna().sum()),"status":STATUS})
    return pd.DataFrame(rows)


def fit_mixed_model(frame: pd.DataFrame, outcome: str, cfg: dict[str, Any], variant: str="primary") -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit a proper binomial mixed model with speech and meeting random intercepts."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = frame[[outcome,"country","year_month","n_words","speech_id","meeting_id"]].dropna().copy()
    zero_words=int(d.n_words.le(0).sum())
    d["log_word_count"] = np.log(d.n_words.where(d.n_words.gt(0)))
    d["quarter"] = pd.PeriodIndex(d.year_month, freq="M").asfreq("Q").astype(str)
    d = d.dropna()
    support = d.groupby("country")[outcome].agg(["sum","count"]); support["negative"] = support["count"]-support["sum"]
    keep = support.index[(support["sum"] >= cfg["analysis"]["min_country_positives"]) &
                         (support["negative"] >= cfg["analysis"]["min_country_negatives"])]
    d = d[d.country.isin(keep)].copy()
    meta={"feature":outcome,"variant":variant,"n_units":len(d),"n_speeches":d.speech_id.nunique(),"n_meetings":d.meeting_id.nunique(),
          "n_countries":d.country.nunique(),"positives":int(d[outcome].sum()),"status":"not_run"}
    month_sets=[set(g.year_month) for _,g in d.groupby("country")]
    meta["zero_or_negative_word_counts_excluded"]=zero_words
    meta["time_min"]=d.year_month.min(); meta["time_max"]=d.year_month.max()
    meta["months_observed"]=d.year_month.nunique()
    meta["months_shared_by_all_countries"]=len(set.intersection(*month_sets)) if month_sets else 0
    meta["countries_zero_positive"]="|".join(support.index[support["sum"].eq(0)])
    meta["countries_zero_negative"]="|".join(support.index[support["negative"].eq(0)])
    meta["country_support_screen"]=(f"kept when positives >= {cfg['analysis']['min_country_positives']} and "
                                    f"negatives >= {cfg['analysis']['min_country_negatives']}; one-sided outcomes flagged")
    if d[outcome].sum() < cfg["analysis"]["min_total_positives"] or (1-d[outcome]).sum() < cfg["analysis"]["min_total_negatives"] or d.country.nunique()<2:
        meta.update(status="skipped_support", reason="frequency screen"); return pd.DataFrame(),meta,pd.DataFrame()
    try:
        rhs={"primary":"C(country) + C(year_month) + log_word_count",
             "no_length":"C(country) + C(year_month)",
             "quarter_time":"C(country) + C(quarter) + log_word_count"}[variant]
        model=BinomialBayesMixedGLM.from_formula(f"{outcome} ~ {rhs}",
              {"speech":"0 + C(speech_id)","meeting":"0 + C(meeting_id)"},d)
        # Keep the declared seed and legacy feature identity stable across this naming correction.
        # statsmodels otherwise uses unseeded global NumPy initialization.
        seed_identity = feature_name(outcome, reverse=True)
        initialization_offset=cfg["analysis"].get("optimizer_initialization_offsets", {}).get(outcome, {}).get(variant, 0)
        initialization_seed=cfg["analysis"]["seed"] + sum(map(ord,seed_identity)) + sum(map(ord,variant)) + initialization_offset
        np.random.seed(initialization_seed)
        meta.update(optimizer_initialization_seed=int(initialization_seed),
                    optimizer_initialization_offset=int(initialization_offset))
        print(f"Fitting {outcome} ({variant}): {len(d)} paragraphs, {d.country.nunique()} countries", flush=True)
        optimizer_options={"maxiter":500}
        optimizer_options.update(cfg["analysis"].get("optimizer_options", {}))
        optimizer_options.update(cfg["analysis"].get("optimizer_feature_options", {}).get(outcome, {}))
        result=model.fit_vb(verbose=False, fit_method=rank_two_bfgs, minim_opts=optimizer_options)
        opt=result.optim_retvals
        meta.update(optimizer_status=int(opt.status), optimizer_message=str(opt.message),
                    optimizer_iterations=int(opt.nit), optimizer_objective=float(opt.fun),
                    optimizer_gradient_inf=float(np.max(np.abs(opt.jac))),
                    optimizer_maxiter=int(optimizer_options["maxiter"]),
                    optimizer_gtol=float(optimizer_options.get("gtol", 1e-5)))
        fe=np.asarray(result.fe_mean); sd=np.asarray(result.fe_sd); names=model.exog_names
        # Same reference time/length distribution for every country; random effects fixed to zero.
        import patsy
        design_info=model.data.design_info
        rng=np.random.default_rng(cfg["analysis"]["seed"] + sum(map(ord,seed_identity)))
        draws=rng.normal(fe,sd,size=(cfg["analysis"]["posterior_draws"],len(fe)))
        preds=[]; draw_rows=[]
        for country in sorted(d.country.unique()):
            ref=d[["country","year_month","quarter","log_word_count"]].copy(); ref["country"]=country
            X=np.asarray(patsy.build_design_matrices([design_info],ref,return_type="dataframe")[0])
            point=(1/(1+np.exp(-(X@fe)))).mean()
            sim=(1/(1+np.exp(-(X@draws.T)))).mean(axis=0)
            preds.append({"country":country,"feature":outcome,"adjusted_probability_re0":point,
                          "variant":variant,"ci_low":np.quantile(sim,.025),"ci_high":np.quantile(sim,.975),"reference_units":len(ref),"status":STATUS})
            if variant == "primary":
                draw_rows.append(pd.DataFrame({"draw_id":np.arange(len(sim)), "country":country,
                                               "feature":outcome, "probability":sim}))
        country_ix=[i for i,n in enumerate(names) if n.startswith("C(country)")]
        wald=float(np.sum((fe[country_ix]/sd[country_ix])**2)) if country_ix else np.nan
        pval=float(__import__("scipy").stats.chi2.sf(wald,len(country_ix))) if country_ix else np.nan
        meta.update(status="ok",converged=bool(getattr(result.optim_retvals,"success",result.optim_retvals.get("success",False)) if hasattr(result.optim_retvals,"get") else False),
                    country_wald=wald,country_df=len(country_ix),country_p=pval,
                    method="statsmodels BinomialBayesMixedGLM mean-field variational Bayes; speech+meeting random intercepts; country omnibus is diagonal-posterior Wald approximation",
                    prediction="standardized over common observed time/length distribution; random effects fixed at zero (not marginal population-average)")
        print(f"Finished {outcome} ({variant}): converged={meta['converged']}", flush=True)
        return pd.DataFrame(preds),meta,(pd.concat(draw_rows,ignore_index=True) if draw_rows else pd.DataFrame())
    except Exception as exc:
        meta.update(status="failed",reason=f"{type(exc).__name__}: {exc}"); return pd.DataFrame(),meta,pd.DataFrame()


def rq2(cfg: dict[str, Any], data: dict[str, pd.DataFrame], max_models: int | None=None, workers: int=1) -> None:
    rdir=outdir(cfg,"rq2_provisional"); desc=country_descriptives(cfg,data); write_csv(desc,rdir/"country_prevalence.csv")
    d=population(data["matrix"],cfg,cfg["population"]["rq2_rq3_forum"])
    elig=pd.read_csv(outdir(cfg,"audit")/"country_eligibility.csv"); d=d[d.country.isin(elig.loc[elig.included_primary.astype(bool),"country"])]
    core=[]
    for cb,dim in cfg["core_profile_dimensions"].items(): core += [c for c in d if c.startswith(f"Y__{cb}__{dim}__")]
    pred,diag,posterior=[],[],[]; todo=core if max_models is None else core[:max_models]
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results=list(ex.map(fit_mixed_model,[d]*len(todo),todo,[cfg]*len(todo)))
    else:
        results=[fit_mixed_model(d,c,cfg) for c in todo]
    for p,m,draws in results:
        diag.append(m)
        if len(p): pred.append(p)
        if len(draws) and m.get("converged"): posterior.append(draws)
    for c in core[len(todo):]: diag.append({"feature":c,"status":"deferred_resource_limit","reason":f"--max-models={max_models}"})
    diagnostics=pd.DataFrame(diag)
    if "country_p" in diagnostics: diagnostics["country_p_bh"]=bh_adjust(diagnostics.country_p)
    write_csv(diagnostics,rdir/"model_diagnostics.csv")
    write_csv(pd.concat(pred,ignore_index=True) if pred else pd.DataFrame(),rdir/"adjusted_probabilities.csv")
    if posterior:
        pd.concat(posterior,ignore_index=True).to_parquet(rdir/"adjusted_probability_draws.parquet",index=False)
    # A prespecified Israel-vs-common-reference contrast; avoids data-driven all-pairs testing.
    if pred:
        ap=pd.concat(pred,ignore_index=True); cons=[]
        for feat,g in ap.groupby("feature"):
            if "Israel" in set(g.country):
                ir=g.loc[g.country.eq("Israel")].iloc[0]; others=g.loc[~g.country.eq("Israel")]
                x=float(ir.adjusted_probability_re0); ref=others.adjusted_probability_re0.mean()
                sei=(ir.ci_high-ir.ci_low)/(2*1.959963984540054); seo=np.sqrt(np.square((others.ci_high-others.ci_low)/(2*1.959963984540054)).sum())/len(others)
                se=np.sqrt(sei**2+seo**2); diff=x-ref; z=diff/se if se else np.nan
                cons.append({"feature":feat,"contrast":"Israel minus mean of other eligible countries","difference":diff,
                             "ci_low":diff-1.959963984540054*se,"ci_high":diff+1.959963984540054*se,
                             "p_approx":2*norm.sf(abs(z)) if pd.notna(z) else np.nan,
                             "interval_note":"approximate propagation from mean-field VB intervals; cross-country posterior covariance unavailable",
                             "status":STATUS})
        contrasts=pd.DataFrame(cons); contrasts["p_approx_bh"]=bh_adjust(contrasts.p_approx)
        write_csv(contrasts,rdir/"planned_contrasts.csv")


def rq2_sensitivity(cfg: dict[str, Any], data: dict[str,pd.DataFrame], workers: int=1) -> None:
    d=population(data["matrix"],cfg,cfg["population"]["rq2_rq3_forum"])
    elig=pd.read_csv(outdir(cfg,"audit")/"country_eligibility.csv")
    d=d[d.country.isin(elig.loc[elig.included_primary.astype(bool),"country"])]
    tasks=[(f,v) for f in cfg["analysis"]["sensitivity_features"] for v in ("no_length","quarter_time")]
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results=list(ex.map(fit_mixed_model,[d]*len(tasks),[x[0] for x in tasks],[cfg]*len(tasks),[x[1] for x in tasks]))
    else:
        results=[fit_mixed_model(d,f,cfg,v) for f,v in tasks]
    pred=[p for p,m,_ in results if len(p) and m.get("converged")]; diag=[m for _,m,_ in results]
    rdir=outdir(cfg,"rq2_provisional")
    write_csv(pd.DataFrame(diag),rdir/"sensitivity_model_diagnostics.csv")
    write_csv(pd.concat(pred,ignore_index=True) if pred else pd.DataFrame(),rdir/"sensitivity_adjusted_probabilities.csv")


def rq3(cfg: dict[str, Any], data: dict[str, pd.DataFrame]) -> None:
    rdir=outdir(cfg,"rq3_provisional"); pth=outdir(cfg,"rq2_provisional")/"adjusted_probabilities.csv"
    for stale in (rdir/"BLOCKED.md",rdir/"partial_adjusted_profile_matrix.csv"):
        if stale.exists(): stale.unlink()
    if not pth.exists() or pth.stat().st_size==0:
        (rdir/"BLOCKED.md").write_text("RQ3 adjusted profiles blocked: no successful RQ2 adjusted models.\n",encoding="utf-8"); return
    ap=pd.read_csv(pth); diagnostics=pd.read_csv(outdir(cfg,"rq2_provisional")/"model_diagnostics.csv")
    conv=diagnostics.converged.astype(str).str.lower().eq("true") & diagnostics.status.eq("ok")
    eligible_features=diagnostics.loc[conv,"feature"].tolist()
    matrix=ap[ap.feature.isin(eligible_features)].pivot(index="country",columns="feature",values="adjusted_probability_re0")
    expected=[]
    for cb,dim in cfg["core_profile_dimensions"].items(): expected += [c for c in data["matrix"] if c.startswith(f"Y__{cb}__{dim}__")]
    excluded_features=[c for c in expected if c not in eligible_features]
    matrix=matrix[[c for c in expected if c in eligible_features]]
    complete_countries=matrix.index[matrix.notna().all(axis=1)]; excluded_countries=matrix.index.difference(complete_countries)
    matrix=matrix.loc[complete_countries]
    exclusion_rows=([{"type":"feature","value":c,"reason":"skipped, failed, or non-converged RQ2 model"} for c in excluded_features] +
                    [{"type":"country","value":c,"reason":"at least one retained feature non-estimable under support screen"} for c in excluded_countries])
    write_csv(pd.DataFrame(exclusion_rows),rdir/"profile_exclusions.csv")
    if len(matrix)<6 or matrix.shape[1]<2:
        (rdir/"BLOCKED.md").write_text("RQ3 blocked after transparent complete-case restriction; no zero imputation was used.\n",encoding="utf-8");
        write_csv(matrix.reset_index(),rdir/"partial_adjusted_profile_matrix.csv"); return
    constant=[c for c in matrix if matrix[c].nunique()<2]; matrix=matrix.drop(columns=constant)
    z=pd.DataFrame(StandardScaler().fit_transform(matrix),index=matrix.index,columns=matrix.columns)
    blocks={cb:[c for c in z if c.startswith(f"Y__{cb}__")] for cb in cfg["core_profile_dimensions"]}
    for cb,cols in blocks.items():
        if cols: z[cols]=z[cols]/math.sqrt(len(cols))
    write_csv(matrix.reset_index(),rdir/"adjusted_profile_matrix.csv"); write_csv(z.reset_index(),rdir/"standardized_block_scaled_matrix.csv")
    ncomp=min(len(z)-1,len(z.columns)); pc=PCA(n_components=ncomp).fit(z)
    scores=pd.DataFrame(pc.transform(z),index=z.index,columns=[f"PC{i+1}" for i in range(ncomp)]).reset_index()
    loads=pd.DataFrame(pc.components_.T,index=z.columns,columns=[f"PC{i+1}" for i in range(ncomp)]).reset_index(names="feature")
    write_csv(scores,rdir/"pca_scores.csv"); write_csv(loads,rdir/"pca_loadings.csv")
    write_csv(pd.DataFrame({"component":range(1,ncomp+1),"explained_variance_ratio":pc.explained_variance_ratio_}),rdir/"pca_explained_variance.csv")
    link=linkage(z,method="ward",metric="euclidean"); kval=[]; assignments=[]
    for k in cfg["analysis"]["profile_k"]:
        if k>=len(z): continue
        lab=fcluster(link,k,criterion="maxclust"); sizes=pd.Series(lab).value_counts()
        kval.append({"k":k,"silhouette":silhouette_score(z,lab),"min_cluster_size":int(sizes.min()),
                     "n_singletons":int(sizes.eq(1).sum()),"largest_cluster_share":float(sizes.max()/sizes.sum())})
        assignments += [{"country":c,"k":k,"profile_id":f"P{x}"} for c,x in zip(z.index,lab)]
    write_csv(pd.DataFrame(kval),rdir/"cluster_k_diagnostics.csv"); write_csv(pd.DataFrame(assignments),rdir/"cluster_assignments.csv")
    best_k=int(pd.DataFrame(kval).sort_values("silhouette",ascending=False).iloc[0].k)
    best=pd.DataFrame(assignments); best=best[best.k.eq(best_k)][["country","profile_id"]].set_index("country")
    centroids=matrix.join(best).groupby("profile_id").mean().reset_index(); write_csv(centroids,rdir/"profile_centroids.csv")
    dev=[]
    for country,row in matrix.iterrows():
        pid=best.loc[country,"profile_id"]; center=centroids.set_index("profile_id").loc[pid]
        dev.append({"country":country,"profile_id":pid,"distance_from_centroid":float(np.linalg.norm(row-center)),"best_k":best_k})
    devdf=pd.DataFrame(dev).sort_values(["profile_id","distance_from_centroid"])
    write_csv(devdf,rdir/"representative_deviant_cases.csv")
    chosen=[]
    for pid,g in devdf.groupby("profile_id"):
        chosen.append({"country":g.iloc[0].country,"profile_id":pid,"case_role":"representative_nearest_centroid"})
        chosen.append({"country":g.iloc[-1].country,"profile_id":pid,"case_role":"deviant_farthest_centroid"})
    chosen=pd.DataFrame(chosen)
    scoped = population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])[
                                     ["unit_id","speech_id","country","meeting_date","text"]]
    ev=data["evidence"].merge(scoped,on="unit_id",how="inner")
    ev=ev.merge(chosen,on="country",how="inner").sort_values(["profile_id","case_role","codebook_id","confidence"],ascending=[True,True,True,False])
    ev=ev.groupby(["country","profile_id","case_role","codebook_id"],group_keys=False).head(1)
    ev["status"]=STATUS; write_csv(ev,rdir/"representative_deviant_evidence.csv")
    np.save(rdir/"ward_linkage.npy",link)
    # Unstandardized probability sensitivity.
    rawlink=linkage(matrix,method="ward",metric="euclidean"); np.save(rdir/"ward_linkage_unstandardized.npy",rawlink)
    # Separate unadjusted descriptive sensitivity input; never mixed with adjusted features.
    desc=pd.read_csv(outdir(cfg,"rq2_provisional")/"country_prevalence.csv")
    unadj=desc[desc.feature.isin(matrix.columns) & desc.country.isin(matrix.index)].pivot(index="country",columns="feature",values="paragraph_weighted")
    write_csv(unadj.reset_index(),rdir/"unadjusted_descriptive_profile_sensitivity.csv")
    np.save(rdir/"ward_linkage_unadjusted_descriptive.npy",linkage(unadj,method="ward",metric="euclidean"))
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR",str(outdir(cfg,"audit","matplotlib_cache")))
    os.environ.setdefault("XDG_CACHE_HOME",str(outdir(cfg,"audit","font_cache")))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,7)); dendrogram(link,labels=list(z.index),leaf_rotation=90,ax=ax,link_color_func=lambda _: "#aec4e3"); ax.set_title("Ward dendrogram"); fig.tight_layout(); fig.savefig(rdir/"dendrogram.png",dpi=180); plt.close(fig)
    from un_speeches.research.research_outputs import cluster_diagnostic_figure, knn_similarity_graph
    cache=outdir(cfg,"audit","figure_cache")
    cluster_diagnostic_figure(rdir,cache)
    knn_similarity_graph(rdir,cache,int(cfg["analysis"]["knn_neighbors"]))


def resample_speeches_within_country(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample speech clusters, carrying all paragraphs and assigning unique bootstrap IDs."""
    chunks=[]
    for country,g in base.groupby("country"):
        speeches=g.speech_id.unique(); sampled=rng.choice(speeches,size=len(speeches),replace=True)
        for j,s in enumerate(sampled):
            x=g[g.speech_id.eq(s)].copy(); x["source_speech_id"]=x.speech_id
            x["speech_id"]=x.speech_id+f"__boot{j}"; chunks.append(x)
    return pd.concat(chunks,ignore_index=True)


def bootstrap(cfg: dict[str, Any], data: dict[str, pd.DataFrame], reps: int, start: int=0, workers: int=1) -> None:
    """Speech-within-country bootstrap; each replicate refits every RQ2 model."""
    bdir=outdir(cfg,"rq3_provisional","bootstrap_checkpoints")
    base=population(data["matrix"],cfg,cfg["population"]["rq2_rq3_forum"])
    elig=pd.read_csv(outdir(cfg,"audit")/"country_eligibility.csv"); base=base[base.country.isin(elig.loc[elig.included_primary.astype(bool),"country"])]
    diagnostics=pd.read_csv(outdir(cfg,"rq2_provisional")/"model_diagnostics.csv")
    core=diagnostics.loc[diagnostics.status.eq("ok") & diagnostics.converged.astype(str).str.lower().eq("true"),"feature"].tolist()
    profile_path=outdir(cfg,"rq3_provisional")/"adjusted_profile_matrix.csv"
    signature=sha256(profile_path)
    manifest_path=bdir/"bootstrap_manifest.json"
    manifest={"base_profile_sha256":signature,"features":core,"seed":cfg["analysis"]["seed"],
              "resampling_unit":"speech within country; all paragraphs carried together",
              "model_recipe":"refit RQ2 BinomialBayesMixedGLM, standardize probabilities, standardize features, equal-codebook block scaling"}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("base_profile_sha256")!=signature or old.get("features")!=core or old.get("seed")!=cfg["analysis"]["seed"]:
            raise RuntimeError("Existing bootstrap checkpoints belong to a different base profile/configuration; archive them before starting a new stability run")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for rep in range(start,start+reps):
        path=bdir/f"replicate_{rep:04d}.csv"
        if path.exists(): continue
        rng=np.random.default_rng(cfg["analysis"]["seed"]+rep)
        d=resample_speeches_within_country(base,rng); preds=[]; failed=[]
        if workers>1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results=list(ex.map(fit_mixed_model,[d]*len(core),core,[cfg]*len(core)))
        else:
            results=[fit_mixed_model(d,c,cfg) for c in core]
        for c,(p,m,_) in zip(core,results):
            if len(p) and m.get("converged"):
                preds.append(p)
            else:
                failed.append(f"{c}:{m.get('reason','non-converged' if len(p) else m['status'])}")
        if failed:
            (bdir/f"replicate_{rep:04d}.failed.txt").write_text("\n".join(failed),encoding="utf-8"); continue
        mat=pd.concat(preds).pivot(index="country",columns="feature",values="adjusted_probability_re0")[core]
        z=pd.DataFrame(StandardScaler().fit_transform(mat),index=mat.index,columns=mat.columns)
        for cb in cfg["core_profile_dimensions"]:
            cols=[c for c in z if c.startswith(f"Y__{cb}__")]; z[cols]=z[cols]/math.sqrt(len(cols))
        link=linkage(z,method="ward"); rows=[]
        for k in cfg["analysis"]["profile_k"]:
            if k<len(z): rows += [{"replicate":rep,"country":c,"k":k,"profile_id":int(v)} for c,v in zip(z.index,fcluster(link,k,criterion="maxclust"))]
        write_csv(pd.DataFrame(rows),path)
        failed_path=bdir/f"replicate_{rep:04d}.failed.txt"
        if failed_path.exists(): failed_path.unlink()
    from un_speeches.research.research_outputs import summarize_bootstrap
    summarize_bootstrap(bdir.parent,cfg,outdir(cfg,"audit","figure_cache"))


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("stage",choices=["prepare","audit","validation","rq1","rq2","rq2_sensitivity","rq3","light_stability","latex","figures","report","all"])
    ap.add_argument("--config",default=str(DEFAULT_CONFIG)); ap.add_argument("--max-models",type=int,default=None)
    ap.add_argument("--workers",type=int,default=1)
    return ap.parse_args()


def main() -> int:
    args=parse_args(); cfg=load_config(args.config); data=prepare(cfg) if args.stage in ("prepare","all") else load_prepared(cfg)
    if args.stage in ("audit","all"): audit(cfg,data)
    if args.stage in ("rq1","all"): rq1(cfg,data)
    if args.stage in ("rq2","all"): rq2(cfg,data,args.max_models,args.workers)
    if args.stage in ("rq2_sensitivity","all"): rq2_sensitivity(cfg,data,args.workers)
    if args.stage in ("rq3","all"): rq3(cfg,data)
    if args.stage in ("light_stability","all"):
        from un_speeches.research.light_stability import run_light_stability
        run_light_stability(cfg,data,ROOT/cfg["source"]["output"])
    if args.stage in ("validation","all"):
        from un_speeches.research.validation_pack import export_validation_pack
        export_validation_pack(cfg,data)
    if args.stage=="figures":
        from un_speeches.research.research_outputs import cluster_diagnostic_figure,knn_similarity_graph
        rdir=outdir(cfg,"rq3_provisional"); cache=outdir(cfg,"audit","figure_cache")
        cluster_diagnostic_figure(rdir,cache); knn_similarity_graph(rdir,cache,int(cfg["analysis"]["knn_neighbors"]))
    if args.stage in ("latex","all"):
        from un_speeches.research.research_outputs import export_latex
        export_latex(ROOT/cfg["source"]["output"],ROOT/cfg["latex"]["output"],int(cfg["latex"]["float_decimals"]))
    if args.stage in ("report","all"):
        from un_speeches.research.reporting import write_provisional_report
        write_provisional_report(cfg,data,ROOT)
    return 0


if __name__=="__main__": raise SystemExit(main())
