"""Versioned, reversible naming correction at read boundaries; never edit source jobs.

Only identifier slots and delimited feature names are translated. Prose, evidence,
raw responses, hashes, versions, run IDs, sample IDs and timestamps stay untouched.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd

MAPPING_VERSION = "galtung-identifiers-v1"
CODEBOOK_IDS = {"relations": "coupling", "imperialism": "relational"}
DIMENSION_IDS = {"relation_type": "coupling_type", "imperialism": "relational"}
DISPLAY_NAMES = {"coupling": "Coupling", "relational": "Relational"}


def codebook_id(value):
    return CODEBOOK_IDS.get(value, value) if isinstance(value, str) else value


def dimension_id(value):
    return DIMENSION_IDS.get(value, value) if isinstance(value, str) else value


def source_codebook_id(value):
    return {v: k for k, v in CODEBOOK_IDS.items()}.get(value, value)


def feature_name(value: str, *, reverse: bool = False) -> str:
    """Map complete __-delimited identifier tokens, never substrings in text."""
    mapping = {**CODEBOOK_IDS, **DIMENSION_IDS}
    if reverse:
        mapping = {v: k for k, v in mapping.items()}
    if value.startswith("eval_"):
        return "eval_" + mapping.get(value.removeprefix("eval_"), value.removeprefix("eval_"))
    return "__".join(mapping.get(token, token) for token in value.split("__"))


def normalize_structured(value):
    """Normalize JSON identifier slots/keys, leaving arbitrary string values intact."""
    if isinstance(value, list):
        return [normalize_structured(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        target = feature_name(key)
        if target in out:
            raise ValueError(f"identifier collision: {key!r} -> {target!r}")
        if key in {"codebook_id", "id"}:
            out[target] = codebook_id(item)
        elif key in {"dimension", "name"}:
            out[target] = dimension_id(item)
        elif key in {"stratify_on", "dimension_names"} and isinstance(item, list):
            out[target] = [dimension_id(v) for v in item]
        else:
            out[target] = normalize_structured(item)
    return out


def normalize_output_records(value):
    """Decode one or several JSON records; evidence strings are never rewritten."""
    if not isinstance(value, str) or not value.strip():
        return value
    decoder = json.JSONDecoder()
    remaining, records = value.strip(), []
    try:
        while remaining:
            record, end = decoder.raw_decode(remaining)
            records.append(json.dumps(normalize_structured(record), ensure_ascii=False))
            remaining = remaining[end:].strip()
            if remaining.startswith("|"):
                remaining = remaining[1:].strip()
            elif remaining:
                return value  # uninterpretable historical payload stays verbatim
    except (ValueError, TypeError):
        return value
    return " | ".join(records)


def normalize_frame(frame: pd.DataFrame, *, provenance: bool = True) -> pd.DataFrame:
    out = frame.copy()
    renamed = [feature_name(c) for c in out.columns]
    if len(renamed) != len(set(renamed)):
        raise ValueError("identifier collision in column names")
    out.columns = renamed
    for column, mapping in (("codebook_id", CODEBOOK_IDS), ("dimension", DIMENSION_IDS)):
        if column not in out:
            continue
        values = set(out[column].dropna())
        for old, new in mapping.items():
            if old in values and new in values:
                raise ValueError(f"mixed legacy/corrected identifiers in {column}: {old}, {new}")
        source = f"source_{column}"
        if provenance and source not in out:
            out[source] = out[column]
        out[column] = out[column].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    for column in ("feature", "feature_a", "feature_b", "dimension_a", "dimension_b"):
        if column in out:
            out[column] = out[column].map(lambda v: feature_name(v) if isinstance(v, str) else v)
    if "original_llm_outputs" in out and "normalized_llm_outputs" not in out:
        out["normalized_llm_outputs"] = out.original_llm_outputs.map(normalize_output_records)
    if {"type", "value"}.issubset(out.columns):
        mask = out["type"].eq("feature") & out["value"].astype(str).str.startswith("Y__")
        out.loc[mask, "value"] = out.loc[mask, "value"].map(feature_name)
    if provenance and ("codebook_id" in out or "dimension" in out):
        out["identifier_mapping_version"] = MAPPING_VERSION
    return out


def legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reverse an active view for exact source-content verification."""
    out = frame.drop(columns=[c for c in frame if c.startswith("source_codebook_id")
                             or c in {"source_dimension", "identifier_mapping_version", "normalized_llm_outputs"}]).copy()
    out.columns = [feature_name(c, reverse=True) for c in out.columns]
    for column, mapping in (("codebook_id", CODEBOOK_IDS), ("dimension", DIMENSION_IDS)):
        if column in out:
            inverse = {v: k for k, v in mapping.items()}
            out[column] = out[column].map(lambda v: inverse.get(v, v) if isinstance(v, str) else v)
    for column in ("feature", "feature_a", "feature_b", "dimension_a", "dimension_b"):
        if column in out:
            out[column] = out[column].map(lambda v: feature_name(v, reverse=True) if isinstance(v, str) else v)
    if {"type", "value"}.issubset(out.columns):
        mask = out["type"].eq("feature") & out["value"].astype(str).str.startswith("Y__")
        out.loc[mask, "value"] = out.loc[mask, "value"].map(lambda v: feature_name(v, reverse=True))
    return out


def normalize_config(cfg: dict) -> dict:
    out = deepcopy(cfg)
    for section in ("selected_runs", "dimensions", "core_profile_dimensions"):
        entries = out.get(section, {})
        mapped = {feature_name(k): v for k, v in entries.items()}
        if len(mapped) != len(entries):
            raise ValueError(f"identifier collision in config.{section}")
        if section == "core_profile_dimensions":
            mapped = {k: dimension_id(v) for k, v in mapped.items()}
        out[section] = mapped
    out["analysis"]["sensitivity_features"] = [feature_name(f) for f in out["analysis"].get("sensitivity_features", [])]
    return out


def mapping_manifest() -> dict:
    return {"version": MAPPING_VERSION, "codebook_ids": CODEBOOK_IDS,
            "dimension_ids": DIMENSION_IDS,
            "historical_provenance": "Stored prompts, raw responses, hashes, versions and runs are original; names are corrected on read only."}
