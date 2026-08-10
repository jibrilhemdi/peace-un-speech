"""Codebook loading, validation, prompt rendering and response validation.

This module is the only place that knows what a codebook YAML looks like. Everything it
produces — the prompt, the response validator, the export columns — is derived from the file,
so adding a codebook never requires editing Python. The contract is codebooks/SCHEMA.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

NONE = "NONE"
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CODE_RE = re.compile(r"^[A-Z][A-Z0-9_-]*$")
VALID_UNITS = {"paragraph"}

# Keys the schema defines at each level. Anything else is almost always an authoring slip —
# most often a YAML flow mapping ({code: X, definition: text, with, commas}) whose commas got
# read as key separators, silently truncating the definition. Reported, never ignored.
TOP_KEYS = {
    "id", "version", "title", "theory", "unit", "allow_multiple_annotations",
    "max_annotations_per_unit", "instructions", "dimensions", "worked_examples",
}
DIM_KEYS = {
    "name", "description", "required", "conditional_on", "allow_multiple_values",
    "categories", "scale", "score_aggregation",
}
CAT_KEYS = {
    "code", "label", "definition", "includes", "excludes",
    "positive_examples", "negative_examples", "value",
}
EXAMPLE_KEYS = {"text", "source", "reasoning", "annotation"}
WORKED_KEYS = {"text", "annotations", "reasoning", "source"}


def _unknown(raw: dict, allowed: set[str], where: str, out: list[str]) -> None:
    extra = [k for k in raw if not str(k).startswith("_") and str(k) not in allowed]
    if extra:
        out.append(
            f"{where}: unrecognised key(s) {sorted(map(str, extra))}. "
            "If a definition contains commas, it cannot sit in a YAML flow mapping "
            "({...}) — the commas become key separators and the text is truncated. "
            "Use block style or quote the value."
        )

# Words ignored when measuring definition overlap.
_STOP = frozenset(
    """a an the and or of to in on for by with as at from that this those these is are be been
    being it its their his her not no nor but if then than when where which who whom what any
    all some each other than more most such only own same so very can will just do does did
    doing have has had having there here into onto over under again further once""".split()
)


class CodebookError(RuntimeError):
    pass


# ======================================================================================
# Data model
# ======================================================================================


@dataclass(frozen=True)
class Example:
    text: str
    source: str | None = None
    reasoning: str | None = None
    annotation: dict[str, str] | None = None


@dataclass(frozen=True)
class Category:
    code: str
    label: str
    definition: str
    value: int | None = None      # position on the scale; ordinal dimensions only
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    positive_examples: tuple[Example, ...] = ()
    negative_examples: tuple[Example, ...] = ()


@dataclass(frozen=True)
class Conditional:
    dimension: str
    codes: tuple[str, ...] | None = None  # None -> "any non-NONE value"

    def satisfied_by(self, value: str) -> bool:
        if value == NONE:
            return False
        return True if self.codes is None else value in self.codes

    def describe(self) -> str:
        if self.codes is None:
            return f"only when {self.dimension} is not NONE"
        return f"only when {self.dimension} is one of {', '.join(self.codes)}"


@dataclass(frozen=True)
class Dimension:
    name: str
    categories: tuple[Category, ...]
    description: str | None = None
    required: bool = True
    conditional_on: Conditional | None = None
    allow_multiple_values: bool = False
    scale: str = "nominal"        # nominal | ordinal
    # How several ordinal annotations on one paragraph collapse to a single score in the
    # matrix export. These scales are "how far along" measures, so max is the default.
    score_aggregation: str = "max"

    @property
    def is_ordinal(self) -> bool:
        return self.scale == "ordinal"

    @property
    def ordered(self) -> tuple[Category, ...]:
        """Categories low to high for an ordinal scale; declaration order otherwise."""
        if not self.is_ordinal:
            return self.categories
        return tuple(sorted(self.categories, key=lambda c: (c.value is None, c.value)))

    def value_of(self, code: str) -> int | None:
        return next((c.value for c in self.categories if c.code == code), None)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(c.code for c in self.ordered)

    def allows_none(self) -> bool:
        """NONE is a legal value inside an annotation."""
        return (not self.required) or (self.conditional_on is not None)


@dataclass(frozen=True)
class WorkedExample:
    text: str
    annotations: tuple[dict[str, Any], ...]
    reasoning: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class Codebook:
    id: str
    version: int
    title: str
    unit: str
    dimensions: tuple[Dimension, ...]
    theory: str | None = None
    instructions: str | None = None
    allow_multiple_annotations: bool = True
    max_annotations_per_unit: int | None = None
    worked_examples: tuple[WorkedExample, ...] = ()
    path: Path | None = None

    def dimension(self, name: str) -> Dimension | None:
        return next((d for d in self.dimensions if d.name == name), None)

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    def code_columns(self) -> list[tuple[str, str]]:
        """(dimension, code) pairs, in declaration order — the export matrix columns."""
        return [(d.name, c.code) for d in self.dimensions for c in d.categories]


@dataclass
class LoadResult:
    codebook: Codebook | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.codebook is not None and not self.errors


# ======================================================================================
# Parsing
# ======================================================================================


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _txt(v: Any) -> str:
    return " ".join(str(v).split()) if v is not None else ""


def _parse_examples(raw: Any, where: str, errors: list[str], negative: bool) -> tuple[Example, ...]:
    out: list[Example] = []
    for i, e in enumerate(_as_list(raw)):
        loc = f"{where}[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{loc}: example is not a mapping")
            continue
        _unknown(e, EXAMPLE_KEYS, loc, errors)
        text = _txt(e.get("text"))
        if not text:
            errors.append(f"{loc}: example has no 'text'")
            continue
        reasoning = _txt(e.get("reasoning")) or None
        if negative and not reasoning:
            errors.append(f"{loc}: negative example requires 'reasoning' (say what miscoding costs)")
        ann = e.get("annotation")
        if ann is not None and not isinstance(ann, dict):
            errors.append(f"{loc}: 'annotation' must be a mapping of dimension -> code")
            ann = None
        out.append(
            Example(
                text=text,
                source=_txt(e.get("source")) or None,
                reasoning=reasoning,
                annotation={str(k): str(v) for k, v in ann.items()} if ann else None,
            )
        )
    return tuple(out)


def _parse_conditional(raw: Any, where: str, errors: list[str]) -> Conditional | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return Conditional(dimension=raw)
    if isinstance(raw, dict):
        dim = raw.get("dimension")
        if not dim:
            errors.append(f"{where}.conditional_on: mapping form requires 'dimension'")
            return None
        codes = raw.get("in")
        if codes is not None and not isinstance(codes, (list, tuple)):
            errors.append(f"{where}.conditional_on.in must be a list of codes")
            codes = None
        return Conditional(
            dimension=str(dim),
            codes=tuple(str(c) for c in codes) if codes else None,
        )
    errors.append(f"{where}.conditional_on must be a string or a mapping")
    return None


def parse_codebook(raw: dict[str, Any], path: Path | None = None) -> LoadResult:
    """Parse and hard-validate. Returns errors rather than raising, so the linter can show all."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, dict):
        return LoadResult(None, ["file is not a YAML mapping"], [], path)

    _unknown(raw, TOP_KEYS, "top level", errors)

    cb_id = str(raw.get("id", "") or "")
    if not cb_id:
        errors.append("missing required key 'id'")
    elif not ID_RE.match(cb_id):
        errors.append(f"id {cb_id!r} must match ^[a-z][a-z0-9_]*$")
    elif path is not None and path.stem != cb_id:
        errors.append(f"id {cb_id!r} does not match filename stem {path.stem!r}")

    version = raw.get("version")
    if version is None:
        errors.append("missing required key 'version'")
    elif not isinstance(version, int) or isinstance(version, bool) or version < 1:
        errors.append(f"version must be an integer >= 1, got {version!r}")

    title = _txt(raw.get("title"))
    if not title:
        errors.append("missing required key 'title'")

    unit = str(raw.get("unit", "") or "")
    if not unit:
        errors.append("missing required key 'unit'")
    elif unit not in VALID_UNITS:
        errors.append(f"unit {unit!r} not supported (expected one of {sorted(VALID_UNITS)})")

    raw_dims = raw.get("dimensions")
    if not isinstance(raw_dims, list) or not raw_dims:
        errors.append("'dimensions' must be a non-empty list")
        raw_dims = []

    dims: list[Dimension] = []
    seen_dims: set[str] = set()
    for di, rd in enumerate(raw_dims):
        where = f"dimensions[{di}]"
        if not isinstance(rd, dict):
            errors.append(f"{where}: not a mapping")
            continue
        name = str(rd.get("name", "") or "")
        if not name:
            errors.append(f"{where}: missing 'name'")
            continue
        where = f"dimension '{name}'"
        _unknown(rd, DIM_KEYS, where, errors)
        if not ID_RE.match(name):
            errors.append(f"{where}: name must match ^[a-z][a-z0-9_]*$")
        if name in seen_dims:
            errors.append(f"{where}: duplicate dimension name")
        seen_dims.add(name)

        scale = str(rd.get("scale", "nominal") or "nominal")
        if scale not in ("nominal", "ordinal"):
            errors.append(f"{where}: scale must be 'nominal' or 'ordinal', got {scale!r}")
            scale = "nominal"
        is_ord = scale == "ordinal"

        agg = str(rd.get("score_aggregation", "max") or "max")
        if agg not in ("max", "min", "mean", "first"):
            errors.append(f"{where}: score_aggregation must be max|min|mean|first, got {agg!r}")
            agg = "max"
        if not is_ord and "score_aggregation" in rd:
            errors.append(f"{where}: score_aggregation only applies to an ordinal dimension")

        raw_cats = rd.get("categories")
        if not isinstance(raw_cats, list):
            errors.append(f"{where}: 'categories' must be a list")
            raw_cats = []
        if len(raw_cats) < 2:
            errors.append(
                f"{where}: has {len(raw_cats)} category/categories, needs at least 2 — "
                "a one-category dimension is a boolean flag, not a variable"
            )

        cats: list[Category] = []
        seen_codes: set[str] = set()
        for ci, rc in enumerate(raw_cats):
            cwhere = f"{where} category[{ci}]"
            if not isinstance(rc, dict):
                errors.append(f"{cwhere}: not a mapping")
                continue
            code = str(rc.get("code", "") or "")
            if not code:
                errors.append(f"{cwhere}: missing 'code'")
                continue
            cwhere = f"{where} code '{code}'"
            _unknown(rc, CAT_KEYS, cwhere, errors)
            if code == NONE:
                errors.append(f"{cwhere}: 'NONE' is reserved and implicit — do not declare it")
                continue
            if not CODE_RE.match(code):
                errors.append(f"{cwhere}: code must match ^[A-Z][A-Z0-9_-]*$")
            if code in seen_codes:
                errors.append(f"{cwhere}: duplicate code within this dimension")
            seen_codes.add(code)

            val = rc.get("value")
            if is_ord:
                if val is None:
                    errors.append(f"{cwhere}: ordinal dimension requires an integer 'value'")
                elif not isinstance(val, int) or isinstance(val, bool):
                    errors.append(f"{cwhere}: 'value' must be an integer, got {val!r}")
                    val = None
            elif val is not None:
                errors.append(
                    f"{cwhere}: 'value' is only meaningful on an ordinal dimension "
                    "(set scale: ordinal, or remove it)"
                )
                val = None

            label = _txt(rc.get("label"))
            if not label:
                errors.append(f"{cwhere}: missing 'label'")
            definition = _txt(rc.get("definition"))
            if not definition:
                errors.append(f"{cwhere}: missing or empty 'definition'")
            elif len(definition.split()) < 10:
                warnings.append(
                    f"{cwhere}: definition is only {len(definition.split())} words — "
                    "write the boundary, not a synonym for the label"
                )

            pos = _parse_examples(rc.get("positive_examples"), f"{cwhere}.positive_examples", errors, False)
            neg = _parse_examples(rc.get("negative_examples"), f"{cwhere}.negative_examples", errors, True)
            if not neg:
                warnings.append(
                    f"{cwhere}: no negative_examples — these suppress over-assignment, "
                    "which is the main threat to usable variance"
                )
            if not pos:
                warnings.append(f"{cwhere}: no positive_examples")

            cats.append(
                Category(
                    code=code,
                    label=label,
                    definition=definition,
                    value=val if is_ord else None,
                    includes=tuple(_txt(x) for x in _as_list(rc.get("includes"))),
                    excludes=tuple(_txt(x) for x in _as_list(rc.get("excludes"))),
                    positive_examples=pos,
                    negative_examples=neg,
                )
            )

        # definition overlap within a dimension
        for a in range(len(cats)):
            for b in range(a + 1, len(cats)):
                j = _jaccard(cats[a].definition, cats[b].definition)
                if j >= 0.5:
                    warnings.append(
                        f"{where}: definitions of {cats[a].code} and {cats[b].code} overlap "
                        f"heavily (Jaccard {j:.2f}) — the model will not be able to separate them"
                    )

        if is_ord:
            vals = [c.value for c in cats if c.value is not None]
            dupes = {v for v in vals if vals.count(v) > 1}
            if dupes:
                errors.append(f"{where}: duplicate ordinal value(s) {sorted(dupes)}")
            if vals:
                lo, hi = min(vals), max(vals)
                missing = sorted(set(range(lo, hi + 1)) - set(vals))
                if missing:
                    warnings.append(
                        f"{where}: ordinal values have gaps at {missing} — usually a deleted "
                        "category; re-anchor the scale so the steps are contiguous"
                    )

        req = rd.get("required", True)
        if not isinstance(req, bool):
            errors.append(f"{where}: 'required' must be a boolean")
            req = True
        amv = rd.get("allow_multiple_values", False)
        if not isinstance(amv, bool):
            errors.append(f"{where}: 'allow_multiple_values' must be a boolean")
            amv = False
        if amv and is_ord:
            errors.append(f"{where}: allow_multiple_values is not permitted on an ordinal scale "
                          "— a position on a scale is one value")
            amv = False

        dims.append(
            Dimension(
                name=name,
                categories=tuple(cats),
                description=_txt(rd.get("description")) or None,
                required=req,
                conditional_on=_parse_conditional(rd.get("conditional_on"), where, errors),
                allow_multiple_values=amv,
                scale=scale,
                score_aggregation=agg,
            )
        )

    by_name = {d.name: d for d in dims}

    # conditional_on referential integrity + cycles
    for d in dims:
        c = d.conditional_on
        if c is None:
            continue
        parent = by_name.get(c.dimension)
        if parent is None:
            errors.append(f"dimension '{d.name}': conditional_on unknown dimension '{c.dimension}'")
            continue
        if parent.name == d.name:
            errors.append(f"dimension '{d.name}': conditional_on itself")
        if c.codes:
            unknown = [x for x in c.codes if x not in parent.codes]
            if unknown:
                errors.append(
                    f"dimension '{d.name}': conditional_on.in references codes not in "
                    f"'{parent.name}': {unknown}"
                )
    cyc = _conditional_cycle(dims)
    if cyc:
        errors.append(f"conditional_on cycle: {' -> '.join(cyc)}")

    # codes reused across dimensions
    seen_global: dict[str, str] = {}
    for d in dims:
        for c in d.categories:
            if c.code in seen_global and seen_global[c.code] != d.name:
                warnings.append(
                    f"code '{c.code}' appears in both '{seen_global[c.code]}' and '{d.name}' — "
                    "legal, but ambiguous when you talk about results"
                )
            seen_global[c.code] = d.name

    # example annotation maps
    for d in dims:
        for c in d.categories:
            for kind, exs in (("positive", c.positive_examples), ("negative", c.negative_examples)):
                for e in exs:
                    if not e.annotation:
                        continue
                    for k, v in e.annotation.items():
                        target = by_name.get(k)
                        if target is None:
                            errors.append(
                                f"dimension '{d.name}' code '{c.code}' {kind} example: "
                                f"annotation references unknown dimension '{k}'"
                            )
                        elif v != NONE and v not in target.codes:
                            errors.append(
                                f"dimension '{d.name}' code '{c.code}' {kind} example: "
                                f"annotation gives '{k}: {v}', not a code of '{k}'"
                            )

    # worked examples
    worked: list[WorkedExample] = []
    for wi, rw in enumerate(_as_list(raw.get("worked_examples"))):
        where = f"worked_examples[{wi}]"
        if not isinstance(rw, dict):
            errors.append(f"{where}: not a mapping")
            continue
        _unknown(rw, WORKED_KEYS, where, errors)
        text = _txt(rw.get("text"))
        if not text:
            errors.append(f"{where}: missing 'text'")
            continue
        if "annotations" not in rw:
            errors.append(f"{where}: missing 'annotations' (use [] for the no-code case)")
            continue
        anns = _as_list(rw.get("annotations"))
        clean: list[dict[str, Any]] = []
        for ai, a in enumerate(anns):
            if not isinstance(a, dict):
                errors.append(f"{where}.annotations[{ai}]: not a mapping")
                continue
            for k, v in a.items():
                if k in ("evidence", "confidence"):
                    continue
                target = by_name.get(k)
                if target is None:
                    errors.append(f"{where}.annotations[{ai}]: unknown dimension '{k}'")
                elif str(v) != NONE and str(v) not in target.codes:
                    errors.append(f"{where}.annotations[{ai}]: '{k}: {v}' is not a code of '{k}'")
                elif str(v) == NONE and not target.allows_none():
                    errors.append(
                        f"{where}.annotations[{ai}]: '{k}' is required and has no conditional_on, "
                        "so NONE is not a legal value"
                    )
            for d in dims:
                if d.name not in a and d.required and d.conditional_on is None:
                    errors.append(
                        f"{where}.annotations[{ai}]: required dimension '{d.name}' is missing"
                    )
            clean.append({str(k): v for k, v in a.items()})
        worked.append(
            WorkedExample(
                text=text,
                annotations=tuple(clean),
                reasoning=_txt(rw.get("reasoning")) or None,
                source=_txt(rw.get("source")) or None,
            )
        )

    if not worked:
        warnings.append(
            "no worked_examples — add at least one with 'annotations: []'. It is the single "
            "most effective guard against inflated prevalence (SCHEMA.md §5)"
        )
    elif not any(len(w.annotations) == 0 for w in worked):
        warnings.append(
            "no worked example with 'annotations: []' — without one the model infers that "
            "returning nothing is a failure and starts reaching"
        )

    ama = raw.get("allow_multiple_annotations", True)
    if not isinstance(ama, bool):
        errors.append("'allow_multiple_annotations' must be a boolean")
        ama = True
    mapu = raw.get("max_annotations_per_unit")
    if mapu is not None and (not isinstance(mapu, int) or isinstance(mapu, bool) or mapu < 1):
        errors.append("'max_annotations_per_unit' must be an integer >= 1")
        mapu = None

    if errors:
        return LoadResult(None, errors, warnings, path)

    cb = Codebook(
        id=cb_id,
        version=int(version),  # type: ignore[arg-type]
        title=title,
        unit=unit,
        dimensions=tuple(dims),
        theory=_txt(raw.get("theory")) or None,
        instructions=_txt(raw.get("instructions")) or None,
        allow_multiple_annotations=ama,
        max_annotations_per_unit=mapu,
        worked_examples=tuple(worked),
        path=path,
    )
    return LoadResult(cb, errors, warnings, path)


def _jaccard(a: str, b: str) -> float:
    ta = {w for w in re.findall(r"[a-z]+", a.lower()) if w not in _STOP and len(w) > 2}
    tb = {w for w in re.findall(r"[a-z]+", b.lower()) if w not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _conditional_cycle(dims: Iterable[Dimension]) -> list[str] | None:
    edges = {d.name: (d.conditional_on.dimension if d.conditional_on else None) for d in dims}
    for start in edges:
        seen, node = [], start
        while node is not None:
            if node in seen:
                return seen[seen.index(node):] + [node]
            seen.append(node)
            node = edges.get(node)
    return None


def load_codebook(path: Path) -> LoadResult:
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return LoadResult(None, [f"YAML parse error: {exc}"], [], path)
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    return parse_codebook(raw, path)


def load_all(directory: Path) -> list[LoadResult]:
    files = sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))
    return [load_codebook(p) for p in files]


def require_codebook(directory: Path, codebook_id: str) -> Codebook:
    """Load one codebook or die with the validation errors."""
    path = directory / f"{codebook_id}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in directory.glob("*.yaml"))) or "(none)"
        raise CodebookError(f"no codebook '{codebook_id}' in {directory}. Available: {available}")
    res = load_codebook(path)
    if not res.ok:
        detail = "\n".join(f"  - {e}" for e in res.errors)
        raise CodebookError(
            f"codebook '{codebook_id}' failed validation:\n{detail}\n"
            "Run: python scripts/codebook/validate.py"
        )
    assert res.codebook is not None
    return res.codebook


# ======================================================================================
# Prompt rendering
#
# Order matters: everything static comes first so the prefix is byte-identical across every
# request for a codebook version and prompt-prefix caching applies. The speech is last.
# ======================================================================================

SYSTEM_PROMPT = (
    "You are a meticulous research assistant performing deductive content analysis for an "
    "academic study. You apply a fixed codebook to text exactly as written. You do not "
    "infer beyond the text, you do not use outside knowledge about the speaker or the "
    "events, and you do not soften or strengthen what a speaker said. You return only JSON."
)


def _fmt_example(e: Example, n: int, cb: Codebook) -> str:
    lines = [f"    {n}. \"{e.text}\""]
    if e.source:
        lines.append(f"       Source: {e.source}")
    if e.annotation and len(cb.dimensions) > 1:
        lines.append(f"       Full label: {json.dumps(e.annotation, ensure_ascii=False)}")
    if e.reasoning:
        lines.append(f"       Why: {e.reasoning}")
    return "\n".join(lines)


def render_prefix(cb: Codebook) -> str:
    """The static part of the prompt: everything except the speech."""
    dim_names = ", ".join(cb.dimension_names)
    p: list[str] = []

    p.append("# TASK")
    p.append(
        "Apply the codebook below to each numbered paragraph of the speech at the end of this "
        "message. Work paragraph by paragraph, independently. Judge only what is written in "
        "that paragraph."
    )
    p.append("")
    p.append("# HOW ANNOTATION WORKS")
    p.append(
        f"Each paragraph carries ZERO OR MORE annotations. An annotation is one complete "
        f"labelling event: one value for each of the {len(cb.dimensions)} dimension(s) "
        f"({dim_names}), plus an evidence span quoted verbatim from that paragraph."
    )
    p.append("")
    p.append(
        "MOST PARAGRAPHS RECEIVE NO ANNOTATION. Security Council speeches are largely "
        "procedural, ceremonial or hortatory. Returning an empty list for a paragraph is the "
        "correct and expected outcome, not a failure. Do not reach for a code because a "
        "paragraph is about the conflict — code it only when it clearly meets a definition "
        "below. Over-assignment destroys this study; a missed borderline case does not."
    )
    if cb.allow_multiple_annotations:
        p.append("")
        p.append(
            "If one paragraph does two distinct things that each meet a definition, return two "
            "separate annotations, each with its own evidence span. Do not blend them into one."
        )
    else:
        p.append("")
        p.append("Return at most ONE annotation per paragraph for this codebook.")
    if cb.max_annotations_per_unit:
        p.append(f"Never return more than {cb.max_annotations_per_unit} annotations for one paragraph.")

    p.append("")
    p.append(f"# CODEBOOK: {cb.title}")
    if cb.theory:
        p.append(f"Theoretical basis: {cb.theory}")
    p.append(f"Unit of analysis: {cb.unit}")

    for i, d in enumerate(cb.dimensions, 1):
        p.append("")
        p.append(f"## Dimension {i} of {len(cb.dimensions)}: {d.name}")
        if d.description:
            p.append(d.description)
        rules = []
        if d.is_ordinal:
            lo, hi = d.ordered[0], d.ordered[-1]
            rules.append(
                f"This is an ORDERED SCALE running from {lo.value} ({lo.label}) to "
                f"{hi.value} ({hi.label}). The steps are ranked: a higher value is further "
                f"along the scale, not merely different."
            )
            if cb.allow_multiple_annotations:
                rules.append(
                    "Assign the step that matches each distinct position the paragraph takes. "
                    "If a paragraph takes two different positions — for example demanding a "
                    "ceasefire in one sentence and structural change in another — return two "
                    "annotations with their own steps and their own evidence, rather than "
                    "averaging them into one."
                )
            else:
                rules.append("Choose the single step that best matches the paragraph.")
            rules.append(
                f"A score of {lo.value} is a REAL SCORE, not an absence. It means the "
                f"paragraph does engage this construct, at its lowest level. If the paragraph "
                f"says nothing about {d.name} at all, return no annotation for it — do not "
                f"score it {lo.value}."
            )
        if d.conditional_on:
            rules.append(f"Assign {d.name} {d.conditional_on.describe()}.")
        if d.required and d.conditional_on is None:
            rules.append(f"Every annotation must give exactly one {d.name}. NONE is not valid here.")
        elif d.required:
            rules.append(f"When elicited, {d.name} must be one of the codes below, never NONE.")
        else:
            rules.append(
                f"NONE is a valid and meaningful value for {d.name}. Use it when the paragraph "
                f"meets the other dimension(s) but gives no basis to choose a {d.name} code. "
                "Do not guess, and do not use knowledge from outside the paragraph."
            )
        if d.allow_multiple_values:
            rules.append(f"{d.name} may take a list of codes in one annotation.")
        p.append(" ".join(rules))
        if d.is_ordinal:
            p.append(f"Valid values for {d.name}, lowest to highest: "
                     + ", ".join(f"{c.code} (={c.value})" for c in d.ordered)
                     + (", NONE" if d.allows_none() else ""))
        else:
            p.append(f"Valid values for {d.name}: " + ", ".join(d.codes)
                     + (", NONE" if d.allows_none() else ""))

        for c in d.ordered:
            p.append("")
            if d.is_ordinal:
                p.append(f"### {c.code} — step {c.value}: {c.label}")
            else:
                p.append(f"### {c.code} — {c.label}")
            p.append(f"Definition: {c.definition}")
            if c.includes:
                p.append(f"Counts as {c.code}:")
                p.extend(f"  - {x}" for x in c.includes)
            if c.excludes:
                p.append(f"Does NOT count as {c.code}:")
                p.extend(f"  - {x}" for x in c.excludes)
            if c.positive_examples:
                p.append(f"  Examples of {c.code}:")
                p.extend(_fmt_example(e, n, cb) for n, e in enumerate(c.positive_examples, 1))
            if c.negative_examples:
                p.append(f"  NOT {c.code} — do not code these:")
                p.extend(_fmt_example(e, n, cb) for n, e in enumerate(c.negative_examples, 1))

    if cb.instructions:
        p.append("")
        p.append("# ADDITIONAL INSTRUCTIONS")
        p.append(cb.instructions)

    if cb.worked_examples:
        p.append("")
        p.append("# WORKED EXAMPLES (whole paragraphs)")
        for i, w in enumerate(cb.worked_examples, 1):
            p.append("")
            p.append(f"Paragraph {i}: \"{w.text}\"")
            p.append(f"Correct output: {json.dumps(list(w.annotations), ensure_ascii=False)}")
            if w.reasoning:
                p.append(f"Why: {w.reasoning}")

    p.append("")
    p.append("# OUTPUT FORMAT")
    p.append(_render_output_spec(cb))
    return "\n".join(p)


def _render_output_spec(cb: Codebook) -> str:
    fields = []
    for d in cb.dimensions:
        opts = " | ".join(f'"{c}"' for c in d.codes)
        if d.allows_none():
            opts += ' | "NONE"'
        fields.append(f'  "{d.name}": {opts}')
    fields.append('  "evidence": "<span copied verbatim from that paragraph>"')
    fields.append('  "confidence": <number between 0 and 1>')

    example_ann = {}
    for d in cb.dimensions:
        example_ann[d.name] = d.codes[0] if d.codes else NONE
    example_ann["evidence"] = "exact words from the paragraph"
    example_ann["confidence"] = 0.8

    return "\n".join(
        [
            "Return ONE JSON object and nothing else. No markdown fences, no commentary.",
            "",
            "The object's keys are the paragraph numbers exactly as given below, as strings.",
            "EVERY paragraph number must appear as a key, including those with no annotation.",
            "Each value is a list of annotation objects. An empty list means no annotation.",
            "",
            "Each annotation object has these fields:",
            *fields,
            "",
            "'evidence' must be copied verbatim from that same paragraph — the exact characters,",
            "not a paraphrase and not text from a different paragraph. Use ... only to elide the",
            "middle of a long span. If you cannot quote it, do not assign the code.",
            "",
            "Example of the shape (values are illustrative only):",
            json.dumps({"1": [], "2": [example_ann], "3": []}, ensure_ascii=False, indent=2),
        ]
    )


def render_speech(paragraphs: list[str]) -> str:
    """The variable part of the prompt. Always last, so the prefix stays cacheable."""
    lines = ["# SPEECH", ""]
    for i, text in enumerate(paragraphs, 1):
        lines.append(f"[{i}] {text}")
        lines.append("")
    lines.append(
        f"Return the JSON object now, with exactly these {len(paragraphs)} keys: "
        + ", ".join(f'"{i}"' for i in range(1, len(paragraphs) + 1))
    )
    return "\n".join(lines)


def prefix_hash(cb: Codebook) -> str:
    """Hash of the static prompt + system prompt.

    This is what detects prompt drift mid-corpus: it is identical for every unit of a codebook
    version, so a change in it between speeches is a measurement-validity problem. The
    per-request hash (which includes the speech) is stored separately as request_hash.
    """
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode())
    h.update(b"\x00")
    h.update(render_prefix(cb).encode())
    return h.hexdigest()[:16]


def request_hash(prefix: str, speech_block: str) -> str:
    h = hashlib.sha256()
    h.update(prefix.encode())
    h.update(b"\x00")
    h.update(speech_block.encode())
    return h.hexdigest()[:16]


# ======================================================================================
# Response validation
#
# DEVIATION FROM SPEC: the brief asked for a pydantic model generated from the codebook YAML.
# This is hand-rolled instead, for one reason: every message raised here is fed straight back
# to the model as the repair instruction on the second attempt. Repair quality depends on the
# error text naming the offending key, the offending value, and the valid alternatives —
# "Annotation 0 of key \"3\": \"violence_type\" is \"VX\", which is not a valid code. Valid
# values: VD, VS, VC." A pydantic ValidationError says roughly "Input should be 'VD', 'VS' or
# 'VC' [type=literal_error]" with a loc tuple, which repairs worse and costs a request. The
# rest of what pydantic would give — dynamic construction from the YAML, strict field types,
# cross-field rules — is provided below and is equally derived from the codebook, so adding a
# codebook still requires no code change.
#
# The semantics are still declarative: valid codes, NONE-admissibility, conditional_on and
# annotation limits all come from the Codebook object, never from constants.
# ======================================================================================

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_ELLIPSIS = re.compile(r"\.\.\.|…")


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE.sub("", t)
    # Some models prepend prose; take the outermost JSON object.
    start, end = t.find("{"), t.rfind("}")
    if start > 0 or (end != -1 and end < len(t) - 1):
        if start != -1 and end > start:
            t = t[start : end + 1]
    return t.strip()


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return " ".join(s.lower().split())


def verify_evidence(evidence: str, paragraph: str) -> bool:
    """True when the span is genuinely from this paragraph.

    Exact match after whitespace/quote normalisation, or — for spans the model elided with
    '...' — every fragment present, in order.
    """
    ev, para = _norm(evidence), _norm(paragraph)
    if not ev:
        return False
    if ev in para:
        return True
    frags = [f.strip() for f in _ELLIPSIS.split(ev) if len(f.strip()) >= 8]
    if not frags:
        return False
    pos = 0
    for f in frags:
        idx = para.find(f, pos)
        if idx == -1:
            return False
        pos = idx + len(f)
    return True


@dataclass
class ParsedAnnotation:
    unit_index: int          # 1-based paragraph number as sent
    annotation_index: int    # position within that paragraph's list
    values: dict[str, str]   # dimension -> code (NONE included)
    evidence: str
    confidence: float | None
    evidence_verified: bool


class ResponseError(ValueError):
    """Raised for a response that cannot be trusted. Message is fed back on the repair attempt."""


def validate_response(
    cb: Codebook,
    payload: Any,
    paragraphs: list[str],
) -> list[ParsedAnnotation]:
    """Validate a decoded response against the codebook. Raises ResponseError with a message
    written to be handed straight back to the model as a repair instruction."""
    if not isinstance(payload, dict):
        raise ResponseError(f"Top level must be a JSON object, got {type(payload).__name__}.")

    expected = {str(i) for i in range(1, len(paragraphs) + 1)}
    got = {str(k) for k in payload}
    if got != expected:
        missing = sorted(expected - got, key=int)
        extra = sorted(got - expected)
        parts = []
        if missing:
            parts.append(f"missing keys {missing}")
        if extra:
            parts.append(f"unexpected keys {extra}")
        # A misaligned batch would silently mislabel a whole speech. Never accept it.
        raise ResponseError(
            "Paragraph keys do not match what was sent: "
            + "; ".join(parts)
            + f". Return exactly the keys \"1\"..\"{len(paragraphs)}\", one per paragraph."
        )

    out: list[ParsedAnnotation] = []
    for i in range(1, len(paragraphs) + 1):
        raw_list = payload[str(i)]
        if raw_list is None:
            raw_list = []
        if not isinstance(raw_list, list):
            raise ResponseError(
                f'Value for key "{i}" must be a list of annotation objects (use [] for none), '
                f"got {type(raw_list).__name__}."
            )
        if not cb.allow_multiple_annotations and len(raw_list) > 1:
            raise ResponseError(
                f'Key "{i}" has {len(raw_list)} annotations but this codebook allows at most one.'
            )
        if cb.max_annotations_per_unit and len(raw_list) > cb.max_annotations_per_unit:
            raise ResponseError(
                f'Key "{i}" has {len(raw_list)} annotations, more than the maximum '
                f"{cb.max_annotations_per_unit}."
            )

        for j, ann in enumerate(raw_list):
            if not isinstance(ann, dict):
                raise ResponseError(f'Annotation {j} of key "{i}" is not an object.')

            # An annotation naming none of this codebook's dimensions is not an empty
            # annotation — it is an answer to a different question. Never drop it silently:
            # doing so would record "no codes found" for a speech the model mislabelled.
            if not any(d.name in ann for d in cb.dimensions):
                raise ResponseError(
                    f'Annotation {j} of key "{i}" contains none of this codebook\'s fields '
                    f"({', '.join(cb.dimension_names)}). Got keys: {sorted(map(str, ann))}."
                )

            values = _coerce_values(cb, ann, i, j)
            if all(v == NONE for v in values.values()):
                # An all-NONE annotation is the empty list written the long way. Drop it
                # rather than storing a row that means nothing, and rather than spending a
                # repair request on a response that was already semantically correct.
                continue
            _enforce_constraints(cb, values, i, j)

            ev = ann.get("evidence")
            if not isinstance(ev, str) or not ev.strip():
                raise ResponseError(
                    f'Annotation {j} of key "{i}" has no "evidence". Every annotation needs a '
                    "span copied verbatim from that paragraph."
                )
            conf = ann.get("confidence")
            if isinstance(conf, bool) or not isinstance(conf, (int, float)):
                conf = None
            else:
                conf = min(1.0, max(0.0, float(conf)))

            out.append(
                ParsedAnnotation(
                    unit_index=i,
                    annotation_index=j,
                    values=values,
                    evidence=ev.strip(),
                    confidence=conf,
                    evidence_verified=verify_evidence(ev, paragraphs[i - 1]),
                )
            )
    return out


def _coerce_values(cb: Codebook, ann: dict, key: int, j: int) -> dict[str, str]:
    """Read one value per dimension, rejecting codes that do not exist.

    Deliberately does NOT enforce `required` or `conditional_on` — those are checked after
    an all-NONE annotation has had the chance to be dropped as legitimately empty.
    """
    values: dict[str, str] = {}
    for d in cb.dimensions:
        raw = ann.get(d.name, NONE)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raw = NONE
        if isinstance(raw, list):
            if not d.allow_multiple_values:
                raise ResponseError(
                    f'Annotation {j} of key "{key}": "{d.name}" is a list, but this dimension '
                    "takes exactly one value. Emit separate annotations instead."
                )
            raw = raw[0] if raw else NONE
        val = str(raw).strip()
        if val.upper() == NONE:
            val = NONE
        if val != NONE and val not in d.codes:
            raise ResponseError(
                f'Annotation {j} of key "{key}": "{d.name}" is "{val}", which is not a valid '
                f"code. Valid values: {', '.join(d.codes)}"
                + (", NONE" if d.allows_none() else "")
                + "."
            )
        values[d.name] = val
    return values


def _enforce_constraints(cb: Codebook, values: dict[str, str], key: int, j: int) -> None:
    for d in cb.dimensions:
        v = values[d.name]
        if d.conditional_on is not None:
            parent = values.get(d.conditional_on.dimension, NONE)
            if not d.conditional_on.satisfied_by(parent) and v != NONE:
                raise ResponseError(
                    f'Annotation {j} of key "{key}": "{d.name}" is "{v}", but it may only be '
                    f"assigned {d.conditional_on.describe()} "
                    f'(here {d.conditional_on.dimension} is "{parent}").'
                )
        if v == NONE and not d.allows_none():
            raise ResponseError(
                f'Annotation {j} of key "{key}": "{d.name}" is required and cannot be NONE. '
                f"If nothing applies, omit the annotation entirely."
            )
