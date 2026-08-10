#!/usr/bin/env python3
"""Convert a codebook CSV into a draft YAML following codebooks/SCHEMA.md.

    python scripts/codebook/build_from_csv.py codebooks/imperialism.csv
    python scripts/codebook/build_from_csv.py codebooks/*.csv --out-dir codebooks
    python scripts/codebook/build_from_csv.py codebooks/transcend.csv --id transcend --stdout

THIS PRODUCES A DRAFT, NOT A FINISHED CODEBOOK. It does the mechanical work reliably —
splitting numbered examples, pulling `S/PV.xxxx, Speaker (Country)` prefixes into `source`,
normalising country names, detecting an ordinal 0–4 scale — and marks everything that needs
your judgement with `# TODO`. What it cannot do is write a definition you can code against.
The CSV definitions are one-liners; they need boundaries (`includes` / `excludes`) before the
model can draw a line with them. Read every definition before running a pilot.

Expected CSV columns (the first column may be named `Code & Label`, `Code & ATR`,
`Score & Label`, or anything containing "code" or "score"):

    Score & Label | Definition EN | Real Example | Reasoning EN

A first column whose entries begin with a digit ("0 Pure Negatif") is treated as an ordinal
scale and emitted with `scale: ordinal` and a `value:` per category. Anything else is nominal.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config import Config  # noqa: E402

# Country names as they appear in the corpus. The CSVs were authored in Indonesian, so the
# parenthetical after a speaker name needs normalising before it lands in `source`.
COUNTRY_FIX = {
    "rusia": "Russian Federation", "russia": "Russian Federation",
    "palestina": "Palestine", "as": "United States", "amerika serikat": "United States",
    "uea": "United Arab Emirates", "uae": "United Arab Emirates",
    "prancis": "France", "perancis": "France", "inggris": "United Kingdom",
    "mesir": "Egypt", "aljazair": "Algeria", "afrika selatan": "South Africa",
    "jepang": "Japan", "china": "China", "tiongkok": "China",
    "arab saudi": "Saudi Arabia", "yordania": "Jordan", "turki": "Türkiye",
    "israel": "Israel", "malta": "Malta", "ghana": "Ghana", "denmark": "Denmark",
    "mozambique": "Mozambique", "mozambik": "Mozambique", "chile": "Chile",
    "pakistan": "Pakistan", "iraq": "Iraq", "irak": "Iraq",
}

# "1. S/PV.9442, Nebenzia (Rusia):" — the number is optional, the source line is optional.
ITEM_SPLIT = re.compile(r"(?m)^\s*(?=\d+\.\s)")
# The attribution ends either at a colon, or — when the author ran straight into the quote —
# at the opening quote itself. Requiring the colon silently dropped the source on entries
# written as `S/PV.9841, Joyini, South Africa "In conclusion, ..."`, and provenance is the
# whole point of the field.
SOURCE_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?"
    r"(?P<src>(?:S/PV\.[\d]+[^,\n\"“]*,?\s*)?[A-ZÀ-Ý][\w'\-., ]{1,40}?"
    r"(?:\s*\((?P<country>[^)]+)\))?)"
    r"\s*(?:[:：]|(?=[\"“]))",
    re.M,
)
QUOTE_RE = re.compile(r"[\"“”]{1,2}(?P<q>[^\"“”]{12,})[\"“”]{1,2}", re.S)
SCORE_RE = re.compile(r"^\s*(?P<value>\d+)\s+(?P<label>.+?)\s*$")
CODE_RE = re.compile(r"^\s*(?P<code>[A-Z][A-Z0-9_-]*)\s+(?P<label>.+?)\s*$")


def norm_ws(s: str) -> str:
    return " ".join(str(s or "").split())


def fix_country(name: str) -> str:
    return COUNTRY_FIX.get(name.strip().lower(), name.strip())


def normalise_source(src: str) -> str:
    """Normalise the country in an attribution, parenthesised or not.

    Both forms occur in the CSVs: `Nebenzia (Rusia)` and `Joyini, Afrika Selatan`.
    """
    src = norm_ws(src).rstrip(",;")
    m = re.search(r"\(([^)]+)\)\s*$", src)
    if m:
        return src[: m.start()].rstrip() + f" ({fix_country(m.group(1))})"
    # Trailing comma-separated country: "S/PV.9841, Joyini, Afrika Selatan"
    parts = [p.strip() for p in src.split(",")]
    if len(parts) >= 2 and parts[-1].lower() in COUNTRY_FIX:
        return ", ".join(parts[:-1]) + f" ({fix_country(parts[-1])})"
    return src


def parse_examples(cell: str) -> list[dict]:
    """Split a Real Example cell into one entry per numbered example."""
    cell = (cell or "").strip()
    if not cell:
        return []
    chunks = [c for c in ITEM_SPLIT.split(cell) if c.strip()]
    out: list[dict] = []
    for chunk in chunks:
        chunk = chunk.strip()
        src = ""
        m = SOURCE_RE.match(chunk)
        if m:
            src = normalise_source(m.group("src"))
            body = chunk[m.end():]
        else:
            body = chunk
        quotes = QUOTE_RE.findall(body)
        text = norm_ws(quotes[0]) if quotes else norm_ws(re.sub(r"^\s*\d+\.\s*", "", body))
        if not text:
            continue
        e: dict = {"text": text}
        if src:
            e["source"] = src
        out.append(e)
    return out


def block(text: str, indent: int, width: int = 96) -> str:
    """Emit a folded YAML scalar so long definitions stay readable and comma-safe."""
    pad = " " * indent
    words, lines, cur = norm_ws(text).split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width - indent:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ">\n" + "\n".join(pad + ln for ln in lines)


def q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def convert(path: Path, codebook_id: str | None = None) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]
    if not rows:
        raise SystemExit(f"{path}: no data rows")

    cols = list(rows[0])
    first = next((c for c in cols if re.search(r"code|score", c, re.I)), cols[0])
    defcol = next((c for c in cols if re.search(r"definition", c, re.I)), None)
    excol = next((c for c in cols if re.search(r"example", c, re.I)), None)
    rescol = next((c for c in cols if re.search(r"reason", c, re.I)), None)
    if defcol is None:
        raise SystemExit(f"{path}: no 'Definition' column found (columns: {cols})")

    cb_id = codebook_id or re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_")
    ordinal = all(SCORE_RE.match(r[first] or "") for r in rows)
    prefix = "".join(w[0] for w in re.split(r"[_\s]+", cb_id) if w)[:3].upper() or "C"

    cats = []
    for r in rows:
        raw = norm_ws(r[first])
        if ordinal:
            m = SCORE_RE.match(raw)
            value, label = int(m.group("value")), m.group("label")
            code = f"{prefix}{value}"
        else:
            m = CODE_RE.match(raw)
            if not m:
                raise SystemExit(f"{path}: cannot read a code from {raw!r}")
            value, code, label = None, m.group("code"), m.group("label")
        cats.append({
            "code": code, "value": value, "label": label,
            "definition": norm_ws(r.get(defcol)),
            "examples": parse_examples(r.get(excol) or ""),
            "reasoning": norm_ws(r.get(rescol) or ""),
        })

    dim = re.sub(r"[^a-z0-9_]+", "_", cb_id).strip("_")
    L = []
    L.append(f"# DRAFT generated by scripts/codebook/build_from_csv.py from {path.name}.")
    L.append("#")
    L.append("# Before running a pilot, do these three things:")
    L.append("#   1. Rewrite every `definition` with a BOUNDARY, not a restatement of the label.")
    L.append("#      The CSV one-liners are too thin for the model to draw a line with. Add")
    L.append("#      `includes:` / `excludes:` bullets — see codebooks/SCHEMA.md §4.")
    L.append("#   2. Check every `source:` — country names were auto-translated from Indonesian")
    L.append("#      and speaker/meeting attributions are copied verbatim, not verified.")
    L.append("#   3. Add at least one worked_example with `annotations: []`.")
    if ordinal:
        L.append("#")
        L.append("# Detected an ORDINAL scale. Note that the lowest step is a real score, not")
        L.append("# absence — a paragraph that says nothing about this construct gets NO")
        L.append("# annotation at all. See SCHEMA.md §3.3.")
    L.append("")
    L.append(f"id: {cb_id}")
    L.append("version: 1")
    L.append(f"title: {q(cb_id.replace('_', ' ').title())}   # TODO: a real title")
    L.append("theory: >   # TODO: name the source and the construct")
    L.append("  ")
    L.append("unit: paragraph")
    # Ordinal scales follow the same annotation model as the nominal codebooks: a paragraph
    # that takes two positions yields two annotations rather than one averaged score.
    L.append("allow_multiple_annotations: true")
    L.append("max_annotations_per_unit: 4")
    L.append("")
    L.append("instructions: >")
    L.append("  Code the speaker's own framing, not the underlying facts. Most paragraphs of a")
    L.append("  Security Council speech engage no construct at all and correctly receive no")
    L.append("  annotation.")
    if ordinal:
        L.append("")
        L.append(f"  Assign a {dim} step for each distinct position the paragraph takes, and no")
        L.append("  annotation at all when it engages the construct nowhere.")
    L.append("")
    L.append("dimensions:")
    L.append(f"  - name: {dim}")
    if ordinal:
        L.append("    scale: ordinal")
        L.append("    score_aggregation: max   # how several steps on one paragraph collapse")
    L.append("    description: >   # TODO: say what the VARIABLE measures, not what the codes mean")
    L.append("      ")
    L.append("    required: true")
    L.append("    categories:")
    for c in cats:
        L.append("")
        L.append(f"      - code: {c['code']}")
        if c["value"] is not None:
            L.append(f"        value: {c['value']}")
        L.append(f"        label: {q(c['label'])}")
        L.append("        definition: " + block(c["definition"], 10))
        L.append("        # TODO: add includes/excludes boundaries")
        if c["examples"]:
            L.append("        positive_examples:")
            for e in c["examples"]:
                L.append("          - text: " + block(e["text"], 14))
                if e.get("source"):
                    L.append(f"            source: {q(e['source'])}")
                if c["reasoning"]:
                    L.append("            reasoning: " + block(c["reasoning"], 14))
        else:
            L.append("        positive_examples: []   # TODO: none in the CSV")
        L.append("        # TODO: negative_examples — spans that look like this code but are not.")
        L.append("        negative_examples: []")
    L.append("")
    L.append("# TODO: at least one worked example with `annotations: []`, and one showing a")
    L.append("# fully-formed annotation. See SCHEMA.md §5.")
    L.append("worked_examples: []")
    return cb_id, "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: codebooks/ from config")
    ap.add_argument("--id", default=None, help="codebook id (only with a single input)")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing YAML")
    args = ap.parse_args()

    if args.id and len(args.csv) > 1:
        print("--id can only be used with a single CSV", file=sys.stderr)
        return 1
    out_dir = args.out_dir or Config.load().path_for("codebooks")

    rc = 0
    for path in args.csv:
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr); rc = 1; continue
        cb_id, text = convert(path, args.id)
        if args.stdout:
            print(text); continue
        target = out_dir / f"{cb_id}.yaml"
        if target.exists() and not args.force:
            print(f"refusing to overwrite {target} (pass --force)", file=sys.stderr)
            rc = 1; continue
        target.write_text(text, encoding="utf-8")
        n_cat = text.count("      - code: ")
        print(f"{path.name}  ->  {target}   ({n_cat} categories, "
              f"{'ordinal' if 'scale: ordinal' in text else 'nominal'})")

    if not args.stdout and rc == 0:
        print("\nDRAFTS ONLY. Next:")
        print("  1. rewrite the definitions with real boundaries, fill every # TODO")
        print("  2. python scripts/codebook/validate.py")
        print("  3. python scripts/annotate/run_annotation.py --codebook <id> --pilot 40")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
