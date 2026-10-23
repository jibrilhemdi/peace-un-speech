# Constructing Peace in UN Discourse on the Palestine Question

This project studies how UN member states and Palestine describe peace on the Palestine
question, using five Galtungian codebooks and LLM-assisted paragraph annotation.
It examines the shared repertoire, differences between states, and similarities
between their overall discourse profiles.

The study covers **7 October 2023–31 December 2025**, selecting meeting records
whose agendas name Palestine. The analyzed subset contains **93 of 112 eligible
records**, with **15,587 paragraphs from 1,826 representative speeches**.
Country comparisons use the Security Council subset.
Palestine retains its non-member observer and conflict-party status.

The active construct names are **Coupling** (`coupling` / `coupling_type`) and
**Relational** (`relational` / `relational`). Historical annotations keep their
original identifiers and prompt provenance. See the [migration guide](docs/IDENTIFIER_MIGRATION.md)
for checks, reproduction, backup locations, and rollback commands.

## Set up

Use Python 3.11 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Source PDFs, the annotation database, and generated analysis outputs are kept
locally and are **not included in Git**. See [data handling](data/README.md) for
required inputs and backups.

## Build the paper

With the saved analysis outputs and source PDFs present, and a TeX installation providing
`acmart` and `latexmk`:

```bash
.venv/bin/python -m un_speeches.research.galtung_research latex
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

## Find what you need

| Location | Purpose |
| --- | --- |
| [Paper guide](paper/README.md) | Build, edit, and finish the manuscript. |
| [Research pipeline](docs/RESEARCH_PIPELINE.md) | Run statistical analyses and score returned human reviews. |
| [Collection and annotation guide](docs/ANNOTATION_PIPELINE.md) | Collect records, prepare paragraphs, and run new annotation jobs. |
| [Decision log](docs/DECISIONS.md) | Understand corpus rules and analytical choices. |
| [Codebook schema](codebooks/SCHEMA.md) | Define and validate coding categories. |
| [Source code](src/un_speeches/) | Collection, annotation, exports, and research modules. |

Analysis settings are in [research.yaml](config/research.yaml); annotation
settings are in [annotation.yaml](config/annotation.yaml). Current outputs are
in `research_outputs/`. [Archived analyses](archive/README.md) use older designs
and are retained for provenance.
