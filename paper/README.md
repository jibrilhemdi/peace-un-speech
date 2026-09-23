# ACM two-column review draft

The paper uses `acmart` with `sigconf,review,anonymous`, following the two-column
layout in the supplied ACM sample PDF while withholding author and venue details.
The PDF is a layout reference; its sample authors, teaser, rights statement, CCS
terms, and bibliography are not part of this study. The empty conference header
is suppressed for this venue-unspecified draft. Add real ACM submission metadata
and CCS terms when a venue is selected.

From the repository root, regenerate **LaTeX assets only** and build the PDF:

```bash
.venv/bin/python -m un_speeches.research.galtung_research latex
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

`paper/main.tex` contains the written Corpus and Measurement and Methods sections.
Its Results section directly inputs four compact generated tables from
`research_outputs/latex/paper/` and includes five current PNG figures from
`research_outputs/rq1_provisional/` and `research_outputs/rq3_provisional/`.
These ignored analysis assets must be present to build the PDF; copying only
`paper/` to another machine is insufficient. The abstract, introduction,
related work, discussion, limitations, and conclusion remain explicit writing
placeholders. Human assessment is pending, so all numbers and visual captions
remain provisional.
