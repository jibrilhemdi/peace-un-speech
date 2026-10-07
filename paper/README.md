# Paper guide

`main.tex` is the anonymous, two-column ACM review draft. The compiled PDF is
`main.pdf`; the current draft has **seven pages**, including references and appendix.

## Structure and ownership

| Section | Lead | Status |
| --- | --- | --- |
| Abstract | — | Drafted |
| Introduction | r | Placeholder |
| Literature Review | r | Placeholder |
| Method | j | Drafted |
| Analysis and Result | Both | Drafted; includes discussion and validation limits |
| Conclusion | — | Drafted |

**Human validation is pending.** Add code-level review findings to the validation
subsection. If errors change a central result, revise the affected claim too.
The AI-assistance statement is in Method.

## Build

From the repository root, with saved analysis outputs, source PDFs, and TeX tools:

```bash
.venv/bin/python -m un_speeches.research.galtung_research latex
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

This regenerates assets and builds the PDF without LLM calls or model refits.
Figures and tables are generated in `research_outputs/latex/paper/`; edit their
Python generators in `src/un_speeches/research/` to make lasting display changes.
When transferring the draft, include that asset directory alongside `paper/`.

The main text uses one figure per research question and three tables. Ten
source-checked excerpts appear in the appendix; extra figures and diagnostics
remain supplementary.

## Finish for submission

- Complete Introduction, Literature Review, and human validation.
- Check the final page count and enter the assigned submission ID.
- Keep the review version anonymous. After acceptance, add author details,
  acknowledgments, and the exact ACM rights metadata, DOI, and ISBN.

CCS concepts, ACM Reference Format, keywords, figure descriptions, and WebSci
conference fields are included. See [venue notes](VENUES.md) for submission
requirements and [research decisions](../docs/DECISIONS.md) for analytical choices.
