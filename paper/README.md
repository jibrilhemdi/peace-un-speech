# ACM two-column review draft

The manuscript uses `acmart` with `sigconf,review,anonymous` and includes
CCS XML metadata and displayed concepts for **Law, social and behavioral
sciences** and **Natural language processing**. The ACM Reference Format
block, page numbers, and standard running headers are enabled. The title,
abstract, keywords, numbered ACM bibliography, and figure accessibility
descriptions are present. WebSci 2027's verified title, dates, and location
populate the conference fields; this is still an anonymous review draft.

Before submission, enter the actual submission ID when assigned and complete
the two writing placeholders and human validation. For the accepted version,
add each author's name, affiliation, and email (and ORCID where available),
funding and acknowledgments, and the exact ACM rights-confirmation metadata,
including copyright/license, DOI, ISBN, and any specified proceedings title.
Unassigned publication identifiers remain empty. WebSci's [current call](https://websci27.webscience.org/call-for-papers)
requires two columns and anonymous review, without funding or acknowledgments
in the submitted version. No publication or data-availability claim is added.

The agreed structure is Abstract, Introduction (r), Literature Review (r),
Method (j), Analysis and Result (both), and Conclusion. Introduction and
Literature Review remain placeholders. The other sections contain a working
manuscript, including the connected research questions, collection and
annotation process, human-review design, results, source-based discussion,
and interpretive limits. Discussion and validation are subsections of
Analysis and Result rather than additional top-level sections.

Method also discloses the roles of AI in annotation, analysis and visualization
code, and manuscript drafting/editing. This statement is visible in anonymous
review; the standard `acmart` acknowledgments environment is suppressed in
anonymous mode. ACM's [authorship policy, updated 14 May 2026](https://prod-www.acm.bloomreach.cloud/publications/policies/new-acm-policy-on-authorship),
requires methodological disclosure of AI used in research, including code.
Disclosure of writing assistance is optional under that general policy and is
included here for transparency. Add any other tools actually used before
submission and check the venue's own instructions.

From the repository root, regenerate assets and build:

```bash
.venv/bin/python -m un_speeches.research.galtung_research latex
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

The `latex` stage reads saved scoped outputs, with no LLM calls or statistical
model refitting. It generates four tables and five vector PDF figures in
`research_outputs/latex/paper/`. The main text selects three figures and
three tables:

- RQ1: a compact pastel prevalence chart and the ten leading overlapping
  configurations, including paragraph shares and meeting-record counts.
- RQ2: a pastel within-code standardized heatmap, paired with absolute
  probabilities for five anchor codes to make its relative scale explicit.
- RQ3: a PCA overview and detail view with a printed country key, plus a
  compact stability table. Map colors do not encode clustering assignments.

Ten PDF-checked illustrative excerpts appear in the appendix; three passages
also support the discussion, with source-record citations and page numbers.
The excerpts establish textual provenance and illustrate the full-paragraph
annotations. They do not establish coding accuracy or summarize a country's
entire position. Association and co-clustering figures, full rankings,
probabilities, and diagnostics remain supplementary.

Human validation and adjudication are still pending. Display captions omit
provisional/temporary badges, while the abstract and the validation subsection
state this limitation. Canonical CSVs and audit manifests retain their actual
validation status. When reviews return, update the code-level evidence in
that subsection, keep random-sample and diagnostic findings separate, and
identify any affected claims. Routine additions can stay localized; material
coding errors require changing the corresponding result or interpretation.
No agreement or accuracy values have been invented.

The compiled draft has seven pages including references and the excerpt
appendix. Filling Introduction and Literature Review, adding scholarship,
and reporting validation will change pagination. Check the chosen venue's
limit against the completed manuscript; see [VENUES.md](VENUES.md) for the
existing venue notes.

Generated analysis assets are ignored by Git and must be copied along with
`paper/` when transferring the draft. A copy of `paper/` alone will not build.
