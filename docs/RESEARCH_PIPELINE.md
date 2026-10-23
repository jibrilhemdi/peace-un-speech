# Galtung research pipeline

This pipeline uses the frozen repository annotations only. The analysis window and
Palestine-agenda rule are documented in `docs/DECISIONS.md`. It never calls an LLM and opens
the annotation database in read-only mode. All current analytical outputs are provisional and
unvalidated; `validation_status` remains `pending` until returned assessments are reviewed.

## Tested commands

Run from the repository root after `pip install -r requirements.txt` and `pip install -e .`:

```bash
.venv/bin/python -m un_speeches.research.galtung_research prepare
.venv/bin/python -m un_speeches.research.galtung_research audit
.venv/bin/python -m un_speeches.research.galtung_research validation
.venv/bin/python -m un_speeches.research.galtung_research rq1
.venv/bin/python -m un_speeches.research.galtung_research rq2 --workers 3
.venv/bin/python -m un_speeches.research.galtung_research rq3
.venv/bin/python -m un_speeches.research.galtung_research light_stability
.venv/bin/python -m un_speeches.research.galtung_research latex
.venv/bin/python -m unittest discover -s tests -v
```

Reviewer files are deliberately protected from overwrite. Move completed returns outside the
generated reviewer directory before regenerating; the exporter will stop if reviewer files
already exist.

RQ3 uses two quick, distinct sensitivity checks after the corrected RQ2 models finish:

1. The 400 saved joint country-probability draws for each fitted code are clustered again.
   This reflects the approximate mean-field fixed-effect posterior, conditional on the
   fitted models. It does not resample speeches or include full posterior covariance.
2. Speeches are resampled 400 times within each country and descriptive code profiles
   are reclustered. This reflects observed speech-composition uncertainty, not the
   adjusted-model uncertainty.

Run `.venv/bin/python -m un_speeches.research.galtung_research light_stability` after
RQ3. The separate co-clustering CSVs, summary, and status JSON are in
`research_outputs/rq3_provisional/`. The earlier unfinished 500-refit bootstrap and its
checkpoints remain archived under `archive/research_outputs_pre_scope_fix/`; they must
not be described as evidence for the corrected corpus. See decision D11.

RQ3 now also writes `fig_cluster_diagnostics.png` and a supplementary
`fig_knn_similarity_network.png`.
The network's edges use Euclidean distance in the complete standardized/block-scaled adjusted
profile, while PC1--PC2 is used only for node placement. It is a local-similarity visualization,
not a separate clustering or significance test. Change `analysis.knn_neighbors` in
`config/research.yaml` to run a declared sensitivity.

Correspondence analysis is not rerun on the adjusted profile: adjusted probabilities are not a
contingency table, so CA would lack its standard row/column-mass interpretation. The existing
raw-count CA figure may be retained as a clearly separate descriptive sensitivity, but PCA is
the primary ordination for RQ3.

## LaTeX / Overleaf

Generate curated LaTeX mirrors after RQ1--RQ3 and the light stability checks:

```bash
.venv/bin/python -m un_speeches.research.galtung_research latex
```

Copy `research_outputs/latex/` plus the referenced figures into Overleaf. Add
`\\usepackage{booktabs,longtable,graphicx}` to the preamble, then use:

```latex
\input{research_tables.tex}
\input{research_figures.tex}
```

Raw paragraph, evidence, coordinator-key, large profile-matrix, and reviewer CSVs are
intentionally not converted into manuscript tables. They remain research data, not printable
tables. Curated analytical summaries retain the CSV as the canonical machine-readable source.

After both validators return completed files, validate and score by stable `sample_id` (never
row order):

```bash
.venv/bin/python -m un_speeches.research.assessment_scoring \
  --key research_outputs/validation/coordinator_only/coordinator_key.csv \
  --out research_outputs/validation/coordinator_only/agreement_after_review.csv \
  returned_validator_1.xlsx returned_validator_2.xlsx
```

Do not run that command on blank reviewer templates. Diagnostic-supplement metrics are kept
separate from probability-sample metrics. Undefined precision, recall, F1, or kappa remain
undefined.

Configuration, selected runs, fixed seeds, thresholds, and feature blocks are in
`config/research.yaml`. Attribution is supplementary and is excluded from the core profile.
The canonical identifiers are `coupling` / `coupling_type` (display **Coupling**) and
`relational` / `relational` (display **Relational**). The versioned read-boundary mapping
preserves the historical SQLite identifiers, responses, hashes, runs and versions.
The main population includes **UN member states and Palestine**; Palestine remains
a non-member observer and conflict party, while the Holy See and officials remain excluded.
Analysis, evidence selection, paper assets and new validation samples share `population()`.
See [the migration guide](IDENTIFIER_MIGRATION.md) for the two-stage verification and rollback.

The preserved historical pack is `research_outputs/validation/`; its sample IDs, assessments,
coordinator links and original-frame weights are unchanged apart from structured naming.
The separately drawn expanded-population pack is
`research_outputs/validation_member_states_palestine_v1/`. Use that directory's own coordinator
key when scoring its returns. The new probability sample records its exact frame probabilities
and weights; diagnostic tasks remain separate. Do not pool weights between pack versions.
