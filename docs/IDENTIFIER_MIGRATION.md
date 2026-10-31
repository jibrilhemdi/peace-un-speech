# Galtung naming and Palestine population migration

Recorded 2026-10-22; execution continued on 2026-10-23. Findings remain **provisional / unvalidated**.

The explicit read-boundary mapping `galtung-identifiers-v1` is defined in
`src/un_speeches/lib/identifiers.py`:

| Original codebook | Active codebook | Original dimension | Active dimension | Display |
| --- | --- | --- | --- | --- |
| `relations` | `coupling` | `relation_type` | `coupling_type` | Coupling |
| `imperialism` | `relational` | `imperialism` | `relational` | Relational |

Category codes, definitions, ordinal values/aggregation, assignments, evidence,
confidence, and all other constructs retain their meanings. Theoretical discussion
and bibliographic titles are preserved. Active codebooks have corrected filenames;
unchanged original YAML/CSV sources are in `archive/codebooks_before_identifier_v1/`.

SQLite is primary. Loading jobs, annotations and pilot metadata produces corrected
identifiers plus `source_codebook_id`, `source_dimension` (where applicable), and the
mapping version. Raw responses, original output bundles, request/prompt hashes, run
IDs, codebook versions and timestamps are untouched. Corrected JSON output bundles
are separate fields; prose and quoted evidence are never searched and replaced.
Historical annotation/status tools recover the original codebook instrument when
working with legacy jobs, preserving prompt hashes rather than invalidating jobs.
No migration command invokes an LLM or changes annotation storage.

## Preserved inputs and existing work

The initial backup is:

```
migration_backups/20261022T232628+0200/
```

It contains 1,910 files (2,014,048,404 bytes), source and backup SHA-256 checksums in
`manifest.json`, the initial Git status, and a binary patch of pre-existing work.
The uncommitted edits in `docs/DECISIONS.md`, `paper/main.tex`, and
`src/un_speeches/research/paper_assets.py` were preserved and incorporated.
SQLite files were snapshotted using `Connection.backup()`, including committed WAL
transactions, and passed `PRAGMA integrity_check`. Physical snapshot bytes can differ
from the original database because the SQLite backup API changes header bytes.
Both hashes are recorded; logical annotations are compared after reversing the mapping.
All raw PDFs, original review materials, interim inputs and annotation databases
remain intact. Existing historical archives were not rewritten.

## Population and coverage

The centralized `population()` selection includes UN member states and Palestine.
Palestine keeps its **non-member observer state** and conflict-party metadata.
Israel remains included; the Holy See and officials remain excluded. The existing
2023-10-07–2025-12-31 window, Palestine-agenda rule, representative role, procedural
exclusions, SC/GA restrictions, selected runs, models, thresholds and configured
seeds are retained.

| Population | Before | After | Added Palestine |
| --- | ---: | ---: | ---: |
| RQ1 paragraphs (SC + GA) | 14,463 | 15,587 | 1,124 |
| RQ1 speeches | 1,751 | 1,826 | 75 |
| SC paragraphs before country thresholds | 14,070 | 15,174 | 1,104 |
| SC Palestine speeches | — | 74 | 74 |
| Primary eligible countries | 26 | 27 | 1 |
| Sensitivity eligible countries | 41 | 42 | 1 |
| Primary eligible SC paragraphs | 10,296 | 11,400 | 1,104 |
| Primary eligible SC speeches | 1,245 | 1,319 | 74 |

All 1,124 eligible Palestine paragraphs have successful evaluations in all five
selected core codebooks. Palestine's SC sample has 1,104 paragraphs, 74 speeches,
and 1,104 evaluated paragraphs per core codebook; it clears both 100/15/20 and
50/10/10 thresholds. Conditional violence attribution retains its own applicable
denominator (723 applicable and 401 not applicable Palestine paragraphs).
The source contains 1,178 Palestine units: 48 nonprocedural units are outside
the date/agenda scope and six procedural units are excluded. No eligible
Palestine core evaluations are missing or failed; excluded/unevaluated procedural
rows do not become negative observations. The unchanged meeting audit identifies 112 eligible collected records,
93 represented in the annotation snapshot, and 19 eligible but unannotated records.
No annotation was generated for those missing records; results describe the annotated subset.
The generic convenience-export builder now follows the research matrix’s existing rule
that inapplicable violence attribution is NA, rather than an observed zero. This corrects
a derived-view discrepancy and does not alter an annotation or an analytical input.

## Reviews and weights

Original `data/gold`, `data/pilots`, `data/samples` and reviewer files remain unchanged.
Corrected copies preserve assessments, stable IDs, coordinator links and original
sampling probabilities/weights. Internal fields determine the mapping: for example,
`2029_relations.csv` contains legacy `imperialism` and becomes `2029_relational.csv`;
`2028_coupling.csv` contains legacy `relations` and keeps its corrected filename.
Original output JSON is retained beside any normalized view.

The corrected historical reviewer pack remains `research_outputs/validation/`.
Corrected gold/pilot/spot-check copies are activated under
`data/review_materials_identifier_v1/`. The expanded-population pack is separately
versioned as `research_outputs/validation_member_states_palestine_v1/`, with its own
coordinator key, frame-specific probabilities, weights, and sampling manifest.
Never reuse historical weights for the expanded frame or pool diagnostic tasks with
probability tasks. Human validation and adjudication remain pending.

## Exact reproduction

Run from the repository root with `.venv` dependencies installed. The tested environment
is Python 3.14.7; exact direct dependency versions are saved in
`config/research_environment_identifier_v1.txt` (including SciPy 1.18.0 and
statsmodels 0.14.6). To recreate it, install that file and then the local package:

```bash
.venv/bin/python -m pip install -r config/research_environment_identifier_v1.txt
.venv/bin/python -m pip install -e .
```

All commands below
are offline research operations. Use the original pre-migration snapshot for the
legacy-code baseline; `backup` creates additional snapshots for later migrations.

```bash
.venv/bin/python -m un_speeches.research.migrate_identifiers --dry-run
.venv/bin/python -m un_speeches.research.migrate_identifiers backup --dry-run
```

The dry run lists affected files, the versioned mapping, staging destination and
expected changes without writing. To create an additional complete WAL-aware snapshot:

```bash
.venv/bin/python -m un_speeches.research.migrate_identifiers backup
```

Reproduce the paired verification and expanded analysis in a fresh staging directory:

```bash
BACKUP='migration_backups/20261022T232628+0200'
STAGE='migration_staging/reproduction_v1'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m un_speeches.research.migrate_identifiers baseline --backup "$BACKUP" --staging "$STAGE"
.venv/bin/python -m un_speeches.research.migrate_identifiers stage --backup "$BACKUP" --staging "$STAGE" --workers 3
.venv/bin/python -m un_speeches.research.migrate_identifiers verify --backup "$BACKUP" --staging "$STAGE"
.venv/bin/python -m un_speeches.research.migrate_identifiers activate --backup "$BACKUP" --staging "$STAGE"
.venv/bin/python -m un_speeches.codebook.validate
.venv/bin/python -m unittest discover -s tests -v
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

`stage` rebuilds prepare/audit, RQ1, RQ2, specification sensitivities, RQ3, both
400-replicate light stability checks, paper/LaTeX assets, corrected review copies,
the separately versioned new validation pack, and CSV/Parquet/Excel exports.
It starts from fresh output directories instead of reusing incompatible model or
bootstrap checkpoints. Completed identical stages can be rerun without redrawing
samples; conflicting input signatures or review targets raise an error.
`activate` reruns verification, retains old output trees under
`STAGE/preactivation/`, and installs the staged outputs. Original input/review folders
and annotation storage are never overwritten. Historical and staged fingerprints
remain recorded when the active prepared-cache signature is aligned with the verified
original database. `paper_values.tex` links manuscript numbers to regenerated outputs.

The initial dense-BFGS attempts used `migration_staging/run_baseline.py` and were
interrupted after identifying the cubic update bottleneck. Their logs are retained as
`*.dense_bfgs_interrupted.log` and supply no final estimates. The completed frozen-code
baseline uses the maintained `baseline` command above; the two conditions can also be
run independently:

```bash
.venv/bin/python -m un_speeches.research.migrate_identifiers stage --only rename_only --workers 3
.venv/bin/python -m un_speeches.research.migrate_identifiers stage --only expanded --workers 3
.venv/bin/python -m un_speeches.research.migrate_identifiers exports
```

The first two separate the spelling correction from the population expansion. Logs,
configs, the frozen-code baseline, both staged result trees, the review-copy manifest,
Palestine coverage and `verification.json` are in `migration_staging/`.

## Verification interpretation

1. **Rename only / original population:** inverse-map and compare prepared annotation
   content, evidence, deduplication, joins, row counts, missing-value patterns, complete
   analysis inputs and codebook content exactly. Refit the same models on the same rows
   and compare numerical CSV results at `rtol=1e-8`, `atol=1e-10`. Row order may differ
   after identifier sorting, so numerical tables are aligned by their categorical keys.
2. **Add Palestine:** require exact equality of prepared observations, retain every
   previously included unit and its values, and permit additions only from eligible
   Palestine representative speech. Check denominators, eligibility, outputs, source
   integrity and review preservation independently.

The pre-migration implementation seeded posterior draws but let statsmodels initialize
variational optimization from unseeded global NumPy state. The paired baseline and
new implementation both pin that initialization to the existing seed plus legacy
feature identity and variant. This preserves the specified model and seed settings
and prevents new spelling from changing random streams. Saved historical unseeded
fits are retained as history and are distinct from this reproducible paired baseline.
The full mixed-model-refit bootstrap remains historically unfinished; the two existing
light checks are recalculated and remain distinct from that uncompleted procedure.

The historical RQ3 evidence picker used all in-scope forums, although country models
use SC records. Central selection corrects this in illustrative evidence only: one
rename-only Algeria/violence example from `A/78/PV.40` (GA, unit
`d2da88272eb07b53`) is replaced by an eligible SC example from `S/PV.10000`
(unit `71bd4cd70c14a37f`). The numerical inputs, country samples and annotation
assignments do not change. Verification separately checks the replacement’s exact
original evidence, value, confidence, and annotation provenance; the historical
choice is retained in the backup. This is recorded as D27, not hidden as an exact
illustration-row match. Historical human-review samples are unchanged.

### BFGS computation

The installed SciPy 1.18.0 computes the BFGS inverse-Hessian update as
`(I - rho*s*y.T) @ H @ (I - rho*y*s.T) + rho*s*s.T`. Expanding the two rank-one
factors yields the same update using matrix-vector products and outer products.
`bfgs_update.py` reuses the installed SciPy function with only that expression
expanded; it changes no likelihood, line search, stopping tolerance, option or
initialization. No dependency files are edited. The snapshot manifest records the
SciPy version, original function SHA-256 and computation version.

`migration_staging/optimizer_equivalence.json` compares the two computations on the
full original-population PN0 mixed model for 20 identical iterations from the same
initialization: maximum parameter difference 4.3e-12 and objective difference
3.9e-10 (`rtol=1e-8`, `atol=1e-8`). Runtime was 43.6 versus 1.2 seconds.
A regression test also compares complete nonlinear BFGS fits. Both frozen-code and
corrected refits use the expanded computation, so their naming-only comparison
shares exactly the same numerical method. Prior dense attempts are retained as
interrupted diagnostics, not completed results. The computation version participates
in prepared-cache invalidation.

## Executed results (2026-10-23)

The staged migration was activated successfully. `migration_staging/verification.json`
records **44 verification groups** covering exact inverse-mapped prepared data,
codebook content, joins, coverage/eligibility, numerical RQ1–RQ3 results, posterior
draws and linkage arrays. Numerical tolerances are `rtol=1e-8`, `atol=1e-10`.
The separately recorded one-row RQ3 illustration correction enforces its SC forum
rule; both old and replacement annotation assignments remain original. All 1,910
backup files passed checksum verification; SQLite integrity checks pass and the
original source inputs and annotation databases are unchanged.

Rebuilt artifacts are active in `research_outputs/` and `data/exports/`: prepared
CSV/Parquet data, source/population audits, RQ1 prevalence/co-occurrence/rankings,
RQ2 primary/specification fits and diagnostics, RQ3 profiles/PCA/Ward diagnostics,
both 400-replicate light stability checks, tables, figures, database/convenience
exports, and the LaTeX/paper assets. The previous output trees are additionally
retained under `migration_staging/preactivation/`.

The following statistics describe the initial activation, before the
[PN3 convergence follow-up](PN3_CONVERGENCE.md). Current results restore PN3:
20 primary models converge, all four sensitivity fits converge, and clustering
uses 20 features. The k=2 groups remain 19/8, with silhouette 0.283.

At initial activation there were **19 converged primary models out of 21** for the expanded population.
AB fails the unchanged support screen. PN3 primary and quarter-time sensitivity
fits are nonconverged; they remain in diagnostics and are excluded from adjusted
profiles, predictions used in paper figures, and estimated anchor ranges. The
PN3 no-length sensitivity and both VD sensitivities converge (three of four
planned sensitivity fits). No iteration limit, threshold or model specification
was changed to force convergence. These are statistical estimation limitations,
not failed or missing Palestine annotations.

The leading RQ1 configuration is PN3+SYM+T3: **733 / 15,587 paragraphs (4.70%)**,
compared with 717 / 14,463 (4.96%) in the original population. The new adjusted
profile contains **27 countries and 19 features**. At k=2 the recomputed groups
have **19 and 8 countries**, silhouette 0.287; PC1/PC2 explain 44.4%/15.8%.
Model-draw and speech-resampling mean ARI are 0.741 and 0.521; weakest-group mean
Jaccard is 0.797 and 0.661. These are within-sample exploratory diagnostics;
previous memberships and narrative estimates are not carried forward.

**46 corrected historical review files** retain their original samples, content,
IDs and weights. The separate new pack contains 750 tasks per reviewer over
**715 distinct paragraphs**, with 100 probability and 50 diagnostic tasks per
codebook. Each complete-frame probability draw has probability 100/15,587 and
weight 155.87; diagnostic probabilities and weights remain undefined.

Verification commands completed:

```bash
.venv/bin/python -m un_speeches.research.migrate_identifiers verify
.venv/bin/python -m un_speeches.codebook.validate
.venv/bin/python -m unittest discover -s tests -v
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

All **34 tests pass**. The rollback integration test restores initial unsaved
work and prior outputs while retaining newer human reviews and leaving original
research inputs untouched. All five codebooks validate with **zero errors and 34
pre-existing warnings**, matching the frozen originals. The staged preview and
active `paper/main.pdf` build successfully as **seven-page** manuscripts. The
source-checked ten-example appendix, regenerated group membership and plot labels
were checked. Logs and execution metadata are retained in `migration_staging/`.
Human validation/adjudication remains pending; 19 eligible collected meetings remain
unannotated. The configured light stability checks are complete; the historical
full model-refit bootstrap remains unperformed. There are no unresolved execution
or dependency blockers, and no LLM calls or annotation reruns were made.

## Rollback

Preview the restore, then run it explicitly:

```bash
.venv/bin/python -m un_speeches.research.migrate_identifiers rollback --backup migration_backups/20261022T232628+0200 --dry-run
.venv/bin/python -m un_speeches.research.migrate_identifiers rollback --backup migration_backups/20261022T232628+0200
```

Rollback verifies backup checksums, preserves the migrated outputs in a timestamped
`migration_staging/rollback_preserved_*` directory, restores the original code,
configuration, paper and full output trees (including the initial uncommitted edits),
and moves migration-only source/test/docs additions out of the active project.
It does not restore or modify original source databases, PDFs or human-review inputs,
because the migration never changed them. Backups and historical audit additions remain.
