# Constructing Peace Meaning from UN Speech in the Context of the Palestine-Israel Conflict

This project examines how countries articulate and frame their understanding of peace in United Nations speeches addressing the Palestine-Israel conflict. Its statistical window is 7 October 2023 through 31 December 2025, using annotated UN verbatim meeting records and an explicit Palestine-agenda rule.

Our analytical framework draws on several interconnected concepts from Johan Galtung’s peace and conflict theory:

- **Triangle of Violence** – distinguishing between direct, structural, and cultural violence.
- **Negative and Positive Peace** – the absence of direct violence versus the presence of social justice and sustainable institutions.
- **Dissociative vs. Associative Relations** – categorizing relationships as abiosis (absence of relation), symbiosis (mutually beneficial), or antibiosis (destructive interdependence).
- **Symmetric/Asymmetric Relations** – derived from Galtung’s structural theory of imperialism, examining power imbalances between parties.
- **Transcend** – the method of transforming conflicts by bridging contradictory goals into a new, inclusive reality.

To systematically interpret diplomatic language at scale, we annotate the speeches using large language models accessed via OpenRouter.


---

# Current project layout and scope

The statistical corpus is selected from existing annotations by meeting date (7 October
2023 through 31 December 2025) and by a meeting agenda that names Palestine. The last
collected meeting satisfying both rules is 16 December 2025. The selection decision,
including meetings excluded under other agendas, is recorded in
[the decision log](docs/DECISIONS.md) and the generated `research_outputs/audit/meeting_scope.csv`.
The collected manifest has 112 eligible meetings; 93 have the selected annotations.
The current results describe those 93 meetings, and human validation is pending.

The active code is an installable package under `src/un_speeches/`; older analysis code
and its results are in `archive/legacy_analysis/`. `research_outputs/` holds current,
provisional analysis products. `paper/` contains a compilable two-column ACM anonymous
review draft.
The data layers and backup needs are described in [data/README.md](data/README.md).

```bash
pip install -r requirements.txt
pip install -e .
python -m un_speeches.collect.fetch_meeting_records --from-date 2023-10-07 --to-date 2025-12-31
python -m un_speeches.collect.select_corpus --rule agenda
python -m un_speeches.collect.extract_speech_turns --pdf-dir data/raw/corpus_pdfs
```

The archived Selenium collector and the older raw-count analysis are retained only for
provenance. The current research analysis reads the existing annotation database and
requires no new LLM calls. See [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) for the
statistical commands.

---

# Annotation pipeline

Deductive content analysis of the corpus: theory-derived codebooks are applied to
speech paragraphs by an LLM, producing paragraph-level variables for the current
descriptive, comparative, and profile analyses. The annotation notes below also
retain historical design context; the current statistical protocol is in
[RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md).

**Adding a codebook is a YAML file and nothing else.** No Python is edited to add codebook 3,
4 or 5. If a change seems to require one, the schema is missing a field — add it to
[`codebooks/SCHEMA.md`](codebooks/SCHEMA.md) first.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...      # environment only; never in a file, DB, or log
```

Everything else is configured in [`config/annotation.yaml`](config/annotation.yaml): model and fallbacks, rate and
quota limits, corpus filter, segmentation rules, regional-group map, export options.

## Adding codebook #3

This is the whole loop. No code edits at any step.

```bash
# 1. write it, following codebooks/SCHEMA.md — or start from a CSV:
python -m un_speeches.codebook.build_from_csv codebooks/legitimacy.csv
$EDITOR codebooks/legitimacy.yaml

# 2. lint — fails on duplicate codes, thin definitions, broken conditional_on;
#    warns about missing negative_examples until you write them
python -m un_speeches.codebook.validate

# 3. pilot: ~40 speeches stratified by regional group x speech length, under an hour of quota
python -m un_speeches.annotate.run_annotation --codebook legitimacy --pilot 40

# 4. adjudicate data/pilots/pilot_legitimacy_<timestamp>.csv — one row per paragraph, with empty
#    adj_agree / adj_correct_codes / adj_notes columns. Write negative_examples for anything
#    over-firing, then bump `version` in the YAML.

# 5. full run. Resumable — run it daily until it finishes.
python -m un_speeches.annotate.run_annotation --codebook legitimacy

# 6. check prevalence and drift
python -m un_speeches.annotate.report_status

# 7. export
python -m un_speeches.export.build_tables
```

A full pass is **refused** for a codebook with no completed pilot unless `--force` is passed.
A bad prompt costs three days of quota; a pilot costs under an hour.

## The scripts

`src/un_speeches/` is laid out by pipeline stage, in the order the stages run. Every script is a
module CLI, run as `python -m un_speeches.<stage>.<module>`, and resolves its paths from the
repo root, so the working directory never matters.

```
src/un_speeches/
├── collect/      the corpus:    ODS → meeting PDFs → selection → speaker turns
├── codebook/     the instrument: CSV → draft YAML → lint
├── annotate/     the labels:    speeches → units → model annotations → progress
├── export/       the tables:    job store → analysis-ready files
├── validation/   the check:     probability sample for human coding
├── research/     the statistics: scoped preparation → RQ1–RQ3 → reports
└── lib/          shared machinery, imported by the stages above
```

Collection runs in three steps — fetch everything, then narrow, then extract:

```bash
python -m un_speeches.collect.fetch_meeting_records --from-date 2023-10-07 --to-date 2025-12-31
python -m un_speeches.collect.fetch_meeting_records --series A/ES-10/PV --start 35 --end 60
python -m un_speeches.collect.select_corpus                     # rule A: agenda names Palestine
python -m un_speeches.collect.extract_speech_turns --pdf-dir data/raw/corpus_pdfs
```

| script | what it does |
| --- | --- |
| `collect/fetch_meeting_records.py` | Fetches verbatim records by symbol from `documents.un.org`. Enumerates meeting numbers rather than paging a search, so coverage is complete by construction. Writes `data/raw/meeting_manifest.csv` with the meeting date, agenda and term counts read from each PDF. `--series` selects S/PV, A/ES-10/PV, A/78/PV, …; `--reparse` re-derives the manifest from PDFs already on disk. |
| `collect/select_corpus.py` | Narrows the download to the corpus and symlinks the chosen PDFs into `data/raw/corpus_pdfs/`. Default rule keeps a meeting only if its agenda item names Palestine. Cheap and re-runnable — the corpus definition is an explicit decision, not a side effect of collection. |
| `collect/extract_speech_turns.py` | PDFs → one row per speaker turn. De-duplicates records by content hash and detects turns from the typography of UN verbatim records, not regexes. |
| `archive/legacy_collector/scrape_un_library.py` | **Superseded — kept for provenance only.** Drove the Digital Library search with Selenium. Its pagination cannot advance past ~500 records, `/search` sits behind an AWS WAF challenge, and its `creation_date` filter matches the catalogue date rather than the meeting date. It also no longer imports: `undetected_chromedriver` requires `distutils`, removed in Python 3.13+. |
| `codebook/build_from_csv.py` | Converts a codebook CSV into a **draft** YAML. Mechanical work only — splits numbered examples, extracts `S/PV.xxxx, Speaker (Country)` sources, normalises country names, detects an ordinal 0–4 scale. Marks everything needing judgement with `# TODO`. |
| `codebook/validate.py` | Lints every `codebooks/*.yaml` against SCHEMA.md. Exit 1 on error. |
| `annotate/segment_units.py` | Corpus → `data/interim/units.parquet` + the `units` table. Paragraph-level procedural marking, word-count floor, stable unit IDs, distribution report. |
| `annotate/run_annotation.py` | The resumable worker. One request per (speech, codebook). |
| `annotate/report_status.py` | Progress, quota, per-code prevalence, model/prompt drift warnings. |
| `export/build_tables.py` | DB → `annotations_long`, `unit_code_matrix`, `country_profiles`, manifest — as parquet, CSV and Excel. |
| `export/dump_database.py` | Dumps the DB's own tables to CSV/Excel, plus an `annotations_with_text` sheet pairing every label with the paragraph it labels. Read-only. |
| `validation/draw_gold_sample.py` | Probability sample with exact, recorded inclusion probabilities. |

Shared machinery is in `src/un_speeches/lib/` — none of it knows the name of any codebook, dimension
or code.

## How annotation works

**One request per (speech, codebook).** At ~8.8 paragraphs per speech, per-paragraph requests
would need ~19,000 calls per codebook. Batching brings it to ~2,245 calls per codebook;
completion time is governed by the configured local budget, pacing, model latency, and any
upstream provider limit.

The static codebook block leads every prompt and the speech trails it, so the prefix is
byte-identical across every request for a codebook version and prompt-prefix caching applies.
The prefix is ~73% of a typical request.

**A paragraph carries 0..n annotations**, each one value per dimension plus a verbatim
evidence span. Most paragraphs correctly get none. Dimensions are independent variables:
`violence_type` is the conception of peace, `attribution` is blame, and they are stored
separately so the analysis can tell "Russia and Algeria share a structural conception of
peace" from "Russia and Algeria both blame Israel."

### Resume and safety

- Every job row for a speech is written in **one transaction**, so a speech is either fully
  annotated or fully pending. Ctrl-C at any moment leaves a consistent database.
- Re-running only processes what is pending. Repeating completed work costs zero requests.
- Quota is committed **at attempt time**, before the outcome is known — the provider counts a
  request whether or not it succeeded, so a crash mid-request must not lose the count.
- Unit IDs are `sha256(speech_id + para_index + normalised_text)`. Re-running segmentation
  after a config change that would orphan annotated units prints a loud warning with counts
  and refuses to write without `--allow-id-change`.

### Error handling

Failed requests consume the daily quota, so nothing retries blindly.

| condition | behaviour |
| --- | --- |
| `429` with `Retry-After` | sleep that long, retry, max 3 attempts |
| `429` without, or repeated | exponential backoff with jitter |
| daily quota exhausted | stop cleanly, print the UTC resume time, **exit 0** so cron does not alarm |
| `5xx` / network | backoff, retry, then the next model in `provider.fallback_models` |
| `404` (model withdrawn) | swap to the next fallback immediately; fatal with an actionable message if none |
| `401` | fatal at once — retrying cannot help |
| unparseable JSON | one repair attempt with the error fed back, then `parse_error`, run continues |
| paragraph keys ≠ what was sent | rejected and re-requested; a misaligned batch would silently mislabel a whole speech |

`provider.fallback_models` ships **empty**. The default model is new and single-provider, so
populate it from the live catalogue before a long run:

```bash
curl -s https://openrouter.ai/api/v1/models \
  | python -c "import json,sys;[print(m['id']) for m in json.load(sys.stdin)['data']]" | grep free
```

### Measurement validity

`status.py` prints per-code prevalence among completed units and warns when:

- a code fires on **>50%** of paragraphs (it carries almost no information and will flatten
  the MCA — fix the prompt before spending more quota)
- **models differ** across completed units of one codebook
- the **prompt hash differs** across completed units — the codebook changed mid-corpus
  without a version bump, so early and late units were coded against different instruments
- **>15% of evidence spans** cannot be matched back to their paragraph (the model is
  paraphrasing, making the evidence column useless for audit)
- **<20% of units** received no annotation at all

Bumping `version` in a codebook invalidates cached annotations for that codebook only. Old
versions are retained, never deleted, so v1 and v2 sit side by side and you can measure how
much a prompt revision moved the labels.

## The five codebooks are one framework, not five instruments

All five are Galtung:

| codebook | construct | source |
| --- | --- | --- |
| `violence` | the violence triangle — direct / structural / cultural | Galtung 1969; 1990 |
| `positive_negative` | negative and positive peace | Galtung 1964; 1969 |
| `relations` | dissociative vs associative relations — abiosis / antibiosis / symbiosis | Galtung |
| `imperialism` | symmetric vs asymmetric structure | Galtung 1971, *A Structural Theory of Imperialism* |
| `transcend` | either/or vs both/and | Galtung 2004, *Transcend and Transform* |


That shared origin has a consequence the analysis has to handle. `violence` and
`positive_negative` are not independent measurements: in Galtung 1969 negative peace *is* the
absence of direct violence and positive peace *is* the absence of structural violence. A
paragraph coded VS and a paragraph scoring high on `positive_negative` are the same observation
seen from two sides. `relations` and `transcend` are similarly close — symbiosis and both/and
are near-synonyms.

Measured on the two codebooks that have been run, the coupling behaves very differently at the
two levels:

- **Paragraph level: effectively independent.** Cramér's V between "violence coded" and
  "relations coded" is **0.046**. They fire on different paragraphs — only 974 of 19,351
  paragraphs carry both.
- **Country level: strongly correlated.** Spearman ρ across country profiles reaches **+0.74**
  (ANTI × VC), **+0.67** (ANTI × VD) and **−0.54** (SYM × VD).

The MCA and the latent class model run on country profiles, so it is the second set of numbers
that governs. Two practical consequences:

1. `ANTI` is the only usable variable in `relations` (see that codebook's results), and it
   correlates 0.6–0.75 with three violence variables. Adding `relations` to an MCA alongside
   `violence` may contribute little independent information.
2. Within `violence`, `VS` and `ATR-ISR` correlate **+0.91** at country level. The two-dimension
   split remains right at paragraph level — VS is attributed to Israel 79% of the time against
   52% for VD — but as country-level shares the two columns are close to collinear. Check this
   before putting both into the same model.

Expect `positive_negative` to be *more* coupled to `violence` than `relations` is, since they
are the same distinction rather than adjacent ones. Worth checking at the pilot stage rather
than after a full pass: run the pilot, export, and correlate the country profiles.

## Changing the model, and re-running on a new one

The provider model is chosen in `config/annotation.yaml` and **recorded on every job row at request
time** — it is never inferred from the response. It is carried into all three exports:

| table | column | meaning |
| --- | --- | --- |
| `annotations_long` | `model`, `model_tag`, `run_id` | per annotation |
| `unit_code_matrix` | `<codebook>__model_tag` | per unit, per codebook |
| `country_profiles` | `<codebook>__models` | `"a+b"` when a codebook spans a model change |

`model_tag` is the short form (`poolside/laguna-s-2.1:free` → `laguna-s-2.1`) — the vendor
prefix and `:free` suffix are dropped because neither identifies the instrument. Use it as the
dummy in a regression; the full id stays in `model`.

**History.** `inclusionai/ling-3.0-flash:free` was withdrawn on 2026-08-07. `violence`,
`relations`, `imperialism` and `positive_negative` are complete on it, as are the first 1,296
speeches of `transcend`. Everything from that date runs on `poolside/laguna-s-2.1:free`.

### Re-running a codebook on a different model

Use a **new `run_id`**, not a new `version`. Version means the codebook changed; run_id means
the same instrument was applied again. Nothing existing is touched:

```bash
# re-run violence on the current model, alongside the original
python -m un_speeches.annotate.run_annotation --codebook violence --run-id laguna --force

# export both side by side
python -m un_speeches.export.build_tables --codebook violence --run-id main --run-id laguna
```

With more than one `--run-id` the tables change shape deliberately:

- `unit_code_matrix` gains a `run_id` column and is keyed on `(unit_id, run_id)`
- `country_profiles` becomes one row per `(country, run_id)` — it never averages two models
  into one profile row
- `export_manifest.json` keys each entry `<codebook>@<run_id>`

With a single `--run-id` (the default) the shapes are exactly as before, plus the model columns.

Because a re-run covers the same `unit_id`s, the two runs are **paired on identical text**,
which is what makes them a usable estimate of how much the model change moves labels.

## The data directory

`data/` is layered by provenance. Each layer is reproducible from the one above it, except
`raw/`, which is not reproducible at all without re-scraping.

```
data/
├── raw/                     collected from the UN Digital Library. Not reproducible.
│   ├── pdfs/                501 files, 101 distinct verbatim meeting records
│   ├── un_speeches_palestine.csv       the catalogue index the collector built
│   └── collection_checkpoint.json      collector resume state
├── interim/                 reproducible from raw/
│   ├── un_speeches_extracted.csv       the corpus: one row per speaker turn
│   └── units.parquet        one row per annotation unit, with para_is_procedural, bloc, is_party
├── db/
│   └── annotations.db       the job store: units, jobs, annotations, quota, pilots
├── exports/                 reproducible from the DB. Never edit by hand.
│   ├── parquet/             canonical analysis tables
│   ├── csv/                 the same tables, openable anywhere
│   ├── excel/               the same tables, one workbook
│   ├── db/                  verbatim dumps of the DB's own tables (db_to_csv.py)
│   └── export_manifest.json codebook versions, models, prompt hashes, row counts
├── pilots/                  pilot adjudication sheets, one CSV per codebook run
│   └── archive/             superseded pilots
├── gold/                    gold sample draws for human coding
└── samples/                 small hand-made spot-check extracts
```

Every one of these locations is set in the `paths:` block of `config/annotation.yaml` and read through
`Config.path_for()`. Move a directory there and the whole pipeline follows; no script hardcodes
a path under `data/`.

## Outputs

`un_speeches.export.build_tables` writes three analysis tables. Each is written once per format, and the
formats are all the same rows:

| table | contents |
| --- | --- |
| `annotations_long` | `unit_id, codebook_id, annotation_index, dimension, value, evidence, confidence` + provenance |
| `unit_code_matrix` | one row per unit, one `Int8` column per `<codebook>__<dimension>__<code>` |
| `country_profiles` | per country, the share of its annotated paragraphs carrying each code |

```bash
python -m un_speeches.export.build_tables                      # parquet + csv + excel
python -m un_speeches.export.build_tables --formats parquet    # canonical only
```

- `data/exports/parquet/` — **canonical.** Analyse from these. They preserve `Int8`/`Float32`
  and, crucially, the difference between `NA` and `0`.
- `data/exports/csv/` — same rows, UTF-8 with a BOM so Excel does not mangle accented names.
- `data/exports/excel/un_speeches_annotations.xlsx` — one workbook, one sheet per table, frozen
  header and autofilter on each.

Treat CSV and Excel as **views, not sources**: reading a CSV back into pandas loses the
nullable-integer types, and an un-annotated unit comes back as a blank rather than as `NA`.

In `unit_code_matrix` an un-annotated unit is `NA`, not `0` — a missing label and an absent code
are different things and must not be conflated. `country_profiles` divides by each country's
**annotated** paragraphs, so a half-finished run reports an honest rate rather than a diluted
one.

### Reading the job store without SQL

```bash
python -m un_speeches.export.dump_database
```

Dumps every table in `annotations.db` to `data/exports/db/csv/` and to one workbook at
`data/exports/db/excel/annotations_db.xlsx`, plus a sheet that is not in the database at all:

| sheet | contents |
| --- | --- |
| `annotations_with_text` | every annotation next to **the paragraph it labels**, with country, meeting date, the model's evidence quote and its confidence |

That is the sheet to open when the question is "what did the model actually say about this
paragraph". `jobs.response_json` (~95 MB of raw model output) and `jobs.error` are dropped
unless you pass `--with-responses`. The dump opens the database read-only, so it can never be
the thing that damages a store that took days of quota to fill.

## Gold sample

```bash
python -m un_speeches.validation.draw_gold_sample --codebook violence -n 300 \
    --stratify-on violence_type --oversample VC=4
```

Writes `data/gold/gold_sample.parquet` with a `pi` column (exact inclusion probability) and
`weight` = 1/pi, plus `gold_sample.design.json` recording the full design. Stratifying on the
model's own predicted labels is legitimate **because** pi is recorded at draw time; a later
measurement-error correction needs those probabilities known by design, not reconstructed. The
human-coding UI is out of scope.
