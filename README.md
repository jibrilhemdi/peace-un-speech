# UN Speeches Collector

Collect UN speeches related to the Palestine question from the UN Digital Library.

## Constraints & Legal Notes

The UN Digital Library Terms of Use state:

> "Web scraping and/or automated downloads of content from the United Nations Digital Library is not permitted without approval by the Library."

**Recommended approaches (in order of preference):**

1. **Manual export** — Create an account at https://digitallibrary.un.org and use the built-in export feature (limited to 100 records per export).
2. **Request bulk access** — Contact the UN Dag Hammarskjöld Library (`library@un.org`) for bulk data export permissions.
3. **Browser automation** — Use the provided undetected Selenium script for local, personal research only. You are responsible for ensuring compliance with the UN Digital Library Terms of Use.

## Search Configuration

- **Subject**: "Palestine question"
- **Date range**: October 7, 2023 to December 31, 2025
- **Collection**: Speeches

## Usage

```bash
pip install -r requirements.txt
python scripts/collect/scrape_un_library.py
```

The script will:
1. Open the search URL with the configured filters
2. Extract speech titles and URLs from each result page
3. Navigate through all pages using pagination
4. Save unique results to `data/raw/un_speeches_palestine.csv`, with the PDFs in
   `data/raw/pdfs/` and resume state in `data/raw/collection_checkpoint.json`

## Output

The CSV file contains:
- `title`: The speaker and country
- `url`: Link to the full record on the UN Digital Library

## Extracting the speeches

```bash
python scripts/collect/extract_speech_turns.py
```

Parses every PDF in `data/raw/pdfs/` into one row per speaker turn and writes
`data/interim/un_speeches_extracted.csv`.

Notes on how it works:

- The downloaded PDFs are **full verbatim meeting records**, not single speeches, and the
  collector saved the same record once per speaker — the 500 files are only 101 distinct
  meetings. The script de-duplicates by content hash and parses each meeting once, so it
  recovers every speech in those meetings, not just the ones that had a catalogue record.
- Turns are detected from **typography, not regexes**: UN records always set the speaker's
  name in bold at the start of their first paragraph, e.g.
  `**Mr. Selim** (Egypt) (*spoke in Arabic*): ...`. Running heads, mastheads, vote tallies
  and italic stage directions are filtered out by font size, style and position.
- 99.7% of body text lands in a speech; the remainder is agenda headings, roll-call vote
  lists and procedural notes.

### Columns

| column | meaning |
| --- | --- |
| `ambassador_name` | Speaker, e.g. `Mr. Selim`. For `The President` the name is resolved from the cover-page roster (`Mr. X/Mr. Y` when the roster lists alternates). |
| `country` | Country/entity, normalised (`United States of America` → `United States`). Empty for UN briefers. |
| `speech` | Full text of the intervention, paragraphs separated by blank lines. |
| `is_procedural` | `True` when the turn is *entirely* chair housekeeping — see below. |
| `substantive_word_count` | Words left after procedural paragraphs are removed. |
| `role` | `representative`, `president` or `official/briefer`. |
| `language` | Language spoken, from `(spoke in ...)`; defaults to English. |
| `speaker_header` | Raw header line as printed in the record. |
| `doc_symbol`, `meeting_date`, `meeting_number` | Meeting identifiers. |
| `turn_index`, `word_count` | Position within the meeting, and speech length. |
| `source_pdf`, `record_ids` | Provenance: the parsed file, and every catalogue record that pointed at it. |

### Procedural turns

Much of what the chair says is housekeeping — "I now give the floor to...", invitations
under rules 37/39, vote mechanics, speaking-time reminders. Every row is kept, but each
**paragraph** is classified, and `is_procedural` is `True` only when *nothing* substantive
remains. That distinction matters because the President regularly opens with housekeeping
and then delivers a national statement ("I shall now make a statement in my capacity as the
representative of..."); those turns stay `False` and keep their full text.

962 of 3,210 rows are procedural — 29% of rows but only 1.3% of the words.

```python
df = df[~df.is_procedural]                      # drop pure housekeeping
df = df[df.role == "representative"]            # ...and keep only Member State statements
df = df[df.substantive_word_count >= 50]        # ...or filter by length instead
```

`is_procedural` is deliberately conservative: it fires only on recognised chair formulas,
so a residue of ~130 short President turns with one-off phrasing is left unflagged. Use
`substantive_word_count` if you want a stricter cut.

## Further Reading

- [UN Digital Library Help](https://digitallibrary.un.org/pages?ln=en)
- [Ask DAG: What datasets are available?](https://ask.un.org/faq/432839)

---

# Annotation pipeline

Deductive content analysis of the corpus: theoretically-derived codebooks are applied to
speech paragraphs by an LLM, producing a document × code matrix for MCA, latent class
analysis and regression.

**Adding a codebook is a YAML file and nothing else.** No Python is edited to add codebook 3,
4 or 5. If a change seems to require one, the schema is missing a field — add it to
[`codebooks/SCHEMA.md`](codebooks/SCHEMA.md) first.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...      # environment only; never in a file, DB, or log
```

Everything else is configured in [`config.yaml`](config.yaml): model and fallbacks, rate and
quota limits, corpus filter, segmentation rules, regional-group map, export options.

## Adding codebook #3

This is the whole loop. No code edits at any step.

```bash
# 1. write it, following codebooks/SCHEMA.md — or start from a CSV:
python scripts/codebook/build_from_csv.py codebooks/legitimacy.csv
$EDITOR codebooks/legitimacy.yaml

# 2. lint — fails on duplicate codes, thin definitions, broken conditional_on;
#    warns about missing negative_examples until you write them
python scripts/codebook/validate.py

# 3. pilot: ~40 speeches stratified by regional group x speech length, under an hour of quota
python scripts/annotate/run_annotation.py --codebook legitimacy --pilot 40

# 4. adjudicate data/pilots/pilot_legitimacy_<timestamp>.csv — one row per paragraph, with empty
#    adj_agree / adj_correct_codes / adj_notes columns. Write negative_examples for anything
#    over-firing, then bump `version` in the YAML.

# 5. full run. Resumable — run it daily until it finishes.
python scripts/annotate/run_annotation.py --codebook legitimacy

# 6. check prevalence and drift
python scripts/annotate/report_status.py

# 7. export
python scripts/export/build_tables.py
```

A full pass is **refused** for a codebook with no completed pilot unless `--force` is passed.
A bad prompt costs three days of quota; a pilot costs under an hour.

## The scripts

`scripts/` is laid out by pipeline stage, in the order the stages run. Every script is a
standalone CLI, run as `python scripts/<stage>/<script>.py`, and resolves its paths from the
repo root, so the working directory never matters.

```
scripts/
├── collect/      the corpus:    ODS → meeting PDFs → selection → speaker turns
├── codebook/     the instrument: CSV → draft YAML → lint
├── annotate/     the labels:    speeches → units → model annotations → progress
├── export/       the tables:    job store → analysis-ready files
├── validation/   the check:     probability sample for human coding
└── lib/          shared machinery, imported by the stages above
```

Collection runs in three steps — fetch everything, then narrow, then extract:

```bash
python scripts/collect/fetch_meeting_records.py --from-date 2023-10-07 --to-date 2025-12-31
python scripts/collect/fetch_meeting_records.py --series A/ES-10/PV --start 35 --end 60
python scripts/collect/select_corpus.py                     # rule A: agenda names Palestine
python scripts/collect/extract_speech_turns.py --pdf-dir data/raw/corpus_pdfs
```

| script | what it does |
| --- | --- |
| `collect/fetch_meeting_records.py` | Fetches verbatim records by symbol from `documents.un.org`. Enumerates meeting numbers rather than paging a search, so coverage is complete by construction. Writes `data/raw/meeting_manifest.csv` with the meeting date, agenda and term counts read from each PDF. `--series` selects S/PV, A/ES-10/PV, A/78/PV, …; `--reparse` re-derives the manifest from PDFs already on disk. |
| `collect/select_corpus.py` | Narrows the download to the corpus and symlinks the chosen PDFs into `data/raw/corpus_pdfs/`. Default rule keeps a meeting only if its agenda item names Palestine. Cheap and re-runnable — the corpus definition is an explicit decision, not a side effect of collection. |
| `collect/extract_speech_turns.py` | PDFs → one row per speaker turn. De-duplicates records by content hash and detects turns from the typography of UN verbatim records, not regexes. |
| `collect/scrape_un_library.py` | **Superseded — kept for provenance only.** Drove the Digital Library search with Selenium. Its pagination cannot advance past ~500 records, `/search` sits behind an AWS WAF challenge, and its `creation_date` filter matches the catalogue date rather than the meeting date. It also no longer imports: `undetected_chromedriver` requires `distutils`, removed in Python 3.13+. |
| `codebook/build_from_csv.py` | Converts a codebook CSV into a **draft** YAML. Mechanical work only — splits numbered examples, extracts `S/PV.xxxx, Speaker (Country)` sources, normalises country names, detects an ordinal 0–4 scale. Marks everything needing judgement with `# TODO`. |
| `codebook/validate.py` | Lints every `codebooks/*.yaml` against SCHEMA.md. Exit 1 on error. |
| `annotate/segment_units.py` | Corpus → `data/interim/units.parquet` + the `units` table. Paragraph-level procedural marking, word-count floor, stable unit IDs, distribution report. |
| `annotate/run_annotation.py` | The resumable worker. One request per (speech, codebook). |
| `annotate/report_status.py` | Progress, quota, per-code prevalence, model/prompt drift warnings. |
| `export/build_tables.py` | DB → `annotations_long`, `unit_code_matrix`, `country_profiles`, manifest — as parquet, CSV and Excel. |
| `export/dump_database.py` | Dumps the DB's own tables to CSV/Excel, plus an `annotations_with_text` sheet pairing every label with the paragraph it labels. Read-only. |
| `validation/draw_gold_sample.py` | Probability sample with exact, recorded inclusion probabilities. |

Shared machinery is in `scripts/lib/` — none of it knows the name of any codebook, dimension
or code.

## How annotation works

**One request per (speech, codebook).** At ~8.8 paragraphs per speech, per-paragraph requests
would need ~19,000 calls per codebook — 20 days at 950/day. Batching brings it to ~2,245
calls, about 3 days per codebook.

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

**Citations are from memory and have not been checked against the sources — verify before
citing them in the thesis.** The construct descriptions are what the model reads; the years are
for you.

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

The provider model is chosen in `config.yaml` and **recorded on every job row at request
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
python scripts/annotate/run_annotation.py --codebook violence --run-id laguna --force

# export both side by side
python scripts/export/build_tables.py --codebook violence --run-id main --run-id laguna
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

Every one of these locations is set in the `paths:` block of `config.yaml` and read through
`Config.path_for()`. Move a directory there and the whole pipeline follows; no script hardcodes
a path under `data/`.

## Outputs

`scripts/export/build_tables.py` writes three analysis tables. Each is written once per format, and the
formats are all the same rows:

| table | contents |
| --- | --- |
| `annotations_long` | `unit_id, codebook_id, annotation_index, dimension, value, evidence, confidence` + provenance |
| `unit_code_matrix` | one row per unit, one `Int8` column per `<codebook>__<dimension>__<code>` |
| `country_profiles` | per country, the share of its annotated paragraphs carrying each code |

```bash
python scripts/export/build_tables.py                      # parquet + csv + excel
python scripts/export/build_tables.py --formats parquet    # canonical only
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
python scripts/export/dump_database.py
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
python scripts/validation/draw_gold_sample.py --codebook violence -n 300 \
    --stratify-on violence_type --oversample VC=4
```

Writes `data/gold/gold_sample.parquet` with a `pi` column (exact inclusion probability) and
`weight` = 1/pi, plus `gold_sample.design.json` recording the full design. Stratifying on the
model's own predicted labels is legitimate **because** pi is recorded at draw time; a later
measurement-error correction needs those probabilities known by design, not reconstructed. The
human-coding UI is out of scope.
# peace-un-speech
