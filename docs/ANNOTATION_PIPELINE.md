# Collection and annotation guide

Use this guide when collecting new records or running new LLM annotation jobs.
The current paper uses saved annotations; its statistical workflow is in
[RESEARCH_PIPELINE.md](../RESEARCH_PIPELINE.md). Run commands from the repository
root after the [project setup](../README.md#set-up).

## 1. Collect and extract records

Examples for Security Council and emergency General Assembly records:

```bash
.venv/bin/python -m un_speeches.collect.fetch_meeting_records --from-date 2023-10-07 --to-date 2025-12-31
.venv/bin/python -m un_speeches.collect.fetch_meeting_records --series A/ES-10/PV --start 35 --end 60
.venv/bin/python -m un_speeches.collect.select_corpus --rule agenda
.venv/bin/python -m un_speeches.collect.extract_speech_turns \
  --pdf-dir data/raw/corpus_pdfs \
  --out data/interim/un_speeches_extracted_extended.csv
```

The collector retrieves PDFs by UN document symbol and records their meeting
dates, agendas, and hashes. Selection keeps records whose agendas name Palestine;
extraction separates speaker turns and removes duplicate PDFs. The explicit
output path above matches the active annotation configuration.

## 2. Prepare paragraphs and codebooks

[annotation.yaml](../config/annotation.yaml) controls paths, paragraph rules,
model settings, and request limits. [SCHEMA.md](../codebooks/SCHEMA.md) defines
the YAML codebook format.

```bash
.venv/bin/python -m un_speeches.annotate.segment_units --dry-run
.venv/bin/python -m un_speeches.annotate.segment_units
.venv/bin/python -m un_speeches.codebook.validate
```

For a new codebook, write its YAML or start with
`python -m un_speeches.codebook.build_from_csv path/to/codebook.csv`.
Review definitions and examples before annotation. Change the codebook version
when the coding rules change; use a new run ID when applying the same rules again.

## 3. Pilot, review, then annotate

New annotation jobs require `OPENROUTER_API_KEY` in the environment. Keep the key
out of source files and logs. The example below uses a separate run ID:

```bash
.venv/bin/python -m un_speeches.annotate.run_annotation --codebook violence --run-id new-study --dry-run
.venv/bin/python -m un_speeches.annotate.run_annotation --codebook violence --run-id new-study --pilot 40
```

Review the generated pilot sheet in `data/pilots/`, revise the codebook if needed,
and validate it again. The worker normally requires a completed pilot before a
full run:

```bash
.venv/bin/python -m un_speeches.annotate.run_annotation --codebook violence --run-id new-study
.venv/bin/python -m un_speeches.annotate.report_status
```

Each request annotates one speech with one codebook. Responses contain paragraph
labels and evidence spans; empty labels are valid. The database records job
status, model, run ID, and prompt hashes. Repeating the command resumes pending
jobs, while completed jobs are retained. Request failures count toward quota;
provider and retry settings are in the configuration.

Segmentation protects existing paragraph IDs. Back up source files, the database,
and human-edited review sheets before changing inputs; see [data handling](../data/README.md).

## 4. Export or inspect annotations

```bash
.venv/bin/python -m un_speeches.export.build_tables --codebook violence --run-id new-study
.venv/bin/python -m un_speeches.export.dump_database
```

The first command exports annotation rows, paragraph-code matrices, and country
shares as Parquet, CSV, and Excel. Multiple `--run-id` arguments keep runs separate.
The second exports database tables and an `annotations_with_text` view for reading
labels alongside their source paragraphs.

A missing evaluation is **NA**, whereas a completed evaluation without a code
is **0**. Preserve that distinction when analyzing exports. The current research
pipeline reads the annotation database directly and selects the runs recorded
in [research.yaml](../config/research.yaml).

Use any module's `--help` for its full options. Historical collector and analysis
implementations are documented in [archive/README.md](../archive/README.md).
