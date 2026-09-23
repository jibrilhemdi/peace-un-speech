# Data directory

`data/` holds research inputs and intermediate products. Its contents are ignored by Git
because the PDFs and annotation database are large, and some files contain unpublished
review material. Back up `raw/` and `db/` separately before cleaning or rebuilding outputs.

| Folder | Contents | Provenance / handling |
| --- | --- | --- |
| `raw/` | Downloaded UN meeting PDFs, full meeting manifest, corpus manifest, and selection links. | Source documents; keep original files and recorded symbols/hashes. The meeting manifest supplies the agenda flag used by the statistical scope rule. |
| `interim/` | Extracted speaker turns and segmented paragraph units. | Derived from PDFs and configuration. `un_speeches_extracted_extended.csv` supplies meeting metadata for the selected annotation snapshot. |
| `db/` | SQLite job store (`annotations.db`) and its WAL/SHM companions. | Primary annotation source; rebuilding it would require annotation calls. Never edit it for a statistical rerun. |
| `exports/` | CSV, Parquet, and Excel views of the database. | Derived convenience copies; the research analysis reads the database instead. |
| `pilots/` | Pilot annotation and adjudication sheets. | Preserve human edits; regenerated pilot files are not equivalent to completed adjudication. |
| `gold/` | Human-validation sample draws and design information. | Preserve completed human coding and inclusion probabilities. |
| `samples/` | Small manual spot-check extracts. | Working research material. |

The current statistical window is **2023-10-07 through 2025-12-31**, with inclusion
determined by the meeting agenda. The prepared research tables retain all source units
with an `in_scope` flag; analyses and new reviewer samples use only flagged units.
The meeting-level inclusion and exclusion reasons are exported to
`../research_outputs/audit/meeting_scope.csv`. See `../docs/DECISIONS.md` for the reason
behind the rule and its coverage tradeoff. The collected manifest currently has 112
date-and-agenda-eligible meetings; 93 occur in the existing annotated snapshot. The
other 19 are listed as `eligible_but_not_annotated` and cannot enter a statistical
rerun without new annotation.
