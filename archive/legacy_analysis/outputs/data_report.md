# Data report — Phase 0 (discovery and validation)

Source of truth: `data/db/annotations.db`

## 1. Database candidates

- `data/db/annotations.db` — score 3/3 (paragraphs=19351, speeches=2245, meetings=101)

## 2. Schema mapping

| Canonical field | Table | Column | Note |
|---|---|---|---|
| paragraph id | `units` | `unit_id` |  |
| paragraph text | `units` | `text` |  |
| speech id | `units` | `speech_id` |  |
| meeting id | `units` | `speech_id` | derived: substring before '#' |
| meeting date | `units` | `meeting_date` | text '%d %B %Y' -> datetime |
| speaker name | `units` | `ambassador_name` |  |
| speaker country | `units` | `country` |  |
| speaker role | `units` | `role` | representative | president | official/briefer |
| violence_type | `annotations` | `value where dimension='violence_type'` |  |
| attribution | `annotations` | `value where dimension='attribution'` |  |
| positive_negative | `annotations` | `value where dimension='positive_negative'` |  |
| transcend | `annotations` | `value where dimension='transcend'` |  |
| imperialism | `annotations` | `value where dimension='imperialism'` |  |
| relation_type | `annotations` | `value where dimension='relation_type'` |  |
| confidence | `annotations` | `confidence` |  |
| evidence quote | `annotations` | `evidence` |  |
| evidence_verified | `annotations` | `evidence_verified` |  |
| run_id | `annotations / jobs` | `run_id` |  |
| annotation model | `jobs` | `model` | joined on (unit_id, codebook_id, run_id) |
| procedural flag | `units` | `para_is_procedural` |  |
| party flag | `units` | `is_party` | 1 = Israel / State of Palestine |
| regional bloc | `units` | `bloc` | WEOG/ASIA_PACIFIC/AFRICAN/GRULAC/EASTERN_EUROPEAN/OBSERVER |

The database stores annotations in long form: one row per
`(unit_id, codebook_id, annotation_index, dimension)`. A paragraph carries 0..n
annotations, and one annotation fixes one value per elicited dimension. The six
canonical dimensions live in five codebooks: `violence` supplies both
`violence_type` and `attribution` (the latter `conditional_on` the former);
`positive_negative`, `transcend`, `imperialism` and `relations` supply one each.

**Absence is not stored.** A dimension with no annotation for a paragraph is
expressed by the absence of a row, so every dimension gains a synthetic
`NONE_ABSENT` category built downstream. `attribution` also has an
*explicit* `NONE` value (violence coded but no attributable actor) which is a
substantive category and is kept distinct from absence.

## 3. Filters applied

1. `run_id = 'main'` — excludes 'pilot-main' (the pilot run, 868 annotations across 5 codebooks).
2. `para_is_procedural = 0` — drops 486 procedural paragraphs.
3. All `jobs` rows for the main run have `status='ok'`; no paragraph is missing a
   job for any codebook, so absence of an annotation is genuine absence rather
   than a failed call.

## 4. Sanity checks

| Check | Expected | Observed | Delta | % | Pass |
|---|---|---|---|---|---|
| non-procedural paragraphs | 19,351 | 19,351 | +0 | +0.000 | PASS |
| speeches | 2,245 | 2,245 | +0 | +0.000 | PASS |
| meetings | 101 | 101 | +0 | +0.000 | PASS |
| violence_type: VD | 8,371 | 8,371 | +0 | +0.000 | PASS |
| violence_type: VS | 5,729 | 5,729 | +0 | +0.000 | PASS |
| violence_type: VC | 675 | 675 | +0 | +0.000 | PASS |
| attribution: ATR-ISR | 9,239 | 9,239 | +0 | +0.000 | PASS |
| attribution: ATR-PAL | 2,773 | 2,773 | +0 | +0.000 | PASS |
| attribution: NONE | 2,763 | 2,763 | +0 | +0.000 | PASS |
| positive_negative: PN1 | 4,528 | 4,528 | +0 | +0.000 | PASS |
| positive_negative: PN3 | 2,560 | 2,560 | +0 | +0.000 | PASS |
| positive_negative: PN0 | 2,003 | 2,003 | +0 | +0.000 | PASS |
| positive_negative: PN2 | 1,849 | 1,849 | +0 | +0.000 | PASS |
| positive_negative: PN4 | 225 | 225 | +0 | +0.000 | PASS |
| transcend: T0 | 1,395 | 1,395 | +0 | +0.000 | PASS |
| transcend: T3 | 1,227 | 1,227 | +0 | +0.000 | PASS |
| transcend: T1 | 919 | 919 | +0 | +0.000 | PASS |
| transcend: T2 | 753 | 753 | +0 | +0.000 | PASS |
| transcend: T4 | 481 | 481 | +0 | +0.000 | PASS |
| imperialism: I1 | 2,262 | 2,262 | +0 | +0.000 | PASS |
| imperialism: I2 | 798 | 798 | +0 | +0.000 | PASS |
| imperialism: I3 | 548 | 548 | +0 | +0.000 | PASS |
| imperialism: I0 | 277 | 277 | +0 | +0.000 | PASS |
| imperialism: I4 | 214 | 214 | +0 | +0.000 | PASS |
| relation_type: SYM | 1,883 | 1,883 | +0 | +0.000 | PASS |
| relation_type: ANTI | 513 | 513 | +0 | +0.000 | PASS |
| relation_type: AB | 58 | 58 | +0 | +0.000 | PASS |

All 27 checks reproduce the expected values **exactly** (delta 0). The expected dimension counts are *annotation-level* counts, not counts of distinct paragraphs — a paragraph carrying VD twice contributes twice. That reading is what reproduces the brief's numbers.

## 5. Transcend model switch

| Model | Paragraphs | First meeting | Last meeting | Meetings |
|---|---|---|---|---|
| `inclusionai/ling-3.0-flash:free` | 10,816 | 2023-08-21 | 2025-09-29 | 58 |
| `poolside/laguna-s-2.1:free` | 8,535 | 2023-10-24 | 2025-09-11 | 49 |

The split **is temporal**. Assigning each meeting its modal model and grouping contiguous runs gives:

| Model | Start | End | Meetings | Paragraphs (by model) | Purity |
|---|---|---|---|---|---|
| `inclusionai/ling-3.0-flash:free` | 2023-08-21 | 2024-09-04 | 54 | 10,167 | 0.994 |
| `poolside/laguna-s-2.1:free` | 2024-09-16 | 2025-09-11 | 44 | 8,473 | 0.992 |
| `inclusionai/ling-3.0-flash:free` | 2025-09-18 | 2025-09-29 | 3 | 577 | 1.000 |

**Largest single-model span:** `inclusionai/ling-3.0-flash:free`, 2023-08-21 .. 2024-09-04 — 54 meetings, 10,167 paragraphs, 99.4% pure.

Policy applied:
- Any transcend-based comparison **across time** (monthly trends, phase re-clustering diagnostics, event studies) is restricted to this span.
- Pooled cross-sectional use (fingerprints, CA, clustering) combines both models; sensitivity variant (c) re-runs the typology on the span alone.
- Regressions on a transcend outcome carry a model indicator where both models are present; where the span restriction already forces a single model the indicator is collinear and omitted.

## 6. Speaker classification

| Class | Speeches | Paragraphs | Distinct countries |
|---|---|---|---|
| `member_state` | 1,858 | 14,771 | 101 |
| `briefer` | 163 | 2,239 | 0 |
| `party` | 160 | 2,111 | 2 |
| `group_rep` | 12 | 150 | 6 |
| `presiding` | 50 | 72 | 18 |
| `other_invitee` | 2 | 8 | 1 |

Rules, in order of precedence:

1. `role='official/briefer'` -> **briefer** (UN Secretariat officials and invited briefers; `country` is empty for all of them). Includes the EU's own representatives (Lambrinidis, Borrell Fontelles, Kallas, Skoog), so EU group statements are already outside the member-state population.
2. `is_party=1` -> **party** (Israel, State of Palestine).
3. Speech opens by *delivering* a statement on behalf of a regional group -> **group_rep**.
4. `bloc='OBSERVER'` and not a party -> **other_invitee** (Holy See; a non-member observer state, so outside the member-state population).
5. `role='president'` -> **member_state** if the speech carries an explicit national-capacity marker or has >= 4 non-procedural paragraphs, else **presiding**.
6. `role='representative'` with a non-empty country -> **member_state**.

### 6.1 Group-representative speeches detected

| Speech | Country | Group delivered on behalf of | Classified as |
|---|---|---|---|
| `S/PV.9451 (Resumption 1)#84` | (delegation) | European Union | `briefer` |
| `S/PV.9534#82` | (delegation) | European Union | `briefer` |
| `S/PV.9881#24` | (delegation) | League of Arab States | `briefer` |
| `S/PV.9534#54` | Bahrain | Group of Arab States | `group_rep` |
| `S/PV.9498#23` | Egypt | Group of Arab States | `group_rep` |
| `S/PV.9830#29` | Egypt | Group of Arab States | `group_rep` |
| `S/PV.9846#33` | Egypt | Group of Arab States | `group_rep` |
| `S/PV.9439#25` | Jordan | Group of Arab States | `group_rep` |
| `S/PV.9443#25` | Jordan | Group of Arab States | `group_rep` |
| `S/PV.9462#31` | Jordan | Arab Group | `group_rep` |
| `S/PV.9730#27` | Syrian Arab Republic | Group of Arab States | `group_rep` |
| `S/PV.9734#34` | Syrian Arab Republic | Group of Arab States | `group_rep` |
| `S/PV.9560#35` | Tunisia | Group of Arab States | `group_rep` |
| `S/PV.9914#25` | United Arab Emirates | Group of Arab States | `group_rep` |
| `S/PV.9923#27` | United Arab Emirates | Group of Arab States | `group_rep` |

15 group deliveries detected. 12 are reclassified out of the member-state population as `group_rep`; the remaining 3 are delivered by the organisations' own representatives (the EU Special Representative, the Secretary-General of the League of Arab States), whom rule 1 has already classified as briefers. Detection requires a first-person *delivery* verb before `on behalf of <group>` and no alignment verb (`aligns itself with`, `associates itself with`, `proposed by`) in the same lead-in — the corpus contains far more speeches that *align with* a group statement than deliver one, and those remain national-capacity.

### 6.2 Ambiguous cases logged

- **Presidency speeches.** A country holding the Council presidency never has a separate `role='representative'` speech in the same meeting (0 overlaps), so its national statement is filed under `role='president'`. Dropping the role wholesale would systematically silence each member during its presidency month. The >= 4-paragraph rule is empirically grounded: the share of paragraphs carrying at least one annotation is 2.8% for 1-paragraph presidency speeches, 18.8% at 2, 40.0% at 3, then 71.5% at 4-5 and 73.7% at 6+ — against 77.9% for ordinary representative speeches. The break at 4 separates presiding boilerplate from substantive national statements.
- **Group statements that are also national statements.** Several delegations deliver an Arab Group statement and then continue in national capacity within the same speech record. The whole speech is classified `group_rep`; this is conservative (it removes some national-capacity text) and affects 15 speeches.
- **EU / OIC / NAM statements** are delivered either by briefers (already excluded) or by a member state whose speech is caught by rule 3; no separate organisational `country` value exists in the source data.
- **Holy See** (2 speeches, 8 paragraphs) is an observer state, not a UN member state, and is excluded from the analysis population.

## 7. Phases

| Phase | Start | End | Meetings | Paragraphs | Member-state paragraphs |
|---|---|---|---|---|---|
| `P1_war_onset` | 2023-10-01 | 2024-01-25 | 21 | 4,382 | 3,396 |
| `P2_post_icj` | 2024-01-26 | 2025-01-18 | 50 | 9,014 | 6,913 |
| `P3-P5_ceasefire_1_to_ceasefire_2` | 2025-01-19 | 2025-12-31 | 28 | 5,703 | 4,267 |
| `outside` | 2023-08-21 | 2023-09-27 | 2 | 252 | 195 |

Corpus date range: **2023-08-21 .. 2025-09-29**. The brief describes the corpus as running to December 2025; the data in fact stop on 2025-09-29. See `report.md` for the consequences.
