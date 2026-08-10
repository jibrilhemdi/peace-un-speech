# Codebook schema

This is the contract between a codebook and the annotation pipeline. A codebook is a single
YAML file in `codebooks/`. Adding a codebook means writing one file and running the pipeline —
**never editing Python**. Everything the pipeline needs (prompt text, output JSON shape,
validation rules, export columns) is derived from the fields below.

Validate with:

```bash
python scripts/codebook/validate.py
```

---

## 1. The data model

The annotation unit is a **paragraph**. A paragraph carries **0..n annotations**.

An **annotation** is one complete labelling event: exactly one value per elicited dimension,
plus the evidence span that licenses it.

```
paragraph
  └── annotations: []                       ← empty is normal and expected
       ├── annotation 0: {violence_type: VD, attribution: ATR-ISR, evidence: "...", confidence: 0.9}
       └── annotation 1: {violence_type: VS, attribution: ATR-ISR, evidence: "...", confidence: 0.7}
```

Two consequences that drive the whole design:

**Dimensions are separate variables, not a flat code list.** `VD ATR-ISR` is not one code; it
is `violence_type=VD` *and* `attribution=ATR-ISR`. Storing them separately is what lets the
downstream MCA distinguish *"Russia and Algeria share a structural conception of peace"* from
*"Russia and Algeria both blame Israel."* Collapsing them to six flat codes destroys that
contrast permanently. If a new codebook has a conceptually distinct second variable, give it
its own dimension.

**A paragraph doing two things yields two annotations, not one blended one.** A paragraph
describing both bombardment and blockade attributed to Israel produces two annotation objects
sharing `attribution=ATR-ISR` but differing in `violence_type`, each with its own evidence
span. Do not merge them.

### NONE is implicit

Every dimension always admits `NONE`. **Never declare it as a category.** `NONE` is a reserved
code and the validator rejects it.

`NONE` is expressed by *absence*: a paragraph with nothing to code returns `[]`. This is the
common outcome for most paragraphs, and the generated prompt says so explicitly. The main
threat to this project is not missed codes — it is over-assignment inflating prevalence until
every country looks alike and the variance the analysis needs is gone. `negative_examples`
exist to fight exactly that.

---

## 2. File layout

One file per codebook: `codebooks/<id>.yaml`. The filename stem **must** equal the `id` field.

### Top level

| key | required | type | notes |
| --- | --- | --- | --- |
| `id` | yes | string | `^[a-z][a-z0-9_]*$`. Must match the filename stem. Used as `codebook_id` in the DB and in export column names. Never change it after a run — it is the join key. |
| `version` | yes | int ≥ 1 | Bump to invalidate cached annotations. See §6. |
| `title` | yes | string | Human-readable, appears in the prompt header. |
| `theory` | no | string | Theoretical warrant, e.g. `"Galtung (1969, 1990)"`. Rendered into the prompt — it measurably helps the model hold the construct. |
| `unit` | yes | `paragraph` | Only `paragraph` is supported. Present so the field exists when a sentence-level codebook arrives. |
| `allow_multiple_annotations` | no | bool (default `true`) | `false` caps a paragraph at one annotation. |
| `max_annotations_per_unit` | no | int | Guard against runaway output. Exceeding it is a `parse_error`, not a silent truncation. |
| `instructions` | no | string | Free-text guidance injected verbatim after the dimension blocks. Use for cross-cutting rules that belong to no single category. |
| `dimensions` | yes | list, ≥ 1 | See §3. |
| `worked_examples` | no | list | Whole-paragraph examples showing the complete expected output. See §5. Strongly recommended. |
| `_anything` | no | any | Top-level keys starting with `_` are ignored by the validator and the prompt renderer. Use them as a YAML anchor scratch area (§4.1). |

---

## 3. Dimensions

```yaml
dimensions:
  - name: violence_type
    description: Which Galtungian form of violence the paragraph invokes.
    required: true
    categories: [...]
```

| key | required | type | notes |
| --- | --- | --- | --- |
| `name` | yes | string | `^[a-z][a-z0-9_]*$`, unique within the codebook. Becomes the `dimension` value in `annotations_long.parquet` and part of the column name in `unit_code_matrix.parquet`. Never change it after a run. |
| `description` | no | string | One line telling the model what the *variable* measures, as opposed to what each category means. Worth writing — it is the difference between the model picking a category and the model understanding the question. |
| `score_aggregation` | no | `max` (default), `min`, `mean`, `first` | Ordinal only. How several annotations on one paragraph collapse to the single `__score` column in the matrix export. |
| `scale` | no | `nominal` (default) or `ordinal` | `ordinal` declares an ordered scale — the categories run low to high and the distance between them is meaningful in one direction. See §3.3. |
| `required` | no | bool (default `true`) | See below. |
| `conditional_on` | no | string or map | See §3.2. |
| `allow_multiple_values` | no | bool (default `false`) | `true` lets one annotation hold a list of codes for this dimension. Use sparingly — it complicates the matrix export. Prefer emitting two annotations. Not permitted on an ordinal dimension. |
| `categories` | yes | list, ≥ 2 | Fewer than two is a validation **failure**: a one-category dimension is a boolean flag, and boolean flags belong in a dimension with a real contrast. |

### 3.1 `required`

- `required: true` — in every annotation where this dimension is elicited, the value must be a
  real category code. It can never be `NONE`.
- `required: false` — the value may be `NONE` *inside* an annotation. Use this when the
  construct can genuinely be present but indeterminate: violence described with no attributable
  actor ("the violence must stop"), a relation asserted with no named counterpart.

Note the asymmetry: `required: false` does **not** mean optional-to-answer. It means `NONE` is
a substantively meaningful answer that the analysis will treat as a category.

### 3.2 `conditional_on`

A dimension is elicited only when another dimension is satisfied within the same annotation.

Short form — elicited when the named dimension is non-`NONE`:

```yaml
conditional_on: violence_type
```

Long form — elicited only for specific values:

```yaml
conditional_on:
  dimension: violence_type
  in: [VD, VS]
```

When the condition does not hold, the dimension's value is `NONE` and the prompt does not ask
for it. The validator rejects unknown dimension names, unknown codes, and cycles.

`conditional_on` has two jobs. In the prompt it says *"only assign attribution once you have
assigned a violence type"* — which suppresses free-floating blame codes. In validation it
rejects a returned annotation where the parent is `NONE` but the child is not.

### 3.3 Ordinal scales

Some constructs are not a set of alternatives but a **position on a scale**: how asymmetric a
proposed settlement is, how far a demand goes beyond a ceasefire, how zero-sum a framing is.
Declare `scale: ordinal` and give every category a `value`:

```yaml
- name: imperialism
  scale: ordinal
  description: How the speaker distributes control over the settlement.
  required: true
  categories:
    - code: IMP0
      value: 0
      label: Total asymmetry
      definition: >
        One party dictates absolute terms; the other must submit to externally set
        benchmarks it had no part in setting.
    - code: IMP4
      value: 4
      label: Total equality
      definition: >
        Absolute autonomy and self-determination; asymmetrical or colonial relations are
        explicitly rejected.
```

`code` stays a symbolic identifier — it names the export column, so it must remain stable and
match `^[A-Z][A-Z0-9_-]*$`. `value` carries the position. Keeping them separate means the
numeric scale can be re-anchored (0–4 → 1–5, or a midpoint moved) without renaming a column
and orphaning existing data.

What changes when a dimension is ordinal:

- **The prompt** presents the categories in `value` order, names the endpoints, and states that
  the scale is ordered — which measurably improves consistency over presenting them as an
  unordered list.
- **The export** gains a numeric column `<codebook>__<dimension>__score` alongside the usual
  binary indicators, so the scale can go into a regression directly while the same annotation
  still enters an MCA as a set of categories.
- `allow_multiple_values` is rejected. A position on a scale is one value *per annotation*.
- `allow_multiple_annotations` behaves exactly as it does for a nominal codebook, and should
  normally stay `true`. A paragraph that demands a ceasefire in one sentence and structural
  change in another has taken two positions on the scale; recording both as separate
  annotations is the same rule as *"bombardment + blockade yields two annotations"* (§1).
  Averaging them into one score would discard the very contrast the scale exists to measure.
- `score_aggregation` decides how several steps on one paragraph collapse to the single number
  in the matrix export: `max` (default), `min`, `mean`, or `first`. These scales are
  *how-far-along* measures, so `max` answers "how far toward positive peace did this paragraph
  get". Set it deliberately — a silent default here would be a buried analytical assumption.
  The long export always keeps every annotation, so nothing is lost either way.

**Score 0 is a real score, not absence.** This is the trap. On a 0–4 scale, `0` means the
construct is present at its lowest level — a speaker who demands a ceasefire and nothing more
scores 0 on a negative/positive-peace scale. A paragraph that says nothing about the construct
at all gets **no annotation**, exactly as in a nominal codebook. Conflating the two would put
every procedural paragraph at the bottom of the scale and destroy the measure. The generated
prompt says this explicitly; keep saying it in your `instructions` too.

Values need not start at 0 and need not be contiguous, but gaps are warned about — usually they
mean a category was deleted and the scale should be re-anchored.

---

## 4. Categories

```yaml
- code: VD
  label: Direct violence
  definition: >
    Direct physical violence — a concrete act causing bodily harm: killing, injury,
    bombardment, physical attack, hostage-taking.
  includes:
    - Named military operations with described physical effect
    - Casualty figures cited as the consequence of an act
  excludes:
    - Generic calls to "end the violence" with no act described
    - Threats of future action not yet carried out
  positive_examples: [...]
  negative_examples: [...]
```

| key | required | type | notes |
| --- | --- | --- | --- |
| `code` | yes | string | `^[A-Z][A-Z0-9_-]*$`, unique within its dimension. `NONE` is reserved and rejected. Becomes the `value` in the long export. Never change it after a run. |
| `value` | only when `scale: ordinal` | int | Position on the scale. Unique within the dimension. See §3.3. Ignored — and rejected — on a nominal dimension. |
| `label` | yes | string | Short human name. |
| `definition` | yes | string | The substantive criterion. Must be non-empty. Write the *boundary*, not a synonym for the label — "structural violence" as a definition of Structural violence tells the model nothing. |
| `includes` | no | list of strings | Short affirmative boundary rules. |
| `excludes` | no | list of strings | Short negative boundary rules. Cheaper than a full negative example when you just need a line. |
| `positive_examples` | no | list | See §4.1. Warn if empty. |
| `negative_examples` | no | list | See §4.1. **Warn if empty** — deliberately nagging. |

### 4.1 Examples

Both example lists take the same shape:

| key | required | type | notes |
| --- | --- | --- | --- |
| `text` | yes | string | The span, quoted from a real speech. Ellipsis (`...`) for elision is fine. |
| `source` | no | string | Provenance, `S/PV.9442, Nebenzia (Russian Federation)`. Keep it — it is what makes the codebook auditable, and it is what you will cite. |
| `reasoning` | positive: recommended<br>negative: **required** | string | *Why* this is or is not the code. For negatives, say what the miscoding would cost — that framing transfers to the model better than a bare "this is not VD". |
| `annotation` | no | map `dimension → code` | The **full joint label** for this span across all dimensions. See below. |

**`annotation` is the field that makes multi-dimension codebooks work.** Without it, a model
reading the `violence_type: VD` section sees a span and learns "this is VD" — but never sees a
complete output object. With it:

```yaml
- text: "condemning indiscriminate attacks on the civilian population..."
  source: "S/PV.9442, Nebenzia (Russian Federation)"
  annotation: {violence_type: VD, attribution: ATR-ISR}
  reasoning: "Concrete physical action causing casualties, with Israel named as the cause."
```

the model sees the shape it must emit. Always set `annotation` on positive examples in a
multi-dimension codebook. The validator checks every key and value against the declared
dimensions, so a typo here fails the lint rather than silently teaching the model a code that
does not exist.

**Sharing one example across dimensions.** A span is usually evidence for *every* dimension at
once, and both dimension blocks want it. Rather than copy-pasting, define it once in a
`_`-prefixed top-level block and alias it — standard YAML, resolved at load time:

```yaml
_examples:
  vd_isr_1: &vd_isr_1
    text: "condemning indiscriminate attacks on the civilian population..."
    source: "S/PV.9442, Nebenzia (Russian Federation)"
    annotation: {violence_type: VD, attribution: ATR-ISR}
    reasoning: "..."

dimensions:
  - name: violence_type
    categories:
      - code: VD
        positive_examples: [*vd_isr_1]
  - name: attribution
    categories:
      - code: ATR-ISR
        positive_examples: [*vd_isr_1]
```

Edit the anchor, both dimensions update. This is optional — inline duplication is equally
valid and the renderer cannot tell the difference. Single-dimension codebooks should just
inline.

(Implementation note: PyYAML resolves an alias to the *same object*, not a copy, so both
dimensions hold one shared dict. The pipeline therefore treats loaded codebooks as read-only
and never mutates an example in place — mutating one would silently alter the other.)

The duplicated text costs prompt tokens, but the codebook block is static and sits at the
front of every request, so prompt-prefix caching absorbs it.

---

## 5. `worked_examples`

Category examples teach *what a code means*. Worked examples teach *what a response looks
like* — including, critically, the empty response.

```yaml
worked_examples:
  - text: >
      We condemn the violence and call on all parties to exercise restraint and return to
      the negotiating table.
    annotations: []
    reasoning: >
      Diplomatic boilerplate. No physical act, no structural arrangement, no narrative claim,
      and no actor named. Coding this would inflate every prevalence estimate with filler
      that appears in nearly every speech.

  - text: >
      The blockade must be lifted and the indiscriminate bombardment of Gaza must stop.
    annotations:
      - {violence_type: VS, attribution: ATR-ISR, evidence: "The blockade must be lifted"}
      - {violence_type: VD, attribution: ATR-ISR, evidence: "the indiscriminate bombardment of Gaza must stop"}
    reasoning: >
      One paragraph, two distinct violence forms, same attribution — therefore two annotation
      objects, not one.
```

| key | required | notes |
| --- | --- | --- |
| `text` | yes | A paragraph, invented or real. |
| `annotations` | yes | The exact list the model should return for it. May be `[]`. |
| `reasoning` | no | Rendered into the prompt. Explain the *decision*, not the definitions. |
| `source` | no | If real. |

**Include at least one `annotations: []` example.** It is the single highest-leverage line in a
codebook. Without it the model infers that returning nothing is a failure and starts reaching.
The validator warns when there is no empty-annotation worked example.

---

## 6. Versioning

`version` is an integer. Bump it whenever a change could alter a label: any edit to a
definition, example, `includes`/`excludes`, `instructions`, or the dimension/category
structure. Typo fixes in a `label` do not require a bump.

Bumping `version` invalidates cached annotations **for that codebook only**. Job rows are keyed
`(unit_id, codebook_id, codebook_version, run_id)`, so `violence` v2 starts a fresh set of
pending jobs while `relations` v1 is untouched and complete. Old rows are retained, never
deleted — v1 and v2 sit side by side, which is what lets you measure how much a prompt revision
moved the labels.

The pilot loop this is built for:

```
write v1 → pilot 40 speeches → adjudicate → revise negative_examples → version: 2 → full run
```

---

## 7. What the pipeline generates from this file

You do not write any of this; it is derived. It is documented so you can predict the effect of
a schema change.

### Prompt

Assembled in this order, so the static prefix is byte-identical across every request for a
codebook version and prefix caching applies:

1. Task framing and the annotation model (constant)
2. `title`, `theory`, `unit`
3. Each dimension: `name`, `description`, `required`/`conditional_on` semantics, then each
   category's `label`, `definition`, `includes`/`excludes`, positive and negative examples
4. `instructions`
5. `worked_examples`
6. Output-format spec and the `NONE`-is-normal statement
7. **The speech**, as numbered paragraphs ← the only variable part, always last

### Output contract

A JSON object keyed by paragraph number *as sent*, every paragraph present:

```json
{
  "1": [],
  "2": [{"violence_type": "VD", "attribution": "ATR-ISR",
         "evidence": "indiscriminate attacks on the civilian population", "confidence": 0.85}],
  "3": [{"violence_type": "VS", "attribution": "ATR-ISR", "evidence": "the blockade", "confidence": 0.7},
        {"violence_type": "VC", "attribution": "ATR-ISR", "evidence": "a distorted narrative", "confidence": 0.5}]
}
```

- Keys are strings. A returned key set that does not exactly match what was sent is rejected
  and re-requested — a misaligned batch would silently mislabel an entire speech, which is the
  worst available failure mode and is never accepted.
- `evidence` is required and must be quoted verbatim from *that* paragraph. The pipeline
  verifies it: exact substring after whitespace normalisation, or every `...`-delimited
  fragment present in order. A span that fails verification is stored with an
  `evidence_unverified` flag rather than discarded — you can filter on it, and its rate is a
  usable quality signal.
- `confidence` is an optional float in `[0, 1]`. Recorded because you asked for the column, but
  treat it as a triage aid for gold-sample stratification, not a calibrated probability —
  self-reported LLM confidence is weakly calibrated and should not enter a model as a weight.
- A dimension not elicited under `conditional_on` may be omitted or set to `"NONE"`.

### Export

- `annotations_long.parquet` — one row per `(unit_id, codebook_id, annotation_index, dimension)`
- `unit_code_matrix.parquet` — one binary column per `(dimension, code)`, named
  `<codebook_id>__<dimension>__<code>`
- `country_profiles.parquet` — per country, the share of its non-procedural paragraphs carrying
  each code

New dimensions and codes appear as new columns automatically. Nothing to edit.

---

## 8. Validator rules

`python scripts/codebook/validate.py`

**Fail** — the file will not run:

- missing `id`, `version`, `title`, `unit`, or `dimensions`
- `id` not snake_case, or not matching the filename stem
- `version` not an integer ≥ 1
- `unit` not `paragraph`
- duplicate dimension `name`
- a dimension with fewer than two categories
- duplicate `code` within a dimension
- a category whose `code` is `NONE` (reserved)
- missing or empty `definition`, `label`, or `code`
- an example with no `text`
- a negative example with no `reasoning`
- `conditional_on` naming an unknown dimension, listing an unknown code, or forming a cycle
- an `annotation` map referencing an unknown dimension or code
- a `worked_examples` annotation violating `required` or `conditional_on`
- `scale` set to anything other than `nominal` or `ordinal`
- `score_aggregation` set to anything other than `max`, `min`, `mean`, `first`
- `score_aggregation` on a nominal dimension
- an ordinal dimension with a category missing `value`, or with duplicate values
- a `value` on a category of a nominal dimension
- `allow_multiple_values: true` on an ordinal dimension

**Warn** — runs, but you will be told:

- a category with no `negative_examples` ← the nag you asked for
- a category with no `positive_examples`
- no `worked_examples`, or none with `annotations: []`
- two categories in one dimension whose definitions overlap heavily by token similarity
- a `code` reused across dimensions (legal, but ambiguous in conversation)
- a `definition` under 10 words
- an ordinal dimension whose `value`s have gaps (usually a deleted category — re-anchor)

---

## 9. Adding a codebook

```bash
# 1. write it
$EDITOR codebooks/legitimacy.yaml

# 2. lint
python scripts/codebook/validate.py

# 3. pilot — ~40 speeches, under an hour of quota
python scripts/annotate/run_annotation.py --codebook legitimacy --pilot 40

# 4. adjudicate data/pilot_legitimacy_<timestamp>.csv, write negative_examples, bump version

# 5. full run, resumable, one day at a time
python scripts/annotate/run_annotation.py --codebook legitimacy

# 6. check
python scripts/annotate/report_status.py

# 7. export
python scripts/export/build_tables.py
```

No Python is edited at any step. If you find yourself needing to, the schema is missing a
field — add the field here first.
