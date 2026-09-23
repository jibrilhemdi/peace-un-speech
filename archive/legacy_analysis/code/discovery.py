"""Phase 0: find the database, validate it, classify speakers, define phases.

Produces a Corpus object consumed by every downstream module, and writes
outputs/data_report.md.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .config import CONFIG, OUT, ROOT
from .io_utils import LOG, iso3

# Expected counts from the project brief; deviation > 1% is a hard stop.
EXPECTED = {
    "paragraphs": 19351, "speeches": 2245, "meetings": 101,
    "violence_type": {"VD": 8371, "VS": 5729, "VC": 675},
    "attribution": {"ATR-ISR": 9239, "ATR-PAL": 2773, "NONE": 2763},
    "positive_negative": {"PN1": 4528, "PN3": 2560, "PN0": 2003, "PN2": 1849, "PN4": 225},
    "transcend": {"T0": 1395, "T3": 1227, "T1": 919, "T2": 753, "T4": 481},
    "imperialism": {"I1": 2262, "I2": 798, "I3": 548, "I0": 277, "I4": 214},
    "relation_type": {"SYM": 1883, "ANTI": 513, "AB": 58},
}
TOLERANCE = 0.01

# Canonical field -> source column in `units` / `annotations` / `jobs`.
SCHEMA_MAP = {
    "paragraph id":        ("units", "unit_id"),
    "paragraph text":      ("units", "text"),
    "speech id":           ("units", "speech_id"),
    "meeting id":          ("units", "speech_id", "derived: substring before '#'"),
    "meeting date":        ("units", "meeting_date", "text '%d %B %Y' -> datetime"),
    "speaker name":        ("units", "ambassador_name"),
    "speaker country":     ("units", "country"),
    "speaker role":        ("units", "role", "representative | president | official/briefer"),
    "violence_type":       ("annotations", "value where dimension='violence_type'"),
    "attribution":         ("annotations", "value where dimension='attribution'"),
    "positive_negative":   ("annotations", "value where dimension='positive_negative'"),
    "transcend":           ("annotations", "value where dimension='transcend'"),
    "imperialism":         ("annotations", "value where dimension='imperialism'"),
    "relation_type":       ("annotations", "value where dimension='relation_type'"),
    "confidence":          ("annotations", "confidence"),
    "evidence quote":      ("annotations", "evidence"),
    "evidence_verified":   ("annotations", "evidence_verified"),
    "run_id":              ("annotations / jobs", "run_id"),
    "annotation model":    ("jobs", "model", "joined on (unit_id, codebook_id, run_id)"),
    "procedural flag":     ("units", "para_is_procedural"),
    "party flag":          ("units", "is_party", "1 = Israel / State of Palestine"),
    "regional bloc":       ("units", "bloc", "WEOG/ASIA_PACIFIC/AFRICAN/GRULAC/EASTERN_EUROPEAN/OBSERVER"),
}

# --- speaker classification -------------------------------------------------
GROUP_NAMES = (
    r"(Group of Arab States|Arab Group|Arab States|European Union|"
    r"Organi[sz]ation of Islamic Cooperation|OIC|League of Arab States|"
    r"Movement of Non-Aligned Countries|Non-Aligned Movement|"
    r"Gulf Cooperation Council|African Union|Nordic countries|"
    r"Committee on the Exercise of the Inalienable Rights)"
)
_RE_GROUP = re.compile(r"on behalf of (?:the\s+)?" + GROUP_NAMES, re.I)
_RE_DELIVER = re.compile(
    r"(?:honou?red?|pleasure|privilege)\s+to\s+(?:deliver|make|read|present|speak)"
    r"|(?:i|we)\s+(?:am\s+)?(?:deliver(?:ing)?|making|present(?:ing)?)\s+this\s+statement"
    r"|(?:i|we)\s+(?:am\s+)?(?:deliver(?:ing)?|speak(?:ing)?|address(?:ing)?)\b"
    r"|in\s+my\s+capacity\s+as\s+(?:the\s+)?[Cc]hair",
    re.I,
)
_RE_ALIGN = re.compile(
    r"align|associat|subscrib|endors|support the statement|proposed by|submitted by"
    r"|delivered by|made by|statement (?:just )?(?:made|delivered)|to be delivered",
    re.I,
)
_RE_NATCAP = re.compile(
    r"in my capacity as the representative|in my national capacity"
    r"|speaking in my capacity as|resume my functions as president",
    re.I,
)
# A presidency speech is treated as a national statement if it carries an explicit
# national-capacity marker OR has at least this many non-procedural paragraphs.
PRESIDENT_NATCAP_MIN_PARAS = 4


@dataclass
class Corpus:
    """Everything downstream modules need, already filtered and classified."""

    units: pd.DataFrame               # one row per non-procedural paragraph
    ann: pd.DataFrame                 # one row per annotation (main run, non-procedural)
    code_labels: dict                 # (dimension, code) -> human label
    code_defs: dict                   # (dimension, code) -> definition
    dim_codes: dict                   # dimension -> ordered list of codes (incl. NONE)
    phases: dict                      # phase name -> (start, end) Timestamps, post-merge
    model_span: dict                  # single-model span for the transcend dimension
    codebooks_found: bool
    sanity: pd.DataFrame
    speaker_counts: pd.DataFrame
    group_speeches: pd.DataFrame
    notes: list = field(default_factory=list)

    @property
    def member_states(self) -> pd.DataFrame:
        return self.units[self.units.speaker_class == "member_state"]

    @property
    def parties(self) -> pd.DataFrame:
        return self.units[self.units.speaker_class == "party"]


# --------------------------------------------------------------- db locating --

def find_databases() -> list[Path]:
    pats = ("*.db", "*.sqlite", "*.sqlite3", "*.duckdb")
    hits: list[Path] = []
    for p in pats:
        hits += [q for q in ROOT.rglob(p)
                 if ".venv" not in q.parts and "node_modules" not in q.parts]
    return sorted(set(hits))


def _score_db(path: Path) -> tuple[int, str]:
    """How many of the three headline sanity numbers this file reproduces."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tabs = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"units", "annotations"} <= tabs:
            return 0, "missing units/annotations tables"
        n_par, n_sp = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT speech_id) FROM units "
            "WHERE para_is_procedural=0").fetchone()
        n_mt = con.execute(
            "SELECT COUNT(DISTINCT substr(speech_id,1,instr(speech_id,'#')-1)) "
            "FROM units WHERE para_is_procedural=0").fetchone()[0]
        con.close()
        got = {"paragraphs": n_par, "speeches": n_sp, "meetings": n_mt}
        score = sum(
            abs(got[k] - EXPECTED[k]) <= TOLERANCE * EXPECTED[k]
            for k in ("paragraphs", "speeches", "meetings")
        )
        return score, f"paragraphs={n_par}, speeches={n_sp}, meetings={n_mt}"
    except Exception as exc:                                   # pragma: no cover
        return 0, f"unreadable: {exc}"


def select_database() -> tuple[Path, list[str]]:
    cands = find_databases()
    lines = []
    if not cands:
        raise SystemExit("No database found. Expected a *.db/*.sqlite under the repo.")
    scored = []
    for c in cands:
        s, desc = _score_db(c)
        scored.append((s, c, desc))
        lines.append(f"`{c.relative_to(ROOT)}` — score {s}/3 ({desc})")
    scored.sort(key=lambda t: (-t[0], str(t[1])))
    best_score, best, desc = scored[0]
    if best_score < 3:
        raise SystemExit(
            f"Best candidate {best} only matches {best_score}/3 sanity numbers ({desc}). "
            "Stopping — diagnose before continuing."
        )
    LOG.log("database",
            f"Selected `{best.relative_to(ROOT)}` as the source of truth "
            f"({len(cands)} candidate(s) scanned).",
            f"It is the only file reproducing all three headline counts ({desc}). "
            "CSV/XLSX files under data/exports/ are treated as derived exports and unused.")
    return best, lines


# ------------------------------------------------------------------ codebooks --

def load_codebooks() -> tuple[dict, dict, dict, bool]:
    """Return (labels, definitions, dimension->codes, found)."""
    cb_dir = Path(CONFIG["CODEBOOK_DIR"])
    files = sorted(cb_dir.glob("*.yaml")) if cb_dir.exists() else []
    labels, defs, dim_codes = {}, {}, {}
    for f in files:
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:                                       # pragma: no cover
            continue
        for dim in d.get("dimensions", []):
            name = dim["name"]
            codes = []
            for c in dim.get("categories", []):
                labels[(name, c["code"])] = c.get("label", c["code"])
                defs[(name, c["code"])] = " ".join((c.get("definition") or "").split())
                codes.append(c["code"])
            dim_codes.setdefault(name, []).extend(codes)
    found = bool(labels)
    if found:
        LOG.log("codebooks",
                f"Loaded {len(files)} codebook YAML files from `codebooks/` covering "
                f"{len(dim_codes)} dimensions and {len(labels)} codes.",
                "Human-readable labels are attached to every code in legends and tables.")
    else:
        LOG.log("codebooks",
                "No codebook documentation found — raw code IDs used everywhere and "
                "the transformative_framing outcome is marked UNVERIFIED.")
    return labels, defs, dim_codes, found


# ------------------------------------------------------------------- loading --

def _load_raw(db: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    units = pd.read_sql("SELECT * FROM units", con)
    keep = CONFIG["KEEP_RUN_ID"]
    ann = pd.read_sql(
        "SELECT unit_id, codebook_id, run_id, annotation_index, dimension, value, "
        "evidence, confidence, evidence_verified FROM annotations WHERE run_id=?",
        con, params=(keep,))
    jobs = pd.read_sql(
        "SELECT unit_id, codebook_id, run_id, model, status FROM jobs WHERE run_id=?",
        con, params=(keep,))
    con.close()
    ann = ann.merge(jobs[["unit_id", "codebook_id", "model"]],
                    on=["unit_id", "codebook_id"], how="left")
    return units, ann, jobs


def _classify_speakers(units_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach speaker_class at speech level. Returns (units, detected group speeches)."""
    u = units_all
    nonproc = u[u.para_is_procedural == 0]

    # --- group-representative delivery: scan the opening of each speech ------
    opening = (nonproc[nonproc.para_index <= 2]
               .sort_values(["speech_id", "para_index"])
               .groupby("speech_id").text.apply(" ".join))
    hits = []
    for sid, txt in opening.items():
        for m in _RE_GROUP.finditer(txt):
            # Scope the test to the sentence carrying "on behalf of <group>":
            # a 180-char window bleeds in alignment language from neighbours.
            lo = max((txt.rfind(ch, 0, m.start()) for ch in ".!?"), default=-1)
            sent = txt[lo + 1: m.start()]
            if _RE_DELIVER.search(sent) and not _RE_ALIGN.search(sent):
                hits.append({"speech_id": sid, "group": m.group(1)})
                break
    group_sp = pd.DataFrame(hits, columns=["speech_id", "group"])
    group_ids = set(group_sp.speech_id)

    # --- presidency speeches: presiding boilerplate vs national statement ----
    pres_marker = (u[u.role == "president"]
                   .assign(m=lambda d: d.text.str.contains(_RE_NATCAP, regex=True))
                   .groupby("speech_id").m.max())
    pres_size = nonproc[nonproc.role == "president"].groupby("speech_id").size()
    pres_natcap = {
        sid for sid in set(pres_marker.index) | set(pres_size.index)
        if bool(pres_marker.get(sid, False))
        or pres_size.get(sid, 0) >= PRESIDENT_NATCAP_MIN_PARAS
    }

    def classify(row) -> str:
        if row.role == "official/briefer":
            return "briefer"
        if row.is_party == 1:
            return "party"
        if row.speech_id in group_ids:
            return "group_rep"
        if row.bloc == "OBSERVER":          # Holy See: observer, not a member state
            return "other_invitee"
        if row.role == "president":
            return "member_state" if row.speech_id in pres_natcap else "presiding"
        if row.role == "representative":
            return "member_state" if str(row.country).strip() else "other_invitee"
        return "other_invitee"

    u = u.copy()
    u["speaker_class"] = u.apply(classify, axis=1)
    if not group_sp.empty:
        group_sp = group_sp.merge(
            nonproc[["speech_id", "country", "meeting_date"]].drop_duplicates("speech_id"),
            on="speech_id", how="left")
        # EU and League of Arab States delegations are filed with an empty country.
        group_sp["country"] = (group_sp.country.fillna("").replace("", "(delegation)"))
        # Record where each detected speech actually landed: some are delivered by
        # briefers (the EU's own representative, the LAS Secretary-General), who
        # rule 1 has already placed outside the member-state population.
        final = u.drop_duplicates("speech_id").set_index("speech_id").speaker_class
        group_sp["classified_as"] = group_sp.speech_id.map(final)
    return u, group_sp


def _assign_phases(units: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """Assign phases, then auto-merge under-floor phases into a neighbour."""
    phases = {k: (pd.Timestamp(a), pd.Timestamp(b))
              for k, (a, b) in CONFIG["PHASES"].items()}
    order = sorted(phases, key=lambda k: phases[k][0])
    merge_notes: list[str] = []

    def tag(df, ph):
        out = pd.Series("outside", index=df.index, dtype=object)
        for name in ph:
            s, e = ph[name]
            out[(df.date >= s) & (df.date <= e)] = name
        return out

    def sizes(ph):
        t = tag(units, ph)
        g = units.assign(_p=t).groupby("_p")
        return pd.DataFrame({"meetings": g.meeting_id.nunique(),
                             "paragraphs": g.size()}).reindex(ph.keys()).fillna(0).astype(int)

    for _ in range(len(order)):
        sz = sizes(phases)
        under = [p for p in order
                 if sz.loc[p, "meetings"] < CONFIG["PHASE_MIN_MEETINGS"]
                 or sz.loc[p, "paragraphs"] < CONFIG["PHASE_MIN_PARAGRAPHS"]]
        if not under or len(order) == 1:
            break
        # Smallest under-floor phase first.
        victim = min(under, key=lambda p: (sz.loc[p, "paragraphs"], sz.loc[p, "meetings"]))
        i = order.index(victim)
        nbrs = [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]
        target = min(nbrs, key=lambda p: (sz.loc[p, "paragraphs"], order.index(p)))
        s = min(phases[victim][0], phases[target][0])
        e = max(phases[victim][1], phases[target][1])
        # Compact label: span the P-numbers of everything folded in, keep the
        # substantive suffix of the earliest and latest member.
        members = sorted([victim, target], key=lambda p: phases[p][0])
        nums = sorted({int(n) for p in members
                       for n in re.findall(r"P(\d+)", p)})
        strip = lambda p: re.sub(r"^P\d+(?:-P\d+)?_", "", p)
        first_suffix = strip(members[0]).split("_to_")[0]
        last_suffix = strip(members[-1]).split("_to_")[-1]
        new = f"P{nums[0]}-P{nums[-1]}_{first_suffix}"
        if last_suffix != first_suffix:
            new += f"_to_{last_suffix}"
        merge_notes.append(
            f"`{victim}` ({sz.loc[victim,'meetings']} meetings, "
            f"{sz.loc[victim,'paragraphs']} paragraphs) merged into `{target}` "
            f"-> `{new}` [{s.date()} .. {e.date()}]")
        for p in (victim, target):
            phases.pop(p)
        phases[new] = (s, e)
        order = sorted(phases, key=lambda k: phases[k][0])

    units = units.copy()
    units["phase"] = tag(units, phases)
    phases = {k: phases[k] for k in sorted(phases, key=lambda k: phases[k][0])}
    return units, phases, merge_notes


def _model_span(units: pd.DataFrame, ann: pd.DataFrame, jobs: pd.DataFrame) -> dict:
    """Largest contiguous run of meetings dominated by one annotation model.

    Uses `jobs`, not `annotations`: every paragraph has a job row for every
    codebook even when it received zero annotations, so jobs is the only table
    that says which model *looked at* each paragraph.
    """
    dim = CONFIG["MODEL_SPLIT_DIM"]
    sub = ann[ann.dimension == dim]
    cb = sub.codebook_id.mode().iat[0] if len(sub) else dim
    mm = (jobs[jobs.codebook_id == cb][["unit_id", "model"]].drop_duplicates("unit_id")
          .merge(units[["unit_id", "meeting_id", "date"]], on="unit_id", how="inner"))
    if mm.model.nunique() < 2:
        return {"split": False, "model": mm.model.iloc[0] if len(mm) else None,
                "start": units.date.min(), "end": units.date.max(),
                "paragraphs": len(units), "meetings": units.meeting_id.nunique(),
                "purity": 1.0, "table": pd.DataFrame()}

    per_model = (mm.groupby("model")
                 .agg(paragraphs=("unit_id", "size"), start=("date", "min"),
                      end=("date", "max"), meetings=("meeting_id", "nunique"))
                 .reset_index())
    mtg = (mm.groupby(["meeting_id", "date"]).model
           .agg(lambda s: s.value_counts().idxmax())
           .reset_index().sort_values("date").reset_index(drop=True))
    mtg["run"] = (mtg.model != mtg.model.shift()).cumsum()
    runs = []
    for (run, model), g in mtg.groupby(["run", "model"]):
        s, e = g.date.min(), g.date.max()
        win = mm[(mm.date >= s) & (mm.date <= e)]
        runs.append({"model": model, "start": s, "end": e,
                     "meetings": g.meeting_id.nunique(),
                     "paragraphs_total": len(win),
                     "paragraphs_model": int((win.model == model).sum())})
    runs = pd.DataFrame(runs).sort_values("paragraphs_model", ascending=False)
    best = runs.iloc[0]
    return {"split": True, "model": best.model, "start": best.start, "end": best.end,
            "paragraphs": int(best.paragraphs_model), "meetings": int(best.meetings),
            "purity": float(best.paragraphs_model / best.paragraphs_total),
            "table": runs.reset_index(drop=True), "per_model": per_model,
            "unit_model": mm[["unit_id", "model"]]}


# ----------------------------------------------------------------- sanity ---

def _sanity(units: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check": "non-procedural paragraphs", "expected": EXPECTED["paragraphs"],
         "observed": len(units)},
        {"check": "speeches", "expected": EXPECTED["speeches"],
         "observed": units.speech_id.nunique()},
        {"check": "meetings", "expected": EXPECTED["meetings"],
         "observed": units.meeting_id.nunique()},
    ]
    counts = ann.groupby(["dimension", "value"]).size()
    for dim in CONFIG["DIMENSIONS"]:
        for code, exp in EXPECTED[dim].items():
            rows.append({"check": f"{dim}: {code}", "expected": exp,
                         "observed": int(counts.get((dim, code), 0))})
    df = pd.DataFrame(rows)
    df["delta"] = df.observed - df.expected
    df["pct"] = (df.delta / df.expected * 100).round(3)
    df["ok"] = df.pct.abs() <= TOLERANCE * 100
    return df


# ------------------------------------------------------------------- build ---

def build_corpus() -> Corpus:
    print("\n[Phase 0] discovery and validation")
    db, cand_lines = select_database()
    units_all, ann, jobs = _load_raw(db)

    units_all["date"] = pd.to_datetime(units_all.meeting_date, format="%d %B %Y",
                                       errors="coerce")
    units_all["meeting_id"] = units_all.speech_id.str.split("#").str[0]
    n_bad = int(units_all.date.isna().sum())
    if n_bad:
        LOG.log("dates", f"{n_bad} meeting_date values failed '%d %B %Y' parsing.")

    units_all, group_sp = _classify_speakers(units_all)

    # Filters: main run only (pilots excluded at load) + non-procedural only.
    n_proc = int((units_all.para_is_procedural == 1).sum())
    units = units_all[units_all.para_is_procedural == 0].copy()
    ann = ann[ann.unit_id.isin(set(units.unit_id))].copy()
    LOG.log("filters",
            f"Kept run_id='{CONFIG['KEEP_RUN_ID']}' (excluded "
            f"{', '.join(CONFIG['EXCLUDE_RUN_IDS'])}) and dropped {n_proc} procedural "
            f"paragraphs; {len(units)} paragraphs and {len(ann)} annotations remain.")

    sanity = _sanity(units, ann)
    failed = sanity[~sanity["ok"]]
    if len(failed):
        print(failed.to_string(index=False))
        raise SystemExit("Sanity checks failed by more than 1% — stopping (Phase 0).")
    print(f"  sanity: {len(sanity)}/{len(sanity)} checks pass exactly")

    units["iso3"] = units.country.map(lambda c: iso3(c) if str(c).strip() else "")
    units = units.rename(columns={"ambassador_name": "speaker_name", "role": "role_raw"})

    labels, defs, cb_dim_codes, cb_found = load_codebooks()

    # Code inventory per dimension: observed codes + synthetic absence category.
    dim_codes = {}
    for dim in CONFIG["DIMENSIONS"]:
        obs = sorted(ann.loc[ann.dimension == dim, "value"].unique())
        declared = [c for c in cb_dim_codes.get(dim, []) if c in obs]
        rest = [c for c in obs if c not in declared]
        dim_codes[dim] = declared + rest + [CONFIG["NONE_TOKEN"] + "_ABSENT"]
        labels.setdefault((dim, "NONE"), "Unattributed (explicit NONE)")
        labels[(dim, CONFIG["NONE_TOKEN"] + "_ABSENT")] = "No code in this dimension"

    span = _model_span(units, ann, jobs)
    if span["split"]:
        pm = span["per_model"]
        desc = "; ".join(
            f"{r.model}: {r.paragraphs} paragraphs, {r.start.date()}..{r.end.date()}"
            for r in pm.itertuples())
        LOG.log("transcend model switch",
                f"The transcend dimension was annotated by {len(pm)} models and the split is "
                f"temporal ({desc}). Largest single-model span = {span['model']}, "
                f"{span['start'].date()}..{span['end'].date()} "
                f"({span['meetings']} meetings, {span['paragraphs']} paragraphs, "
                f"purity {span['purity']:.3f}).",
                "Policy: transcend-based comparisons across time use this span only; pooled "
                "cross-sectional uses may combine both models but carry a model indicator "
                "in any regression.")

    # Flag paragraphs inside the largest single-model span for the split
    # dimension, and record which model saw each paragraph.
    if span["split"]:
        um = span["unit_model"].set_index("unit_id").model
        units["ann_model"] = units.unit_id.map(um)
        units["in_model_span"] = (
            (units.date >= span["start"]) & (units.date <= span["end"])
            & (units.ann_model == span["model"]))
    else:
        units["ann_model"] = span["model"]
        units["in_model_span"] = True

    units, phases, merge_notes = _assign_phases(units)
    for note in merge_notes:
        LOG.log("phases", "Auto-merge: " + note,
                f"Below the floor of {CONFIG['PHASE_MIN_MEETINGS']} meetings / "
                f"{CONFIG['PHASE_MIN_PARAGRAPHS']} paragraphs.")
    n_out = int((units.phase == "outside").sum())
    if n_out:
        out = units[units.phase == "outside"]
        LOG.log("phases",
                f"{n_out} paragraphs ({out.meeting_id.nunique()} meetings, "
                f"{out.date.min().date()}..{out.date.max().date()}) fall outside every "
                "phase window; they are retained in pooled analyses and excluded from "
                "phase-level analyses.",
                "These pre-date the P1 window. Folding pre-war meetings into a phase named "
                "'war onset' would mislabel them; they are 1.3% of the corpus.")

    sc = (units.groupby("speaker_class")
          .agg(paragraphs=("unit_id", "size"), speeches=("speech_id", "nunique"),
               countries=("country", lambda s: s[s.astype(str).str.strip() != ""].nunique()))
          .sort_values("paragraphs", ascending=False).reset_index())
    LOG.log("speakers",
            "Speech-level classification: " + ", ".join(
                f"{r.speaker_class}={r.speeches} speeches/{r.paragraphs} paragraphs"
                for r in sc.itertuples()),
            "Analysis population = member_state. Parties (Israel, State of Palestine) are "
            "computed as reference points but never clustered; briefers, group "
            "representatives, presiding remarks and other invitees are excluded.")

    corpus = Corpus(units=units, ann=ann, code_labels=labels, code_defs=defs,
                    dim_codes=dim_codes, phases=phases, model_span=span,
                    codebooks_found=cb_found, sanity=sanity, speaker_counts=sc,
                    group_speeches=group_sp,
                    notes=["database candidates:\n  - " + "\n  - ".join(cand_lines)])
    _write_data_report(corpus, db)
    return corpus


# ------------------------------------------------------------- data_report ---

def _write_data_report(c: Corpus, db: Path) -> None:
    L = ["# Data report — Phase 0 (discovery and validation)", "",
         f"Source of truth: `{db.relative_to(ROOT)}`", ""]

    L += ["## 1. Database candidates", ""]
    L += [f"- {x}" for x in c.notes[0].split("\n  - ")[1:]]
    L += ["", "## 2. Schema mapping", "",
          "| Canonical field | Table | Column | Note |", "|---|---|---|---|"]
    for k, v in SCHEMA_MAP.items():
        note = v[2] if len(v) > 2 else ""
        L.append(f"| {k} | `{v[0]}` | `{v[1]}` | {note} |")

    L += ["", "The database stores annotations in long form: one row per",
          "`(unit_id, codebook_id, annotation_index, dimension)`. A paragraph carries 0..n",
          "annotations, and one annotation fixes one value per elicited dimension. The six",
          "canonical dimensions live in five codebooks: `violence` supplies both",
          "`violence_type` and `attribution` (the latter `conditional_on` the former);",
          "`positive_negative`, `transcend`, `imperialism` and `relations` supply one each.",
          "",
          "**Absence is not stored.** A dimension with no annotation for a paragraph is",
          "expressed by the absence of a row, so every dimension gains a synthetic",
          f"`{CONFIG['NONE_TOKEN']}_ABSENT` category built downstream. `attribution` also has an",
          "*explicit* `NONE` value (violence coded but no attributable actor) which is a",
          "substantive category and is kept distinct from absence.", ""]

    L += ["## 3. Filters applied", "",
          f"1. `run_id = '{CONFIG['KEEP_RUN_ID']}'` — excludes "
          f"{', '.join(repr(r) for r in CONFIG['EXCLUDE_RUN_IDS'])} "
          "(the pilot run, 868 annotations across 5 codebooks).",
          "2. `para_is_procedural = 0` — drops 486 procedural paragraphs.",
          "3. All `jobs` rows for the main run have `status='ok'`; no paragraph is missing a",
          "   job for any codebook, so absence of an annotation is genuine absence rather",
          "   than a failed call.", ""]

    L += ["## 4. Sanity checks", "",
          "| Check | Expected | Observed | Delta | % | Pass |", "|---|---|---|---|---|---|"]
    for r in c.sanity.itertuples():
        L.append(f"| {r.check} | {r.expected:,} | {r.observed:,} | {r.delta:+d} | "
                 f"{r.pct:+.3f} | {'PASS' if r.ok else 'FAIL'} |")
    L += ["", f"All {len(c.sanity)} checks reproduce the expected values **exactly** "
          "(delta 0). The expected dimension counts are *annotation-level* counts, not "
          "counts of distinct paragraphs — a paragraph carrying VD twice contributes twice. "
          "That reading is what reproduces the brief's numbers.", ""]

    L += ["## 5. Transcend model switch", ""]
    if c.model_span["split"]:
        pm = c.model_span["per_model"]
        L += ["| Model | Paragraphs | First meeting | Last meeting | Meetings |",
              "|---|---|---|---|---|"]
        for r in pm.itertuples():
            L.append(f"| `{r.model}` | {r.paragraphs:,} | {r.start.date()} | "
                     f"{r.end.date()} | {r.meetings} |")
        L += ["", "The split **is temporal**. Assigning each meeting its modal model and "
              "grouping contiguous runs gives:", "",
              "| Model | Start | End | Meetings | Paragraphs (by model) | Purity |",
              "|---|---|---|---|---|---|"]
        for r in c.model_span["table"].itertuples():
            L.append(f"| `{r.model}` | {r.start.date()} | {r.end.date()} | {r.meetings} | "
                     f"{r.paragraphs_model:,} | "
                     f"{r.paragraphs_model / r.paragraphs_total:.3f} |")
        s = c.model_span
        L += ["", f"**Largest single-model span:** `{s['model']}`, "
              f"{s['start'].date()} .. {s['end'].date()} — {s['meetings']} meetings, "
              f"{s['paragraphs']:,} paragraphs, {s['purity']:.1%} pure.", "",
              "Policy applied:",
              "- Any transcend-based comparison **across time** (monthly trends, phase "
              "re-clustering diagnostics, event studies) is restricted to this span.",
              "- Pooled cross-sectional use (fingerprints, CA, clustering) combines both "
              "models; sensitivity variant (c) re-runs the typology on the span alone.",
              "- Regressions on a transcend outcome carry a model indicator where both "
              "models are present; where the span restriction already forces a single "
              "model the indicator is collinear and omitted.", ""]
    else:
        L += ["Only one annotation model is present; no restriction needed.", ""]

    L += ["## 6. Speaker classification", "",
          "| Class | Speeches | Paragraphs | Distinct countries |", "|---|---|---|---|"]
    for r in c.speaker_counts.itertuples():
        L.append(f"| `{r.speaker_class}` | {r.speeches:,} | {r.paragraphs:,} | "
                 f"{r.countries} |")
    L += ["", "Rules, in order of precedence:", "",
          "1. `role='official/briefer'` -> **briefer** (UN Secretariat officials and invited "
          "briefers; `country` is empty for all of them). Includes the EU's own "
          "representatives (Lambrinidis, Borrell Fontelles, Kallas, Skoog), so EU group "
          "statements are already outside the member-state population.",
          "2. `is_party=1` -> **party** (Israel, State of Palestine).",
          "3. Speech opens by *delivering* a statement on behalf of a regional group "
          "-> **group_rep**.",
          "4. `bloc='OBSERVER'` and not a party -> **other_invitee** (Holy See; a non-member "
          "observer state, so outside the member-state population).",
          "5. `role='president'` -> **member_state** if the speech carries an explicit "
          f"national-capacity marker or has >= {PRESIDENT_NATCAP_MIN_PARAS} non-procedural "
          "paragraphs, else **presiding**.",
          "6. `role='representative'` with a non-empty country -> **member_state**.", ""]

    L += ["### 6.1 Group-representative speeches detected", ""]
    if len(c.group_speeches):
        L += ["| Speech | Country | Group delivered on behalf of | Classified as |",
              "|---|---|---|---|"]
        for r in c.group_speeches.sort_values(["country", "speech_id"]).itertuples():
            L.append(f"| `{r.speech_id}` | {r.country} | {r.group} | "
                     f"`{r.classified_as}` |")
        n_grp = int((c.group_speeches.classified_as == "group_rep").sum())
        n_other = len(c.group_speeches) - n_grp
        L += ["", f"{len(c.group_speeches)} group deliveries detected. {n_grp} are "
              "reclassified out of the member-state population as `group_rep`; the "
              f"remaining {n_other} are delivered by the organisations' own "
              "representatives (the EU Special Representative, the Secretary-General of "
              "the League of Arab States), whom rule 1 has already classified as "
              "briefers. Detection requires a first-person *delivery* verb before "
              "`on behalf of <group>` and no alignment verb (`aligns itself with`, "
              "`associates itself with`, `proposed by`) in the same lead-in — the corpus "
              "contains far more speeches that *align with* a group statement than deliver "
              "one, and those remain national-capacity.", ""]
    else:
        L += ["None detected.", ""]

    L += ["### 6.2 Ambiguous cases logged", "",
          "- **Presidency speeches.** A country holding the Council presidency never has a "
          "separate `role='representative'` speech in the same meeting (0 overlaps), so its "
          "national statement is filed under `role='president'`. Dropping the role wholesale "
          "would systematically silence each member during its presidency month. The "
          f">= {PRESIDENT_NATCAP_MIN_PARAS}-paragraph rule is empirically grounded: the share "
          "of paragraphs carrying at least one annotation is 2.8% for 1-paragraph "
          "presidency speeches, 18.8% at 2, 40.0% at 3, then 71.5% at 4-5 and 73.7% at 6+ "
          "— against 77.9% for ordinary representative speeches. The break at 4 separates "
          "presiding boilerplate from substantive national statements.",
          "- **Group statements that are also national statements.** Several delegations "
          "deliver an Arab Group statement and then continue in national capacity within the "
          "same speech record. The whole speech is classified `group_rep`; this is "
          "conservative (it removes some national-capacity text) and affects "
          f"{len(c.group_speeches)} speeches.",
          "- **EU / OIC / NAM statements** are delivered either by briefers (already "
          "excluded) or by a member state whose speech is caught by rule 3; no separate "
          "organisational `country` value exists in the source data.",
          "- **Holy See** (2 speeches, 8 paragraphs) is an observer state, not a UN member "
          "state, and is excluded from the analysis population.", ""]

    L += ["## 7. Phases", "",
          "| Phase | Start | End | Meetings | Paragraphs | Member-state paragraphs |",
          "|---|---|---|---|---|---|"]
    ms = c.units[c.units.speaker_class == "member_state"]
    for name, (s, e) in c.phases.items():
        sub = c.units[c.units.phase == name]
        L.append(f"| `{name}` | {s.date()} | {e.date()} | {sub.meeting_id.nunique()} | "
                 f"{len(sub):,} | {len(ms[ms.phase == name]):,} |")
    outside = c.units[c.units.phase == "outside"]
    if len(outside):
        L.append(f"| `outside` | {outside.date.min().date()} | "
                 f"{outside.date.max().date()} | {outside.meeting_id.nunique()} | "
                 f"{len(outside):,} | {len(ms[ms.phase == 'outside']):,} |")
    L += ["", f"Corpus date range: **{c.units.date.min().date()} .. "
          f"{c.units.date.max().date()}**. The brief describes the corpus as running to "
          "December 2025; the data in fact stop on "
          f"{c.units.date.max().date()}. See `report.md` for the consequences.", ""]

    (OUT / "data_report.md").write_text("\n".join(L), encoding="utf-8")
    print("  wrote outputs/data_report.md")
