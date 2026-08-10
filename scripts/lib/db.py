"""SQLite job store.

Design rules, in priority order:

1. Never lose or silently corrupt work. Everything a speech produces is written in one
   transaction, so a Ctrl-C leaves either all of a speech's rows or none of them.
2. Quota accounting is committed at attempt time, before the outcome is known. A crash
   mid-request must still count the request, because the provider counted it.
3. ``response_json`` is the source of truth. The ``annotations`` table is derived from it in
   the same transaction, purely so that status and export do not have to re-parse 2,000
   responses.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS units (
  unit_id           TEXT PRIMARY KEY,
  speech_id         TEXT NOT NULL,
  para_index        INTEGER NOT NULL,
  text              TEXT NOT NULL,
  country           TEXT,
  ambassador_name   TEXT,
  role              TEXT,
  language          TEXT,
  meeting_date      TEXT,
  bloc              TEXT,
  is_party          INTEGER NOT NULL DEFAULT 0,
  para_is_procedural INTEGER NOT NULL DEFAULT 0,
  n_words           INTEGER NOT NULL,
  segmented_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_speech  ON units(speech_id);
CREATE INDEX IF NOT EXISTS idx_units_country ON units(country);

CREATE TABLE IF NOT EXISTS jobs (
  unit_id           TEXT NOT NULL,
  codebook_id       TEXT NOT NULL,
  codebook_version  INTEGER NOT NULL,
  run_id            TEXT NOT NULL,
  speech_id         TEXT,
  status            TEXT NOT NULL,          -- pending | ok | parse_error | api_error | skipped
  request_hash      TEXT,                   -- hash of the fully rendered prompt (prefix+speech)
  prompt_hash       TEXT,                   -- hash of the static prefix only; detects drift
  model             TEXT,                   -- model actually resolved and used
  response_json     TEXT,                   -- raw model output
  error             TEXT,
  attempts          INTEGER NOT NULL DEFAULT 0,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  n_annotations     INTEGER,
  n_evidence_unverified INTEGER,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL,
  PRIMARY KEY (unit_id, codebook_id, codebook_version, run_id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_cb     ON jobs(codebook_id, codebook_version, run_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_speech ON jobs(speech_id, codebook_id, codebook_version, run_id);

CREATE TABLE IF NOT EXISTS annotations (
  unit_id           TEXT NOT NULL,
  codebook_id       TEXT NOT NULL,
  codebook_version  INTEGER NOT NULL,
  run_id            TEXT NOT NULL,
  annotation_index  INTEGER NOT NULL,
  dimension         TEXT NOT NULL,
  value             TEXT NOT NULL,
  evidence          TEXT,
  confidence        REAL,
  evidence_verified INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (unit_id, codebook_id, codebook_version, run_id, annotation_index, dimension)
);
CREATE INDEX IF NOT EXISTS idx_ann_cb ON annotations(codebook_id, codebook_version, run_id, dimension, value);

CREATE TABLE IF NOT EXISTS quota (
  day           TEXT PRIMARY KEY,
  requests_used INTEGER NOT NULL DEFAULT 0,
  updated_at    TEXT NOT NULL
);

-- What a model turned out not to support, learned at runtime. Saves rediscovering it (and
-- burning a request) on every resumed run.
CREATE TABLE IF NOT EXISTS provider_caps (
  model              TEXT PRIMARY KEY,
  supports_json_mode INTEGER NOT NULL DEFAULT 1,
  note               TEXT,
  detected_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pilots (
  codebook_id      TEXT NOT NULL,
  codebook_version INTEGER NOT NULL,
  run_id           TEXT NOT NULL,
  n_speeches       INTEGER NOT NULL,
  n_units          INTEGER NOT NULL,
  csv_path         TEXT,
  prompt_hash      TEXT,
  model            TEXT,
  completed_at     TEXT NOT NULL,
  PRIMARY KEY (codebook_id, codebook_version, run_id)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quota_day(reset_hour: int = 0, now: datetime | None = None) -> str:
    """The provider-day a moment belongs to, given a reset hour in UTC."""
    n = now or datetime.now(timezone.utc)
    return (n - timedelta(hours=reset_hour)).strftime("%Y-%m-%d")


def next_reset(reset_hour: int = 0, now: datetime | None = None) -> datetime:
    n = now or datetime.now(timezone.utc)
    today = n.replace(hour=reset_hour % 24, minute=0, second=0, microsecond=0)
    return today if today > n else today + timedelta(days=1)


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit of work. Rolls back on any exception, including KeyboardInterrupt."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- units --------------------------------------------------------------------------
    def replace_units(self, rows: Sequence[dict[str, Any]]) -> None:
        stamp = utcnow()
        with self.transaction() as c:
            c.execute("DELETE FROM units")
            c.executemany(
                """INSERT INTO units
                   (unit_id, speech_id, para_index, text, country, ambassador_name, role,
                    language, meeting_date, bloc, is_party, para_is_procedural, n_words,
                    segmented_at)
                   VALUES (:unit_id, :speech_id, :para_index, :text, :country,
                           :ambassador_name, :role, :language, :meeting_date, :bloc,
                           :is_party, :para_is_procedural, :n_words, :segmented_at)""",
                [{**r, "segmented_at": stamp} for r in rows],
            )

    def existing_unit_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT unit_id FROM units")}

    def units_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM units").fetchone()[0])

    # -- work selection -----------------------------------------------------------------
    def pending_speeches(
        self,
        codebook_id: str,
        version: int,
        run_id: str,
        *,
        include_procedural: bool = False,
        speech_ids: Iterable[str] | None = None,
    ) -> list[str]:
        """Speeches with at least one unit lacking an 'ok' job for this codebook version.

        Grouping by speech is what makes resume exact: a speech is either fully annotated or
        fully pending, because all its rows are written in one transaction.
        """
        where = "" if include_procedural else "WHERE u.para_is_procedural = 0"
        sql = f"""
            SELECT u.speech_id
            FROM units u
            LEFT JOIN jobs j
              ON j.unit_id = u.unit_id
             AND j.codebook_id = ? AND j.codebook_version = ? AND j.run_id = ?
             AND j.status = 'ok'
            {where}
            GROUP BY u.speech_id
            HAVING SUM(CASE WHEN j.unit_id IS NULL THEN 1 ELSE 0 END) > 0
            ORDER BY u.speech_id
        """
        out = [r[0] for r in self.conn.execute(sql, (codebook_id, version, run_id))]
        if speech_ids is not None:
            keep = set(speech_ids)
            out = [s for s in out if s in keep]
        return out

    def speech_units(self, speech_id: str, *, include_procedural: bool = False) -> list[sqlite3.Row]:
        where = "" if include_procedural else "AND para_is_procedural = 0"
        return list(
            self.conn.execute(
                f"SELECT * FROM units WHERE speech_id = ? {where} ORDER BY para_index",
                (speech_id,),
            )
        )

    def all_speech_ids(self, *, include_procedural: bool = False) -> list[str]:
        where = "" if include_procedural else "WHERE para_is_procedural = 0"
        return [
            r[0]
            for r in self.conn.execute(
                f"SELECT DISTINCT speech_id FROM units {where} ORDER BY speech_id"
            )
        ]

    # -- job writes ---------------------------------------------------------------------
    def write_speech_result(
        self,
        *,
        speech_id: str,
        codebook_id: str,
        version: int,
        run_id: str,
        unit_ids: Sequence[str],
        status: str,
        request_hash: str | None,
        prompt_hash: str | None,
        model: str | None,
        response_json: str | None,
        error: str | None,
        attempts: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        annotations_by_unit: dict[str, list] | None = None,
    ) -> None:
        """All rows for one speech, atomically. This is the commit-per-speech guarantee."""
        now = utcnow()
        anns = annotations_by_unit or {}
        with self.transaction() as c:
            for uid in unit_ids:
                per_unit = anns.get(uid, [])
                n_unver = sum(1 for a in per_unit if not a.evidence_verified)
                c.execute(
                    """INSERT INTO jobs
                       (unit_id, codebook_id, codebook_version, run_id, speech_id, status,
                        request_hash, prompt_hash, model, response_json, error, attempts,
                        prompt_tokens, completion_tokens, n_annotations,
                        n_evidence_unverified, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(unit_id, codebook_id, codebook_version, run_id) DO UPDATE SET
                         status=excluded.status,
                         request_hash=excluded.request_hash,
                         prompt_hash=excluded.prompt_hash,
                         model=excluded.model,
                         response_json=excluded.response_json,
                         error=excluded.error,
                         attempts=jobs.attempts + excluded.attempts,
                         prompt_tokens=excluded.prompt_tokens,
                         completion_tokens=excluded.completion_tokens,
                         n_annotations=excluded.n_annotations,
                         n_evidence_unverified=excluded.n_evidence_unverified,
                         updated_at=excluded.updated_at""",
                    (
                        uid, codebook_id, version, run_id, speech_id, status,
                        request_hash, prompt_hash, model, response_json, error, attempts,
                        prompt_tokens, completion_tokens,
                        len(per_unit) if status == "ok" else None,
                        n_unver if status == "ok" else None,
                        now, now,
                    ),
                )
                c.execute(
                    """DELETE FROM annotations
                       WHERE unit_id=? AND codebook_id=? AND codebook_version=? AND run_id=?""",
                    (uid, codebook_id, version, run_id),
                )
                for a in per_unit:
                    for dim, val in a.values.items():
                        c.execute(
                            """INSERT INTO annotations
                               (unit_id, codebook_id, codebook_version, run_id,
                                annotation_index, dimension, value, evidence, confidence,
                                evidence_verified)
                               VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            (
                                uid, codebook_id, version, run_id, a.annotation_index,
                                dim, val, a.evidence, a.confidence,
                                1 if a.evidence_verified else 0,
                            ),
                        )

    def clear_jobs(self, codebook_id: str, version: int, run_id: str) -> int:
        with self.transaction() as c:
            n = c.execute(
                "DELETE FROM jobs WHERE codebook_id=? AND codebook_version=? AND run_id=?",
                (codebook_id, version, run_id),
            ).rowcount
            c.execute(
                "DELETE FROM annotations WHERE codebook_id=? AND codebook_version=? AND run_id=?",
                (codebook_id, version, run_id),
            )
        return n

    # -- quota --------------------------------------------------------------------------
    def quota_used(self, day: str) -> int:
        row = self.conn.execute("SELECT requests_used FROM quota WHERE day=?", (day,)).fetchone()
        return int(row[0]) if row else 0

    def record_attempt(self, day: str) -> int:
        """Increment and commit immediately.

        Deliberately outside the per-speech transaction: the provider counts an attempt the
        moment it is made, so a crash between request and response must not lose the count.
        """
        with self.transaction() as c:
            c.execute(
                """INSERT INTO quota (day, requests_used, updated_at) VALUES (?, 1, ?)
                   ON CONFLICT(day) DO UPDATE SET
                     requests_used = quota.requests_used + 1,
                     updated_at = excluded.updated_at""",
                (day, utcnow()),
            )
        return self.quota_used(day)

    # -- pilots -------------------------------------------------------------------------
    def record_pilot(
        self,
        *,
        codebook_id: str,
        version: int,
        run_id: str,
        n_speeches: int,
        n_units: int,
        csv_path: str,
        prompt_hash: str,
        model: str,
    ) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO pilots (codebook_id, codebook_version, run_id, n_speeches,
                                       n_units, csv_path, prompt_hash, model, completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(codebook_id, codebook_version, run_id) DO UPDATE SET
                     n_speeches=excluded.n_speeches, n_units=excluded.n_units,
                     csv_path=excluded.csv_path, prompt_hash=excluded.prompt_hash,
                     model=excluded.model, completed_at=excluded.completed_at""",
                (codebook_id, version, run_id, n_speeches, n_units, csv_path,
                 prompt_hash, model, utcnow()),
            )

    def pilot_for(self, codebook_id: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM pilots WHERE codebook_id=? AND codebook_version=?
               ORDER BY completed_at DESC LIMIT 1""",
            (codebook_id, version),
        ).fetchone()

    # -- provider capabilities ----------------------------------------------------------
    def supports_json_mode(self, model: str) -> bool:
        row = self.conn.execute(
            "SELECT supports_json_mode FROM provider_caps WHERE model=?", (model,)
        ).fetchone()
        return True if row is None else bool(row[0])

    def set_json_mode_support(self, model: str, supported: bool, note: str = "") -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO provider_caps (model, supports_json_mode, note, detected_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(model) DO UPDATE SET
                     supports_json_mode=excluded.supports_json_mode,
                     note=excluded.note, detected_at=excluded.detected_at""",
                (model, 1 if supported else 0, note[:500], utcnow()),
            )

    # -- reporting ----------------------------------------------------------------------
    def status_counts(self, codebook_id: str, version: int, run_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            """SELECT status, COUNT(*) FROM jobs
               WHERE codebook_id=? AND codebook_version=? AND run_id=? GROUP BY status""",
            (codebook_id, version, run_id),
        )
        return {r[0]: int(r[1]) for r in rows}

    def distinct_models(self, codebook_id: str, version: int, run_id: str) -> list[tuple[str, int]]:
        return [
            (r[0], int(r[1]))
            for r in self.conn.execute(
                """SELECT model, COUNT(*) FROM jobs
                   WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'
                   GROUP BY model ORDER BY 2 DESC""",
                (codebook_id, version, run_id),
            )
        ]

    def distinct_prompt_hashes(
        self, codebook_id: str, version: int, run_id: str
    ) -> list[tuple[str, int, str, str]]:
        return [
            (r[0], int(r[1]), r[2], r[3])
            for r in self.conn.execute(
                """SELECT prompt_hash, COUNT(*), MIN(updated_at), MAX(updated_at) FROM jobs
                   WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'
                   GROUP BY prompt_hash ORDER BY 2 DESC""",
                (codebook_id, version, run_id),
            )
        ]

    def codebook_runs(self) -> list[tuple[str, int, str]]:
        return [
            (r[0], int(r[1]), r[2])
            for r in self.conn.execute(
                """SELECT DISTINCT codebook_id, codebook_version, run_id FROM jobs
                   ORDER BY codebook_id, codebook_version, run_id"""
            )
        ]
