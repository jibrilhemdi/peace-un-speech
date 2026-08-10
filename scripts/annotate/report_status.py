#!/usr/bin/env python3
"""Progress, quota, drift and per-code prevalence.

    python scripts/annotate/report_status.py
    python scripts/annotate/report_status.py --codebook violence
    python scripts/annotate/report_status.py --run-id pilot-main

Prevalence is printed among completed units, early and often. A code firing on more than
~50% of paragraphs is almost never a finding — it is a prompt that needs fixing, and seeing
it on day one saves a week of quota.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import codebook as cbmod  # noqa: E402
from lib.config import Config, model_tag  # noqa: E402
from lib.db import Store, next_reset, quota_day  # noqa: E402

HIGH_PREVALENCE = 0.50
LOW_PREVALENCE = 0.01


def bar(frac: float, width: int = 30) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "·" * (width - n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codebook", help="restrict to one codebook id")
    ap.add_argument("--run-id", default=None, help="restrict to one run_id")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    db_path = cfg.path_for("db")
    if not db_path.exists():
        print(f"no database at {db_path} — run: python scripts/annotate/segment_units.py", file=sys.stderr)
        return 1

    store = Store(db_path)
    reset_hour = int(cfg.get("limits.quota_reset_hour_utc", 0))
    budget = int(cfg.get("limits.daily_budget", 950))
    day = quota_day(reset_hour)
    used = store.quota_used(day)
    remaining = max(0, budget - used)

    total_units = store.units_count()
    if total_units == 0:
        print("units table is empty — run: python scripts/annotate/segment_units.py", file=sys.stderr)
        store.close()
        return 1

    n_substantive = int(store.conn.execute(
        "SELECT COUNT(*) FROM units WHERE para_is_procedural = 0").fetchone()[0])
    n_speeches = int(store.conn.execute(
        "SELECT COUNT(DISTINCT speech_id) FROM units WHERE para_is_procedural = 0"
    ).fetchone()[0])

    print("=" * 78)
    print("CORPUS")
    print(f"  units            {total_units:,} total, {n_substantive:,} non-procedural")
    print(f"  speeches         {n_speeches:,}")
    print()
    print("QUOTA")
    print(f"  today ({day})  {used}/{budget} used, {remaining} remaining")
    print(f"  resets           {next_reset(reset_hour).isoformat()}")

    cb_dir = cfg.path_for("codebooks")
    runs = store.codebook_runs()
    if args.codebook:
        runs = [r for r in runs if r[0] == args.codebook]
    if args.run_id:
        runs = [r for r in runs if r[2] == args.run_id]

    known = sorted(p.stem for p in cb_dir.glob("*.yaml"))
    started = {r[0] for r in runs}
    unstarted = [c for c in known if c not in started and (not args.codebook or c == args.codebook)]

    warnings: list[str] = []

    for cb_id, version, run_id in runs:
        print()
        print("=" * 78)
        print(f"{cb_id} v{version}   run_id={run_id}")

        try:
            cb = cbmod.require_codebook(cb_dir, cb_id)
            current_version = cb.version
            current_hash = cbmod.prefix_hash(cb)
        except cbmod.CodebookError:
            cb, current_version, current_hash = None, None, None

        counts = store.status_counts(cb_id, version, run_id)
        ok = counts.get("ok", 0)
        errored = counts.get("parse_error", 0) + counts.get("api_error", 0)

        # denominator: substantive units only
        pending = max(0, n_substantive - ok - errored)
        pct = ok / n_substantive if n_substantive else 0
        print(f"  units            {ok:,} done   {pending:,} pending   {errored:,} errored"
              f"   of {n_substantive:,}")
        print(f"                   {bar(pct)} {pct * 100:.1f}%")
        for k in ("parse_error", "api_error", "skipped"):
            if counts.get(k):
                print(f"    {k:<14} {counts[k]:,}")

        speeches_done = int(store.conn.execute(
            """SELECT COUNT(DISTINCT speech_id) FROM jobs
               WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'""",
            (cb_id, version, run_id)).fetchone()[0])
        speeches_left = max(0, n_speeches - speeches_done)
        print(f"  speeches         {speeches_done:,} done, {speeches_left:,} to go "
              f"(= requests remaining)")
        if speeches_left:
            days = -(-speeches_left // budget) if budget else 0
            today_possible = min(speeches_left, remaining)
            print(f"  est. completion  {days} day(s) at {budget}/day; "
                  f"{today_possible} more possible today")

        if current_version is not None and current_version != version:
            warnings.append(
                f"{cb_id}: DB holds v{version} but codebooks/{cb_id}.yaml is now v{current_version}. "
                f"Those are separate job sets — v{current_version} starts from zero."
            )

        # -- drift ----------------------------------------------------------------------
        models = store.distinct_models(cb_id, version, run_id)
        hashes = store.distinct_prompt_hashes(cb_id, version, run_id)
        if models:
            if len(models) > 1:
                span = store.conn.execute(
                    """SELECT model, MIN(updated_at), MAX(updated_at) FROM jobs
                       WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'
                       GROUP BY model ORDER BY MIN(updated_at)""",
                    (cb_id, version, run_id)).fetchall()
                detail = "; ".join(f"{model_tag(m)} {lo[:10]}..{hi[:10]}" for m, lo, hi in span)
                warnings.append(
                    f"{cb_id} v{version} run={run_id}: MIXED MODELS across completed units — "
                    + ", ".join(f"{model_tag(m)} ({n:,})" for m, n in models)
                    + f". Ran {detail}. Labels from different models are not directly "
                      "comparable. The model is recorded per unit and carried into every "
                      "export (annotations_long.model_tag, "
                      f"unit_code_matrix.{cb_id}__model_tag), so it can be controlled for — "
                      "but it cannot be removed after the fact."
                )
            print("  model(s)         " + ", ".join(f"{model_tag(m)} ({n:,})" for m, n in models))
        if hashes:
            if len(hashes) > 1:
                detail = "; ".join(f"{h or '?'} ({n:,}, {lo}..{hi})" for h, n, lo, hi in hashes)
                warnings.append(
                    f"{cb_id} v{version} run={run_id}: MIXED PROMPT HASHES across completed "
                    f"units — {detail}. The codebook changed mid-corpus without a version "
                    "bump, so early and late units were coded against different instruments."
                )
            print(f"  prompt hash(es)  " + ", ".join(f"{h or '?'} ({n:,})" for h, n, *_ in hashes))
            if current_hash and hashes[0][0] and hashes[0][0] != current_hash:
                warnings.append(
                    f"{cb_id} v{version}: the YAML now renders to prompt {current_hash} but "
                    f"completed units used {hashes[0][0]}. Bump `version` before continuing, "
                    "or the run will mix instruments."
                )

        unver = store.conn.execute(
            """SELECT SUM(n_evidence_unverified), SUM(n_annotations) FROM jobs
               WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'""",
            (cb_id, version, run_id)).fetchone()
        if unver and unver[1]:
            frac = (unver[0] or 0) / unver[1]
            print(f"  evidence spans   {unver[1]:,} total, {unver[0] or 0:,} unverified "
                  f"({frac * 100:.1f}%)")
            if frac > 0.15:
                warnings.append(
                    f"{cb_id} v{version}: {frac * 100:.0f}% of evidence spans could not be "
                    "matched back to their paragraph. The model is paraphrasing rather than "
                    "quoting, which makes the evidence column unusable for audit."
                )

        # -- prevalence -----------------------------------------------------------------
        if ok and cb is not None:
            print()
            print(f"  Per-code prevalence among {ok:,} completed units")
            for dim in cb.dimensions:
                print(f"    {dim.name}")
                rows = {
                    r[0]: int(r[1])
                    for r in store.conn.execute(
                        """SELECT a.value, COUNT(DISTINCT a.unit_id)
                           FROM annotations a
                           WHERE a.codebook_id=? AND a.codebook_version=? AND a.run_id=?
                             AND a.dimension=?
                           GROUP BY a.value""",
                        (cb_id, version, run_id, dim.name),
                    )
                }
                codes = list(dim.codes) + (["NONE"] if dim.allows_none() else [])
                for code in codes:
                    n = rows.get(code, 0)
                    frac = n / ok
                    flag = ""
                    if frac > HIGH_PREVALENCE:
                        flag = "  <-- OVER 50%, check the prompt"
                    elif n and frac < LOW_PREVALENCE:
                        flag = "  <-- very rare"
                    print(f"      {code:<10} {n:>7,}  {frac * 100:>5.1f}%  "
                          f"{bar(frac, 24)}{flag}")
                    if frac > HIGH_PREVALENCE:
                        warnings.append(
                            f"{cb_id} v{version}: {dim.name}={code} fires on {frac * 100:.0f}% "
                            f"of completed units. At that rate it carries almost no information "
                            f"and will flatten the MCA. Add negative_examples and bump `version` "
                            "before spending more quota."
                        )
            n_units_with_ann = int(store.conn.execute(
                """SELECT COUNT(DISTINCT unit_id) FROM annotations
                   WHERE codebook_id=? AND codebook_version=? AND run_id=?""",
                (cb_id, version, run_id)).fetchone()[0])
            empty = ok - n_units_with_ann
            print(f"    (no annotation at all: {empty:,} units, {empty / ok * 100:.1f}%)")
            if empty / ok < 0.20:
                warnings.append(
                    f"{cb_id} v{version}: only {empty / ok * 100:.0f}% of units received no "
                    "annotation. Most Council paragraphs are procedural or hortatory; a low "
                    "empty rate usually means the model is reaching."
                )

        pilot = store.pilot_for(cb_id, version)
        if pilot:
            print(f"  pilot            {pilot['n_speeches']} speeches, "
                  f"{pilot['completed_at']}, {pilot['csv_path']}")

    for c in unstarted:
        print()
        print("=" * 78)
        try:
            cb = cbmod.require_codebook(cb_dir, c)
            print(f"{c} v{cb.version}   not started")
            print(f"  python scripts/annotate/run_annotation.py --codebook {c} --pilot 40")
        except cbmod.CodebookError:
            print(f"{c}   not started, and fails validation — "
                  "run python scripts/codebook/validate.py")

    print()
    print("=" * 78)
    if warnings:
        print(f"WARNINGS ({len(warnings)})")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("No warnings.")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
