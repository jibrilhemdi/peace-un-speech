#!/usr/bin/env python3
"""Resumable annotation worker. One request per (speech, codebook).

    python scripts/annotate/run_annotation.py --codebook violence --dry-run
    python scripts/annotate/run_annotation.py --codebook violence --pilot 40
    python scripts/annotate/run_annotation.py --codebook violence            # run daily until done
    python scripts/annotate/run_annotation.py --codebook violence --limit 50

Resume is exact: all of a speech's job rows are written in one transaction, so a speech is
either fully annotated or fully pending. Ctrl-C at any moment leaves a consistent database.

Failed requests consume the daily quota, so nothing here retries blindly. A transient 429 is
backed off and retried; daily-quota exhaustion stops the run cleanly with exit 0 and the UTC
resume time, so a cron wrapper does not alarm.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import codebook as cbmod  # noqa: E402
from lib.client import (  # noqa: E402
    DailyQuotaExhausted, FatalProviderError, OpenRouterClient, Outcome, Pacer, backoff_delay,
)
from lib.codebook import Codebook, ResponseError  # noqa: E402
from lib.config import Config  # noqa: E402
from lib.db import Store, next_reset, quota_day  # noqa: E402


# --------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------
def build_messages(cb: Codebook, prefix: str, paragraphs: list[str]) -> tuple[str, str, str, str]:
    """Returns (system, user, request_hash, prompt_hash).

    The static codebook block leads and the speech trails, so the prefix is byte-identical
    across every request for this codebook version and prompt-prefix caching applies.
    """
    speech = cbmod.render_speech(paragraphs)
    user = f"{prefix}\n\n{speech}"
    return (
        cbmod.SYSTEM_PROMPT,
        user,
        cbmod.request_hash(prefix, speech),
        cbmod.prefix_hash(cb),
    )


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def stratified_speech_sample(store: Store, n: int, strata: int, seed: int) -> list[str]:
    """N speeches stratified by (bloc x length tercile), allocated proportionally.

    Speeches with no country (UN briefers) form their own stratum rather than being dropped,
    so a pilot still exercises them if they are in the corpus.
    """
    rows = list(
        store.conn.execute(
            """SELECT speech_id,
                      COALESCE(MAX(bloc), 'NO_BLOC') AS bloc,
                      COUNT(*) AS n_units
               FROM units WHERE para_is_procedural = 0
               GROUP BY speech_id"""
        )
    )
    if not rows:
        return []
    lengths = sorted(r["n_units"] for r in rows)
    cuts = [lengths[int(len(lengths) * (i + 1) / strata) - 1] for i in range(strata)]

    def tercile(v: int) -> int:
        return next((i for i, c in enumerate(cuts) if v <= c), strata - 1)

    buckets: dict[tuple[str, int], list[str]] = {}
    for r in rows:
        buckets.setdefault((r["bloc"], tercile(r["n_units"])), []).append(r["speech_id"])

    rng = random.Random(seed)
    total = sum(len(v) for v in buckets.values())
    picked: list[str] = []
    # Largest-remainder allocation, so small strata are not rounded out of existence.
    quotas = []
    for key, ids in sorted(buckets.items()):
        exact = n * len(ids) / total
        quotas.append([key, ids, int(exact), exact - int(exact)])
    left = n - sum(q[2] for q in quotas)
    for q in sorted(quotas, key=lambda q: -q[3])[: max(0, left)]:
        q[2] += 1
    for key, ids, k, _ in quotas:
        pool = sorted(ids)
        rng.shuffle(pool)
        picked.extend(pool[: min(k, len(pool))])
    rng.shuffle(picked)
    return picked[:n]


# --------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------
def dry_run(cb: Codebook, store: Store, prefix: str, speech_ids: Sequence[str],
            cfg: Config, run_id: str) -> int:
    cpt = float(cfg.get("limits.chars_per_token", 3.7))
    budget = int(cfg.get("limits.daily_budget", 950))
    spacing = float(cfg.get("limits.min_request_spacing_s", 3.2))
    max_out = int(cfg.get("provider.max_output_tokens", 8192))

    n_units = 0
    prompt_chars = 0
    longest = 0
    for sid in speech_ids:
        units = store.speech_units(sid)
        if not units:
            continue
        paras = [u["text"] for u in units]
        _, user, _, _ = build_messages(cb, prefix, paras)
        prompt_chars += len(user) + len(cbmod.SYSTEM_PROMPT)
        longest = max(longest, len(user))
        n_units += len(units)

    n_req = len(speech_ids)
    est_prompt_tokens = prompt_chars / cpt
    prefix_tokens = len(prefix) / cpt
    days = -(-n_req // budget) if budget else 0
    reset = next_reset(int(cfg.get("limits.quota_reset_hour_utc", 0)))

    print()
    print("DRY RUN — no API call was made")
    print("-" * 74)
    print(f"codebook              {cb.id} v{cb.version}   run_id={run_id}")
    print(f"model                 {', '.join(cfg.models())}")
    print(f"requests to make      {n_req:,}   (one per speech)")
    print(f"paragraphs covered    {n_units:,}")
    print(f"per-request paragraph mean {n_units / max(1, n_req):.1f}")
    print()
    print(f"prompt tokens (est)   {est_prompt_tokens:,.0f} total, "
          f"{est_prompt_tokens / max(1, n_req):,.0f} per request")
    print(f"  of which static     {prefix_tokens:,.0f} per request "
          f"({prefix_tokens / max(1, est_prompt_tokens / max(1, n_req)) * 100:.0f}% — "
          f"cacheable prefix)")
    print(f"longest single prompt {longest / cpt:,.0f} tokens")
    print(f"max output allowed    {max_out:,} tokens/request")
    print(f"  Estimated at {cpt} chars/token. The served tokeniser is not available locally;")
    print("  expect +/-10% on English prose.")
    print()
    print(f"daily budget          {budget:,} requests/day "
          f"(provider cap 1,000; failures count too)")
    print(f"days to completion    {days} day(s) at the current cap")
    print(f"wall clock per day    ~{budget * spacing / 3600:.1f} h at {spacing}s spacing")
    print(f"next quota reset      {reset.isoformat()}")
    print("-" * 74)
    return 0


# --------------------------------------------------------------------------------------
# Pilot CSV
# --------------------------------------------------------------------------------------
def write_pilot_csv(cb: Codebook, store: Store, run_id: str, speech_ids: Sequence[str],
                    out_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pilot_{cb.id}_{stamp}.csv"
    dims = list(cb.dimension_names)
    header = (
        ["unit_id", "speech_id", "para_index", "country", "n_words", "text",
         "annotation_index"]
        + dims
        + ["evidence", "evidence_verified", "confidence",
           # blank columns for manual adjudication
           "adj_agree", "adj_correct_codes", "adj_notes"]
    )
    placeholders = ",".join("?" * len(speech_ids)) or "NULL"
    rows = list(
        store.conn.execute(
            f"""SELECT u.unit_id, u.speech_id, u.para_index, u.country, u.n_words, u.text
                FROM units u
                WHERE u.para_is_procedural = 0 AND u.speech_id IN ({placeholders})
                ORDER BY u.speech_id, u.para_index""",
            list(speech_ids),
        )
    )
    ann_by_unit: dict[str, dict[int, dict[str, Any]]] = {}
    for a in store.conn.execute(
        f"""SELECT a.unit_id, a.annotation_index, a.dimension, a.value, a.evidence,
                   a.confidence, a.evidence_verified
            FROM annotations a JOIN units u ON u.unit_id = a.unit_id
            WHERE a.codebook_id=? AND a.codebook_version=? AND a.run_id=?
              AND u.speech_id IN ({placeholders})""",
        [cb.id, cb.version, run_id, *speech_ids],
    ):
        slot = ann_by_unit.setdefault(a["unit_id"], {}).setdefault(
            a["annotation_index"],
            {"evidence": a["evidence"], "confidence": a["confidence"],
             "evidence_verified": a["evidence_verified"]},
        )
        slot[a["dimension"]] = a["value"]

    # utf-8-sig, not utf-8: the corpus is full of curly quotes, é and Türkiye, and Excel
    # (especially on macOS) misreads a BOM-less UTF-8 CSV as Latin-1 and renders them as
    # mojibake. The BOM is invisible to pandas, which reads utf-8-sig transparently.
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            anns = ann_by_unit.get(r["unit_id"], {})
            if not anns:
                w.writerow(
                    [r["unit_id"], r["speech_id"], r["para_index"], r["country"],
                     r["n_words"], r["text"], ""]
                    + ["NONE"] * len(dims) + ["", "", "", "", "", ""]
                )
                continue
            for idx in sorted(anns):
                a = anns[idx]
                w.writerow(
                    [r["unit_id"], r["speech_id"], r["para_index"], r["country"],
                     r["n_words"], r["text"], idx]
                    + [a.get(d, "NONE") for d in dims]
                    + [a.get("evidence", ""),
                       "" if a.get("evidence_verified") is None
                       else ("yes" if a["evidence_verified"] else "NO"),
                       a.get("confidence", ""), "", "", ""]
                )
    return path


# --------------------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------------------
class Worker:
    def __init__(self, cfg: Config, cb: Codebook, store: Store, client: OpenRouterClient,
                 run_id: str, model_chain: list[str]):
        self.cfg = cfg
        self.cb = cb
        self.store = store
        self.client = client
        self.run_id = run_id
        self.models = model_chain
        self.model_idx = 0
        self.prefix = cbmod.render_prefix(cb)
        self.prompt_hash = cbmod.prefix_hash(cb)
        self.pacer = Pacer(float(cfg.get("limits.min_request_spacing_s", 3.2)))
        self.reset_hour = int(cfg.get("limits.quota_reset_hour_utc", 0))
        self.daily_budget = int(cfg.get("limits.daily_budget", 950))
        self.max_attempts = int(cfg.get("limits.max_attempts", 3))
        self.backoff_base = float(cfg.get("limits.backoff_base_s", 4.0))
        self.backoff_cap = float(cfg.get("limits.backoff_max_s", 120.0))
        self.backoff_jitter = float(cfg.get("limits.backoff_jitter", 0.3))
        self.temperature = float(cfg.get("provider.temperature", 0.0))
        self.seed = cfg.get("provider.seed")
        self.max_tokens = int(cfg.get("provider.max_output_tokens", 8192))
        mode = str(cfg.get("provider.json_mode", "auto")).lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"provider.json_mode must be auto|on|off, got {mode!r}")
        self.json_mode = mode != "off"
        self.reasoning = cfg.get("provider.reasoning")
        self.stats = {"ok": 0, "parse_error": 0, "api_error": 0, "requests": 0}
        self.progress_prefix = ""

    def notice(self, msg: str) -> None:
        """Emit a retry/fallback message without mangling the in-progress status line."""
        print(f"\n    {msg}", flush=True)
        if self.progress_prefix:
            print(self.progress_prefix, end="", flush=True)

    # -- quota --------------------------------------------------------------------------
    def _check_budget(self) -> None:
        day = quota_day(self.reset_hour)
        used = self.store.quota_used(day)
        if used >= self.daily_budget:
            raise DailyQuotaExhausted(
                f"local daily budget reached: {used}/{self.daily_budget} requests on {day}"
            )

    def _spend(self) -> int:
        used = self.store.record_attempt(quota_day(self.reset_hour))
        self.stats["requests"] += 1
        return used

    # -- one speech ---------------------------------------------------------------------
    def process_speech(self, speech_id: str) -> str:
        units = self.store.speech_units(speech_id)
        if not units:
            return "skipped"
        paragraphs = [u["text"] for u in units]
        unit_ids = [u["unit_id"] for u in units]
        system, user, req_hash, _ = build_messages(self.cb, self.prefix, paragraphs)

        attempts = 0
        repair_hint: str | None = None
        last_error = ""
        last_status = "api_error"

        while attempts < self.max_attempts:
            self._check_budget()
            self.pacer.wait()
            model = self.models[self.model_idx]
            content = user if repair_hint is None else (
                f"{user}\n\n# CORRECTION\nYour previous reply was rejected: {repair_hint}\n"
                "Return the corrected JSON object only."
            )

            attempts += 1
            self._spend()  # committed before the call; the provider counts it either way
            reply = self.client.complete(
                model=model, system=system, user=content,
                temperature=self.temperature, seed=self.seed, max_tokens=self.max_tokens,
                json_mode=self.json_mode and self.store.supports_json_mode(model),
                reasoning=self.reasoning,
            )

            if reply.outcome is Outcome.OK:
                try:
                    payload = json.loads(cbmod.strip_fences(reply.text))
                except json.JSONDecodeError as exc:
                    last_error = f"JSON decode failed: {exc}"
                    last_status = "parse_error"
                    if repair_hint is None:
                        repair_hint = f"the reply was not valid JSON ({exc})"
                        continue
                    break
                try:
                    anns = cbmod.validate_response(self.cb, payload, paragraphs)
                except ResponseError as exc:
                    last_error = f"validation failed: {exc}"
                    last_status = "parse_error"
                    if repair_hint is None:
                        repair_hint = str(exc)
                        continue
                    break

                by_unit: dict[str, list] = {uid: [] for uid in unit_ids}
                for a in anns:
                    by_unit[unit_ids[a.unit_index - 1]].append(a)
                self.store.write_speech_result(
                    speech_id=speech_id, codebook_id=self.cb.id, version=self.cb.version,
                    run_id=self.run_id, unit_ids=unit_ids, status="ok",
                    request_hash=req_hash, prompt_hash=self.prompt_hash,
                    model=reply.model or model, response_json=reply.text,
                    error=None, attempts=attempts,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    annotations_by_unit=by_unit,
                )
                self.stats["ok"] += 1
                return "ok"

            if reply.outcome is Outcome.DAILY_QUOTA:
                raise DailyQuotaExhausted(reply.error or "provider reported daily quota exhausted")

            if reply.outcome is Outcome.AUTH:
                raise FatalProviderError(
                    f"authentication failed ({reply.status_code}): {reply.error}\n"
                    "Check OPENROUTER_API_KEY."
                )

            if reply.outcome is Outcome.JSON_MODE_UNSUPPORTED:
                # The model does not accept response_format. Record it so no future run pays
                # this request again, then retry immediately without it. The prompt already
                # demands bare JSON, so nothing about the task changes.
                self.store.set_json_mode_support(model, False, reply.error)
                self.notice(
                    f"'{model}' does not support JSON mode — disabling it for this model "
                    "and retrying (recorded, so this costs one request ever)"
                )
                attempts -= 1  # a capability probe is not an attempt at the job
                continue

            if reply.outcome is Outcome.BAD_REQUEST:
                last_error = f"bad request ({reply.status_code}): {reply.error}"
                last_status = "api_error"
                break  # our payload is wrong; another attempt just burns quota

            if reply.outcome is Outcome.MODEL_GONE:
                last_error = f"model unavailable: {reply.error}"
                if not self._next_model(f"404 on {model}"):
                    raise FatalProviderError(
                        f"model '{model}' returned 404 and no fallback is configured.\n"
                        f"{reply.error}\n"
                        "Set provider.fallback_models in config.yaml, or pass --model."
                    )
                attempts -= 1  # a model swap is not a retry of the same request
                continue

            if reply.outcome is Outcome.RATE_LIMIT:
                delay = reply.retry_after_s
                if delay is None:
                    delay = backoff_delay(attempts, self.backoff_base, self.backoff_cap,
                                          self.backoff_jitter)
                    src = "backoff"
                else:
                    src = "Retry-After"
                self.notice(f"429 — sleeping {delay:.1f}s ({src})")
                time.sleep(delay)
                last_error = f"rate limited: {reply.error}"
                last_status = "api_error"
                continue

            # SERVER or NETWORK
            last_error = f"{reply.outcome.value}: {reply.error}"
            last_status = "api_error"
            if attempts >= self.max_attempts and self._next_model(reply.outcome.value):
                attempts = 0
                continue
            if attempts < self.max_attempts:
                delay = backoff_delay(attempts, self.backoff_base, self.backoff_cap,
                                      self.backoff_jitter)
                self.notice(f"{reply.outcome.value} — sleeping {delay:.1f}s")
                time.sleep(delay)

        self.store.write_speech_result(
            speech_id=speech_id, codebook_id=self.cb.id, version=self.cb.version,
            run_id=self.run_id, unit_ids=unit_ids, status=last_status,
            request_hash=req_hash, prompt_hash=self.prompt_hash,
            model=self.models[self.model_idx], response_json=None,
            error=last_error[:2000], attempts=attempts,
            prompt_tokens=None, completion_tokens=None, annotations_by_unit=None,
        )
        self.stats[last_status] = self.stats.get(last_status, 0) + 1
        return last_status

    def _next_model(self, why: str) -> bool:
        if self.model_idx + 1 >= len(self.models):
            return False
        self.model_idx += 1
        self.notice(f"falling back to '{self.models[self.model_idx]}' ({why})")
        return True


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codebook", required=True, help="codebook id (filename stem in codebooks/)")
    ap.add_argument("--limit", type=int, default=None, help="cap requests this session")
    ap.add_argument("--daily-budget", type=int, default=None, help="override config daily budget")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts, report exact request count and token estimate, call nothing")
    ap.add_argument("--pilot", type=int, default=None, metavar="N",
                    help="sample N speeches stratified by bloc and length, run them, write a CSV")
    ap.add_argument("--model", default=None, help="override provider.model")
    ap.add_argument("--force", action="store_true",
                    help="run the full corpus without a completed pilot")
    ap.add_argument("--retry-errors", action="store_true",
                    help="also re-attempt speeches previously marked parse_error / api_error")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.daily_budget is not None:
        cfg.raw.setdefault("limits", {})["daily_budget"] = args.daily_budget

    try:
        cb = cbmod.require_codebook(cfg.path_for("codebooks"), args.codebook)
    except cbmod.CodebookError as exc:
        print(exc, file=sys.stderr)
        return 1

    db_path = cfg.path_for("db")
    if not db_path.exists() and not cfg.path_for("units").exists():
        print("no units found — run: python scripts/annotate/segment_units.py", file=sys.stderr)
        return 1

    store = Store(db_path)
    if store.units_count() == 0:
        print("units table is empty — run: python scripts/annotate/segment_units.py", file=sys.stderr)
        store.close()
        return 1

    run_id = args.run_id
    prefix = cbmod.render_prefix(cb)

    # -- select work --------------------------------------------------------------------
    is_pilot = args.pilot is not None
    pilot_run_id = f"pilot-{run_id}" if is_pilot else run_id
    if is_pilot:
        run_id = pilot_run_id
        seed = int(cfg.get("provider.seed") or 0)
        candidates = stratified_speech_sample(store, args.pilot, int(cfg.get("pilot.length_strata", 3)), seed)
        pending = store.pending_speeches(cb.id, cb.version, run_id, speech_ids=candidates)
        if args.retry_errors:
            pending = [s for s in candidates if s in set(pending)]
        print(f"pilot: {len(candidates)} speeches sampled (stratified by bloc x length tercile), "
              f"{len(pending)} still pending")
        selected = pending
    else:
        selected = store.pending_speeches(cb.id, cb.version, run_id)
        if not args.retry_errors:
            # A speech with a recorded parse_error/api_error is still "pending" by unit count.
            # Skip it unless asked, so a systematically bad speech does not eat quota daily.
            errored = {
                r[0] for r in store.conn.execute(
                    """SELECT DISTINCT speech_id FROM jobs
                       WHERE codebook_id=? AND codebook_version=? AND run_id=?
                         AND status IN ('parse_error','api_error')""",
                    (cb.id, cb.version, run_id),
                )
            }
            if errored:
                before = len(selected)
                selected = [s for s in selected if s not in errored]
                print(f"skipping {before - len(selected)} previously errored speech(es); "
                      "re-attempt with --retry-errors")

    if args.limit is not None:
        selected = selected[: args.limit]

    if args.dry_run:
        rc = dry_run(cb, store, prefix, selected, cfg, run_id)
        store.close()
        return rc

    if not selected:
        print(f"nothing pending for {cb.id} v{cb.version} (run_id={run_id}) — already complete")
        store.close()
        return 0

    # -- pilot gate ---------------------------------------------------------------------
    if not is_pilot and cfg.get("pilot.require_pilot_before_full_run", True) and not args.force:
        pilot = store.pilot_for(cb.id, cb.version)
        min_n = int(cfg.get("pilot.min_pilot_speeches", 20))
        if pilot is None or int(pilot["n_speeches"]) < min_n:
            have = 0 if pilot is None else int(pilot["n_speeches"])
            print(
                f"\nRefusing a full corpus pass for {cb.id} v{cb.version}: no completed pilot "
                f"(have {have} speeches, need {min_n}).\n"
                f"\n  python scripts/annotate/run_annotation.py --codebook {cb.id} --pilot 40\n"
                "\nThen adjudicate the pilot CSV, write negative_examples, and bump `version`.\n"
                f"A bad prompt costs {len(selected)} requests "
                f"(~{-(-len(selected) // int(cfg.get('limits.daily_budget', 950)))} day(s) of "
                "quota); a pilot costs under an hour.\n"
                "Pass --force to override.",
                file=sys.stderr,
            )
            store.close()
            return 1

    # -- go -----------------------------------------------------------------------------
    try:
        api_key = Config.api_key()
    except Exception as exc:
        print(exc, file=sys.stderr)
        store.close()
        return 1

    models = cfg.models(args.model)
    reset_hour = int(cfg.get("limits.quota_reset_hour_utc", 0))
    day = quota_day(reset_hour)
    budget = int(cfg.get("limits.daily_budget", 950))
    used = store.quota_used(day)

    print(f"codebook   {cb.id} v{cb.version}   run_id={run_id}")
    print(f"model      {models[0]}" + (f"  (fallbacks: {', '.join(models[1:])})" if len(models) > 1 else ""))
    print(f"prompt     {cbmod.prefix_hash(cb)}")
    print(f"quota      {used}/{budget} used on {day} (UTC day, reset hour {reset_hour:02d}:00)")
    print(f"speeches   {len(selected)} to process this session")
    print()

    client = OpenRouterClient(
        api_key=api_key,
        base_url=str(cfg.get("provider.base_url")),
        timeout_s=float(cfg.get("provider.request_timeout_s", 180)),
        http_referer=cfg.get("provider.http_referer"),
        x_title=cfg.get("provider.x_title"),
        daily_quota_markers=cfg.get("limits.daily_quota_markers", []),
        daily_quota_retry_after_threshold_s=float(
            cfg.get("limits.daily_quota_retry_after_threshold_s", 600)
        ),
    )
    worker = Worker(cfg, cb, store, client, run_id, models)

    started = time.monotonic()
    stopped_early = ""
    exit_code = 0
    try:
        for i, sid in enumerate(selected, 1):
            n_units = len(store.speech_units(sid))
            worker.progress_prefix = f"[{i}/{len(selected)}] {sid}  ({n_units} paras) ... "
            print(worker.progress_prefix, end="", flush=True)
            status = worker.process_speech(sid)
            worker.progress_prefix = ""
            print(status, flush=True)
    except DailyQuotaExhausted as exc:
        reset = next_reset(reset_hour)
        wait_h = (reset - datetime.now(timezone.utc)).total_seconds() / 3600
        stopped_early = "daily quota"
        print()
        print("-" * 74)
        print(f"Daily quota reached: {exc}")
        print(f"Resume after {reset.isoformat()} (in {wait_h:.1f} h).")
        print(f"  python scripts/annotate/run_annotation.py --codebook {cb.id}"
              + (f" --run-id {run_id}" if run_id != "main" else ""))
        print("-" * 74)
        exit_code = 0  # a cron wrapper should not alarm on a normal day boundary
    except KeyboardInterrupt:
        stopped_early = "interrupted"
        print("\n\nInterrupted. The database is consistent — every completed speech is committed.")
        exit_code = 130
    except FatalProviderError as exc:
        stopped_early = "fatal"
        print(f"\n\nFATAL: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        client.close()

    elapsed = time.monotonic() - started
    s = worker.stats
    done = store.status_counts(cb.id, cb.version, run_id)
    print()
    print(f"this session   ok={s['ok']}  parse_error={s['parse_error']}  "
          f"api_error={s['api_error']}  requests={s['requests']}  "
          f"elapsed={elapsed / 60:.1f} min")
    print(f"quota today    {store.quota_used(quota_day(reset_hour))}/{budget}")
    print(f"unit rows      " + "  ".join(f"{k}={v:,}" for k, v in sorted(done.items())))

    if is_pilot and not stopped_early:
        speeches_done = [
            r[0] for r in store.conn.execute(
                """SELECT DISTINCT speech_id FROM jobs
                   WHERE codebook_id=? AND codebook_version=? AND run_id=? AND status='ok'""",
                (cb.id, cb.version, run_id),
            )
        ]
        if speeches_done:
            out = write_pilot_csv(cb, store, run_id, speeches_done, cfg.path_for("pilots"))
            n_units = int(store.conn.execute(
                """SELECT COUNT(*) FROM jobs WHERE codebook_id=? AND codebook_version=?
                   AND run_id=? AND status='ok'""",
                (cb.id, cb.version, run_id)).fetchone()[0])
            store.record_pilot(
                codebook_id=cb.id, version=cb.version, run_id=run_id,
                n_speeches=len(speeches_done), n_units=n_units, csv_path=str(out),
                prompt_hash=cbmod.prefix_hash(cb), model=models[0],
            )
            print()
            print(f"pilot CSV      {out}")
            print("               One row per paragraph. Columns adj_agree / adj_correct_codes /")
            print("               adj_notes are empty for your adjudication.")
            print()
            print("Next: read the CSV, write negative_examples for anything over-firing,")
            print(f"      bump `version` in codebooks/{cb.id}.yaml, then run the full pass.")

    store.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
