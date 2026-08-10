"""OpenRouter client: pacing, error classification, model fallback.

Failed requests consume daily quota, so nothing here retries blindly. Every failure mode is
classified before deciding whether another attempt is worth a request:

  RATE_LIMIT   the provider says slow down          -> back off, retry, still costs quota
  DAILY_QUOTA  the day's allowance is gone          -> stop the run cleanly, resume tomorrow
  SERVER       provider-side fault                  -> back off, retry, then next model
  MODEL_GONE   404 — withdrawn or renamed           -> next model immediately, no retry
  AUTH         bad or missing key                   -> fatal, retrying cannot help
  BAD_REQUEST  our payload is wrong                 -> fatal for this job, retrying cannot help
  JSON_MODE_UNSUPPORTED
               provider rejects response_format       -> drop it, retry, remember for this model
  NETWORK      timeout, connection reset            -> back off, retry
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx


# Providers word this differently; match the concept, not one vendor's phrasing.
_JSON_MODE_REJECTED = re.compile(
    r"structured[-_ ]?outputs?|response_format|json[-_ ]?object|json[-_ ]?schema|json[-_ ]?mode",
    re.I,
)


class Outcome(str, Enum):
    OK = "ok"
    RATE_LIMIT = "rate_limit"
    DAILY_QUOTA = "daily_quota"
    SERVER = "server"
    MODEL_GONE = "model_gone"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    JSON_MODE_UNSUPPORTED = "json_mode_unsupported"
    NETWORK = "network"


class DailyQuotaExhausted(RuntimeError):
    """Raised to unwind the run cleanly. Not an error condition — the day simply ended."""


class FatalProviderError(RuntimeError):
    """Retrying cannot help: bad key, malformed request, no usable model."""


@dataclass
class Reply:
    outcome: Outcome
    text: str = ""
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retry_after_s: float | None = None
    error: str = ""
    status_code: int | None = None


class Pacer:
    """Sleep-based spacing. Simpler to reason about than a token bucket at 20 req/min, and it
    cannot burst past the cap after an idle stretch."""

    def __init__(self, min_spacing_s: float):
        self.min_spacing = max(0.0, float(min_spacing_s))
        self._last = 0.0

    def wait(self) -> float:
        if self._last == 0.0:
            self._last = time.monotonic()
            return 0.0
        gap = time.monotonic() - self._last
        slept = 0.0
        if gap < self.min_spacing:
            slept = self.min_spacing - gap
            time.sleep(slept)
        self._last = time.monotonic()
        return slept


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _reset_header_seconds(headers: httpx.Headers) -> float | None:
    """OpenRouter sends X-RateLimit-Reset as epoch milliseconds."""
    raw = headers.get("x-ratelimit-reset")
    if not raw:
        return None
    try:
        ts = float(raw)
    except ValueError:
        return None
    if ts > 1e11:  # milliseconds
        ts /= 1000.0
    return max(0.0, ts - time.time())


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_s: float = 180.0,
        http_referer: str | None = None,
        x_title: str | None = None,
        daily_quota_markers: list[str] | None = None,
        daily_quota_retry_after_threshold_s: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._daily_markers = [m.lower() for m in (daily_quota_markers or [])]
        self._daily_threshold = float(daily_quota_retry_after_threshold_s)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
        self._client = httpx.Client(timeout=timeout_s, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        reasoning: dict[str, Any] | None = None,
    ) -> Reply:
        """One request. Never raises on an API error — classify and let the caller decide.

        ``json_mode`` sends response_format={"type":"json_object"}. Not every model behind
        OpenRouter supports it — inclusionai/ling-3.0-flash:free, served by Novita, returns
        400 "does not support feature: structured-outputs". The caller detects that once and
        turns it off for that model; the prompt already demands bare JSON, and strip_fences
        handles a stray fence, so nothing is lost.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                # Static codebook block first, speech last: the prefix is byte-identical
                # across every request for a codebook version, so prefix caching applies.
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if reasoning is not None:
            payload["reasoning"] = reasoning

        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            return Reply(Outcome.NETWORK, error=f"timeout: {exc}")
        except httpx.HTTPError as exc:
            return Reply(Outcome.NETWORK, error=f"network: {exc}")

        return self._classify(r, model)

    def _classify(self, r: httpx.Response, model: str) -> Reply:
        body = r.text or ""
        retry_after = _parse_retry_after(r.headers.get("retry-after"))

        if r.status_code == 200:
            try:
                data = r.json()
            except json.JSONDecodeError:
                return Reply(Outcome.SERVER, error=f"non-JSON 200 body: {body[:300]}",
                             status_code=200)
            # OpenRouter can return 200 with an error envelope.
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                code = err.get("code") if isinstance(err, dict) else None
                if code in (429, "429") or self._looks_daily(msg, retry_after, r.headers):
                    return Reply(
                        Outcome.DAILY_QUOTA if self._looks_daily(msg, retry_after, r.headers)
                        else Outcome.RATE_LIMIT,
                        error=msg, retry_after_s=retry_after, status_code=200,
                    )
                return Reply(Outcome.SERVER, error=f"error envelope: {msg[:300]}", status_code=200)

            try:
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                return Reply(Outcome.SERVER, error=f"unexpected payload: {body[:300]}",
                             status_code=200)

            finish = (choice.get("finish_reason") or "").lower()
            if finish == "length":
                return Reply(
                    Outcome.SERVER,
                    error="response truncated (finish_reason=length) — raise "
                          "provider.max_output_tokens in config.yaml",
                    status_code=200,
                )

            usage = data.get("usage") or {}
            return Reply(
                Outcome.OK,
                text=text,
                model=data.get("model") or model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                status_code=200,
            )

        msg = self._error_message(body)

        if r.status_code == 429:
            reset_in = _reset_header_seconds(r.headers)
            if self._looks_daily(msg, retry_after, r.headers, reset_in):
                return Reply(Outcome.DAILY_QUOTA, error=msg, retry_after_s=retry_after,
                             status_code=429)
            return Reply(Outcome.RATE_LIMIT, error=msg, retry_after_s=retry_after,
                         status_code=429)

        if r.status_code in (401, 403):
            return Reply(Outcome.AUTH, error=msg, status_code=r.status_code)
        if r.status_code == 404:
            return Reply(Outcome.MODEL_GONE, error=msg, status_code=404)
        if r.status_code == 402:
            # Out of credits reads as a day boundary for our purposes: stop, do not hammer.
            return Reply(Outcome.DAILY_QUOTA, error=msg, status_code=402)
        if r.status_code in (400, 422):
            if _JSON_MODE_REJECTED.search(msg):
                return Reply(Outcome.JSON_MODE_UNSUPPORTED, error=msg, status_code=r.status_code)
            return Reply(Outcome.BAD_REQUEST, error=msg, status_code=r.status_code)
        if 500 <= r.status_code < 600 or r.status_code == 408:
            return Reply(Outcome.SERVER, error=msg, status_code=r.status_code)
        return Reply(Outcome.SERVER, error=f"HTTP {r.status_code}: {msg}", status_code=r.status_code)

    def _looks_daily(
        self,
        message: str,
        retry_after: float | None,
        headers: httpx.Headers | None = None,
        reset_in: float | None = None,
    ) -> bool:
        low = (message or "").lower()
        if any(m in low for m in self._daily_markers):
            return True
        if retry_after is not None and retry_after > self._daily_threshold:
            return True
        if reset_in is None and headers is not None:
            reset_in = _reset_header_seconds(headers)
        return reset_in is not None and reset_in > self._daily_threshold

    @staticmethod
    def _error_message(body: str) -> str:
        """Flatten an error body to one line, keeping the provider's own text.

        OpenRouter wraps upstream failures as {"error": {"message": "Provider returned
        error", "metadata": {"raw": "...the real reason...", "provider_name": "..."}}}.
        Reporting only `message` throws away the only diagnostic that matters, so the
        metadata is folded in here.
        """
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return re.sub(r"\s+", " ", body)[:600]
        if not isinstance(data, dict):
            return str(data)[:600]

        err = data.get("error", data)
        if not isinstance(err, dict):
            return str(err)[:600]

        parts = [str(err.get("message") or "").strip()]
        meta = err.get("metadata")
        if isinstance(meta, dict):
            if meta.get("provider_name"):
                parts.append(f"[provider: {meta['provider_name']}]")
            raw = meta.get("raw")
            if raw:
                inner = raw
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        inner = parsed.get("message", raw) if isinstance(parsed, dict) else raw
                    except json.JSONDecodeError:
                        inner = raw
                parts.append(str(inner).strip())
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p))[:600]


def backoff_delay(attempt: int, base: float, cap: float, jitter: float) -> float:
    """Exponential with proportional jitter. attempt is 1-based."""
    raw = min(cap, base * (2 ** (attempt - 1)))
    return max(0.0, raw * (1.0 + random.uniform(-jitter, jitter)))
