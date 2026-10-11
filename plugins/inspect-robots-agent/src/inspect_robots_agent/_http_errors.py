"""Concise HTTP error diagnostics shared by OpenAI-compatible wire clients."""

from __future__ import annotations

import re
from typing import Any

import httpx

_NONRETRYABLE_REASONS = {
    "credit_balance_exhausted": "credit balance exhausted",
    "organization_spend_limit_exceeded": "organization spend limit reached",
    "project_spend_limit_exceeded": "project spend limit reached",
    "organization_usage_limit_exceeded": "organization usage limit reached",
    "insufficient_quota": "API quota unavailable; check credits or spending/usage limits",
}
_RATE_WINDOW = re.compile(r"\b(requests|tokens) per (minute|day)\b", re.IGNORECASE)
_RATE_COUNTS = re.compile(
    r"\bLimit\s*:?\s*([0-9]{1,16}(?:\.[0-9]{1,16})?)\s*,\s*"
    r"Used\s*:?\s*([0-9]{1,16}(?:\.[0-9]{1,16})?)\b",
    re.IGNORECASE,
)


def _error_fields(response: httpx.Response) -> dict[str, Any]:
    """Return structured provider errors without assuming JSON or its shape."""
    try:
        payload = response.json()
    except ValueError:
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def nonretryable_reason(response: httpx.Response) -> str | None:
    """Identify explicit billing or usage limits that waiting cannot resolve."""
    error = _error_fields(response)
    for key in ("code", "type"):
        value = error.get(key)
        if isinstance(value, str) and value in _NONRETRYABLE_REASONS:
            return _NONRETRYABLE_REASONS[value]
    return None


def _duration(delay: float) -> str:
    """Render hours, minutes, and seconds without changing the actual sleep."""
    hours, remainder = divmod(delay, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours:g} h")
    if minutes:
        parts.append(f"{minutes:g} min")
    if seconds or not parts:
        parts.append(f"{seconds:g} s")
    return " ".join(parts)


def retry_notice(
    response: httpx.Response,
    *,
    model: str,
    delay: float,
    next_attempt: int,
    max_attempts: int,
) -> str:
    """Describe one scheduled retry without promising a quota reset.

    Only known codes/types identify throttling. Metrics are extracted from
    recognized provider wording; absent or unknown details remain unspecified.
    Raw error text and organization identifiers never enter the notice.
    """
    error = _error_fields(response)
    model_label = " ".join(model.split())[:120]
    code = error.get("code")
    if code == "slow_down":
        label = f"Request rate increased too quickly: {model_label}"
    elif code == "rate_limit_exceeded" or error.get("type") == "rate_limit_error":
        label = f"Rate limit reached: {model_label}"
        message = error.get("message")
        if isinstance(message, str):
            window = _RATE_WINDOW.search(message[:4096])
            counts = _RATE_COUNTS.search(message[:4096])
            if window is not None:
                label += f", {window.group(0).lower()}"
                if counts is not None:
                    limit, used = counts.groups()
                    label += f" ({used}/{limit})"
    else:
        label = f"HTTP {response.status_code}: request rejected for {model_label}"
    status = "" if label.startswith("HTTP ") else f" (HTTP {response.status_code})"
    return (
        f"[llm] {label}. Retry in {_duration(delay)}; "
        f"attempt {next_attempt}/{max_attempts}{status}."
    )
