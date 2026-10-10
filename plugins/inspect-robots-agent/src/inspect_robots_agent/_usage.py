"""Account for Responses usage without credentials or additional API requests."""

from __future__ import annotations

import math
from typing import Any

PRICING_DATE = "2026-10-10"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-astra"


def normalized_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    """Normalize valid Responses counters, retaining cache and reasoning details."""
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    result: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[name] = value
    for container, source, target in (
        ("input_tokens_details", "cached_tokens", "cache_read_input_tokens"),
        ("input_tokens_details", "cache_write_tokens", "cache_creation_input_tokens"),
        ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
    ):
        detail = raw.get(container)
        value = detail.get(source) if isinstance(detail, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[target] = value
    return result or None


class UsageLedger:
    """Retain per-attempt usage and an explicitly qualified Astra cost estimate.

    Unknown pricing or missing successful-response usage makes the cost unknown.
    Reasoning is already part of output_tokens. No account balance is inferred.
    """

    def __init__(self, model: str, max_cost_usd: float | None = None) -> None:
        self.model = model.removeprefix("openai/")
        if max_cost_usd is not None:
            if (
                isinstance(max_cost_usd, bool)
                or not math.isfinite(max_cost_usd)
                or max_cost_usd <= 0
            ):
                raise ValueError("max_cost_usd must be finite and positive")
            if self.model != "gpt-6-astra":
                raise ValueError("max_cost_usd requires known gpt-6-astra pricing")
        self.max_cost_usd = max_cost_usd
        self.reset()

    def reset(self) -> None:
        """Start a new trial without carrying another scene's spend into its budget."""
        self.totals: dict[str, int] = {}
        self.attempts = 0
        self.responses_with_usage = 0
        self.missing_usage = 0
        self.request_time_s = 0.0
        self._cost = 0.0
        self._cost_known = self.model == "gpt-6-astra"

    def record(self, status: int | None, payload: Any, duration_s: float) -> None:
        """Count every attempt, including paid incomplete or failed responses."""
        self.attempts += 1
        self.request_time_s += max(0.0, duration_s)
        usage = normalized_usage(payload) if isinstance(payload, dict) else None
        if usage is None:
            if status is None or status == 200 or status >= 500:
                self.missing_usage += 1
                self._cost_known = False
            return
        self.responses_with_usage += 1
        for name, value in usage.items():
            self.totals[name] = self.totals.get(name, 0) + value
        price = self._price(payload, usage)
        if price is None:
            self._cost_known = False
        else:
            self._cost += price

    def _price(self, payload: dict[str, Any], usage: dict[str, int]) -> float | None:
        model = str(payload.get("model", self.model)).removeprefix("openai/")
        if model != "gpt-6-astra" or not {"input_tokens", "output_tokens"} <= usage.keys():
            return None
        inputs, outputs = usage["input_tokens"], usage["output_tokens"]
        reads = usage.get("cache_read_input_tokens", 0)
        writes = usage.get("cache_creation_input_tokens", 0)
        if reads + writes > inputs or usage.get("reasoning_tokens", 0) > outputs:
            return None
        tier = payload.get("service_tier", "default")
        if not isinstance(tier, str):
            return None
        multiplier = {"default": 1.0, "standard": 1.0, "flex": 0.5, "priority": 2.0}.get(tier)
        if multiplier is None:
            return None
        input_factor = 2.0 if inputs > 272_000 else 1.0
        output_factor = 1.5 if inputs > 272_000 else 1.0
        return (
            multiplier
            * (
                input_factor * (10 * (inputs - reads - writes) + reads + 12.5 * writes)
                + output_factor * 50 * outputs
            )
            / 1_000_000
        )

    def check_budget(self) -> None:
        """Stop before the next request; one completed request can exceed the budget."""
        if self.max_cost_usd is None:
            return
        if not self._cost_known:
            raise RuntimeError("LLM cost budget cannot be enforced: usage or pricing is unknown")
        if self._cost >= self.max_cost_usd:
            raise RuntimeError(
                f"LLM cost budget reached (${self._cost:.6f} >= ${self.max_cost_usd:.6f}); "
                "no additional request sent"
            )

    def report(self) -> dict[str, Any]:
        """Return serializable counters and price provenance, never credentials."""
        return {
            **self.totals,
            "http_attempts": self.attempts,
            "responses_with_usage": self.responses_with_usage,
            "responses_missing_usage": self.missing_usage,
            "usage_complete": self.missing_usage == 0,
            "request_time_s": self.request_time_s,
            "estimated_cost_usd": self._cost if self._cost_known else None,
            "pricing_date": PRICING_DATE,
            "pricing_source": PRICING_SOURCE,
            "max_cost_usd": self.max_cost_usd,
            "budget_scope": "per trial; checked between requests; one request may overshoot",
        }
