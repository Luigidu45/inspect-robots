"""Usage survives model failures and spend guards prevent additional requests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inspect_robots_agent._llm import Provider
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent._usage import UsageLedger, normalized_usage


def payload(inputs: int = 100, outputs: int = 10) -> dict[str, Any]:
    """Return a representative paid Astra response, including explicit cache usage."""
    return {
        "model": "gpt-6-astra",
        "service_tier": "default",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "FINISH"}]}],
        "usage": {
            "input_tokens": inputs,
            "output_tokens": outputs,
            "total_tokens": inputs + outputs,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }


def test_cost_uses_disjoint_cache_categories_and_does_not_double_count_reasoning() -> None:
    ledger = UsageLedger("gpt-6-astra")
    ledger.record(200, payload(), 0.25)
    report = ledger.report()
    assert report["estimated_cost_usd"] == pytest.approx((50 * 10 + 20 + 30 * 12.5 + 10 * 50) / 1e6)
    assert report["reasoning_tokens"] == 2 and report["output_tokens"] == 10
    assert report["request_time_s"] == 0.25
    ledger.record(429, {"error": {"code": "rate_limit_exceeded"}}, 0.1)
    assert ledger.report()["http_attempts"] == 2
    assert ledger.report()["estimated_cost_usd"] == report["estimated_cost_usd"]
    ledger.reset()
    assert ledger.report()["http_attempts"] == 0


def test_recover_actual_agent_usage_cost_from_four_response_categories() -> None:
    ledger = UsageLedger("gpt-6-astra")
    for inputs, outputs, reads, writes in [
        (1555, 118, 0, 1552),
        (2050, 146, 1552, 495),
        (2185, 137, 0, 2182),
        (2521, 159, 1163, 1355),
    ]:
        data = payload(inputs, outputs)
        data["usage"]["input_tokens_details"] = {
            "cached_tokens": reads,
            "cache_write_tokens": writes,
        }
        ledger.record(200, data, 0)
    assert ledger.report()["estimated_cost_usd"] == pytest.approx(0.100635)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected_before_client_creation(budget: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        UsageLedger("gpt-6-astra", budget)


def test_unknown_usage_prevents_spending_again_under_a_budget() -> None:
    ledger = UsageLedger("gpt-6-astra", 1)
    ledger.record(200, {"output": []}, 0)
    assert ledger.report()["estimated_cost_usd"] is None
    with pytest.raises(RuntimeError, match="unknown"):
        ledger.check_budget()
    assert normalized_usage({"usage": {"input_tokens": True, "output_tokens": -1}}) is None


def test_output_cap_and_budget_are_sent_and_checked_without_network() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload())

    client = ResponsesClient(
        Provider("https://api.openai.com/v1", "dummy", "gpt-6-astra"),
        transport=httpx.MockTransport(respond),
        max_output_tokens=256,
        max_cost_usd=0.0001,
    )
    try:
        message = client.complete([], [])
        assert message.usage is not None
        assert message.usage["cache_creation_input_tokens"] == 30
        assert requests[0]["max_output_tokens"] == 256
        with pytest.raises(RuntimeError, match="budget reached"):
            client.complete([], [])
        assert len(requests) == 1
    finally:
        client.close()


def test_truncated_code_is_billed_but_not_returned_for_execution() -> None:
    data = payload()
    data["status"] = "incomplete"
    data["incomplete_details"] = {"reason": "max_output_tokens"}
    client = ResponsesClient(
        Provider("http://mock/v1", "", "gpt-6-astra"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data)),
    )
    try:
        with pytest.raises(RuntimeError, match="partial code"):
            client.complete([], [])
        assert client.usage_ledger.report()["output_tokens"] == 10
    finally:
        client.close()


def test_long_context_pricing_and_unknown_tiers_are_not_silently_underpriced() -> None:
    ledger = UsageLedger("gpt-6-astra")
    data = payload(300_000, 10)
    data["service_tier"] = "flex"
    ledger.record(200, data, 0)
    expected = 0.5 * (2 * (299950 * 10 + 20 + 30 * 12.5) + 1.5 * 10 * 50) / 1e6
    assert ledger.report()["estimated_cost_usd"] == pytest.approx(expected)
    data["service_tier"] = "unknown"
    ledger.record(200, data, 0)
    assert ledger.report()["estimated_cost_usd"] is None


def test_paid_failed_response_is_accounted_before_parse_raises() -> None:
    data = payload()
    data["status"] = "failed"
    data["error"] = {"message": "model execution failed"}
    client = ResponsesClient(
        Provider("http://mock/v1", "", "gpt-6-astra"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data)),
    )
    try:
        with pytest.raises(RuntimeError, match="execution failed"):
            client.complete([], [])
        assert client.usage_ledger.report()["output_tokens"] == 10
        assert client.usage_ledger.report()["estimated_cost_usd"] > 0
    finally:
        client.close()


@pytest.mark.parametrize("code", ["credit_balance_exhausted", "project_spend_limit_exceeded"])
def test_billing_rejection_is_not_retried(code: str) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": code, "type": "insufficient_quota"}})

    client = ResponsesClient(
        Provider("http://mock/v1", "", "gpt-6-astra"),
        transport=httpx.MockTransport(respond),
        backoff_s=0,
    )
    try:
        with pytest.raises(RuntimeError, match="no automatic retry"):
            client.complete([], [])
        assert len(requests) == 1
    finally:
        client.close()


def test_timeout_spend_is_unknown_and_budget_prevents_a_blind_retry() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("response was lost", request=request)

    client = ResponsesClient(
        Provider("http://mock/v1", "", "gpt-6-astra"),
        transport=httpx.MockTransport(respond),
        max_cost_usd=1,
        backoff_s=0,
    )
    try:
        with pytest.raises(RuntimeError, match="unknown"):
            client.complete([], [])
        assert len(requests) == 1
        assert client.usage_ledger.report()["estimated_cost_usd"] is None
        assert client.usage_ledger.report()["usage_complete"] is False
    finally:
        client.close()
