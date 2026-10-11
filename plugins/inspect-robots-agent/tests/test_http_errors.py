"""Shared HTTP diagnostics distinguish throttling from nonretryable API limits."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from inspect_robots_agent import ChatClient, Provider, ResponsesClient
from inspect_robots_agent._capture import WireCapture


def _success(wire: str) -> dict[str, Any]:
    if wire == "chat":
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    }


def _client(
    wire: str, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> ChatClient | ResponsesClient:
    provider = Provider(base_url="http://llm.test/v1", api_key="", model="gpt-6-astra")
    client_type = ChatClient if wire == "chat" else ResponsesClient
    return client_type(provider, transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize(
    ("metric", "limit", "used", "delay", "expected_time"),
    [
        ("requests per day (RPD)", "50", "50", 1728, "28 min 48 s"),
        ("requests per minute (RPM)", "100", "100", 60, "1 min"),
        ("tokens per minute (TPM)", "20000", "19592.5", 3661, "1 h 1 min 1 s"),
    ],
)
def test_rate_limit_notice_uses_provider_metrics_and_precedes_sleep(
    wire: str,
    metric: str,
    limit: str,
    used: str,
    delay: int,
    expected_time: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": str(delay)},
                json={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "type": "requests",
                        "message": (
                            "Rate limit reached for gpt-6-astra in organization org-private "
                            f"on {metric}: Limit {limit}, Used {used}, Requested 1."
                        ),
                    }
                },
            )
        return httpx.Response(200, json=_success(wire))

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        notice = capsys.readouterr().err
        window = metric.split(" (")[0]
        assert notice == (
            f"[llm] Rate limit reached: gpt-6-astra, {window} ({used}/{limit}). "
            f"Retry in {expected_time}; attempt 2/3 (HTTP 429).\n"
        )
        assert "org-private" not in notice
        assert "reset" not in notice

    monkeypatch.setattr("inspect_robots_agent._llm.time.sleep", sleep)
    client = _client(wire, handler)
    try:
        assert client.complete(messages=[], tools=[]).content == "ok"
        assert calls == 2
        assert sleeps == [float(delay)]
        assert capsys.readouterr().err == ""
    finally:
        client.close()


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ({"code": "credit_balance_exhausted"}, "credit balance exhausted"),
        ({"code": "organization_spend_limit_exceeded"}, "organization spend limit reached"),
        ({"code": "project_spend_limit_exceeded"}, "project spend limit reached"),
        ({"code": "organization_usage_limit_exceeded"}, "organization usage limit reached"),
        ({"code": "insufficient_quota"}, "API quota unavailable"),
        ({"type": "insufficient_quota"}, "API quota unavailable"),
    ],
)
def test_billing_and_usage_errors_stop_without_sleep_or_retry_and_are_captured(
    wire: str,
    error: dict[str, str],
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0
    payload = {"error": {**error, "message": "private billing details"}}
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "1728"}, json=payload)

    def sleep(seconds: float) -> None:
        pytest.fail("billing or usage errors must not wait or retry")

    monkeypatch.setattr("inspect_robots_agent._llm.time.sleep", sleep)
    client = _client(wire, handler, capture=capture)
    try:
        with pytest.raises(RuntimeError, match=reason) as exc:
            client.complete(messages=[], tools=[])
        assert "no automatic retry" in str(exc.value)
        assert "private billing details" not in str(exc.value)
        assert calls == 1
        assert capsys.readouterr().err == ""
        relative = capture.end_trial()
        assert relative is not None
        rows = [json.loads(line) for line in (tmp_path / relative).read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == 429
        assert rows[0]["response"] == payload
    finally:
        client.close()


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize(
    ("payload", "label"),
    [
        ({"error": {"code": "rate_limit_exceeded"}}, "Rate limit reached: gpt-6-astra"),
        ({"error": {"type": "rate_limit_error"}}, "Rate limit reached: gpt-6-astra"),
        ({"error": {"code": "slow_down"}}, "Request rate increased too quickly: gpt-6-astra"),
        ({"error": {"code": "unknown"}}, "HTTP 429: request rejected for gpt-6-astra"),
        ({"error": "private details"}, "HTTP 429: request rejected for gpt-6-astra"),
        (["private details"], "HTTP 429: request rejected for gpt-6-astra"),
        ("<html>private details</html>", "HTTP 429: request rejected for gpt-6-astra"),
    ],
)
def test_missing_or_unrecognized_error_details_do_not_invent_a_quota_or_reset(
    wire: str,
    payload: Any,
    label: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            if isinstance(payload, str):
                return httpx.Response(429, text=payload)
            return httpx.Response(429, json=payload)
        return httpx.Response(200, json=_success(wire))

    def sleep(seconds: float) -> None:
        assert seconds == 2
        notice = capsys.readouterr().err
        assert label in notice
        assert "Retry in 2 s; attempt 2/3" in notice
        assert notice.count("\n") == 1
        assert "private details" not in notice
        assert "50" not in notice and "reset" not in notice

    monkeypatch.setattr("inspect_robots_agent._llm.time.sleep", sleep)
    client = _client(wire, handler, backoff_s=2.0)
    try:
        assert client.complete(messages=[], tools=[]).content == "ok"
        assert calls == 2
    finally:
        client.close()
