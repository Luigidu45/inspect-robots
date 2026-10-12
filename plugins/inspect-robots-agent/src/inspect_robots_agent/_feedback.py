"""Render bounded physical feedback and retain only its latest prompt snapshot."""

from __future__ import annotations

from typing import Any

from inspect_robots.types import Observation

_PREFIX = "Motion feedback: "
_MAX_CHARS = 320


def motion_feedback_part(observation: Observation) -> dict[str, Any] | None:
    """Expose only the controller's compact summary, leaving audit arrays in the log."""
    report = observation.extra.get("motion_feedback")
    if not isinstance(report, dict) or not isinstance(report.get("summary"), str):
        return None
    return {"type": "text", "text": _PREFIX + report["summary"][:_MAX_CHARS]}


def latest_motion_feedback(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop older physical summaries from a request view without altering the audit."""
    latest: tuple[int, int] | None = None
    for i, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            for j, part in enumerate(content):
                if isinstance(part, dict) and str(part.get("text", "")).startswith(_PREFIX):
                    latest = i, j
    if latest is None:
        return messages
    result: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            content = [
                part
                for j, part in enumerate(content)
                if (i, j) == latest
                or not (isinstance(part, dict) and str(part.get("text", "")).startswith(_PREFIX))
            ]
            message = {**message, "content": content}
        result.append(message)
    return result
