"""Keep physical summaries compact while preserving the full local transcript."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from inspect_robots.types import Observation
from inspect_robots_agent._feedback import latest_motion_feedback, motion_feedback_part
from inspect_robots_agent.policy import _observation_content


def test_summary_is_bounded_and_does_not_render_audit_arrays() -> None:
    observation = Observation(
        extra={
            "motion_feedback": {
                "summary": "x" * 500,
                "requested_configuration": [12345] * 7,
            }
        }
    )
    part = motion_feedback_part(observation)
    assert part is not None
    assert part["text"] == "Motion feedback: " + "x" * 320
    assert part in _observation_content(observation)
    assert "12345" not in str(part)
    assert motion_feedback_part(Observation()) is None


def test_only_latest_feedback_is_sent_and_transcript_remains_complete() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Motion feedback: old"},
                {"type": "text", "text": "Original observation"},
            ],
        },
        {"role": "assistant", "content": "Chosen action"},
        {"role": "user", "content": [{"type": "text", "text": "Motion feedback: latest"}]},
    ]
    original = deepcopy(messages)
    view = latest_motion_feedback(messages)
    assert "Motion feedback: old" not in str(view)
    assert "Motion feedback: latest" in str(view)
    assert "Original observation" in str(view)
    assert messages == original
    assert latest_motion_feedback(messages[:0]) == []
