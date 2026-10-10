"""Reject long plans and roll back all actions when generated code fails."""

import json

import httpx
import numpy as np
import pytest
from test_assistance import _info, _observation

from inspect_robots.scene import Scene
from inspect_robots_capx import CapxPolicy


@pytest.mark.parametrize("caught", [False, True])
def test_queue_limit_does_not_execute_a_partial_plan(caught: bool) -> None:
    code = "move_to_joints([0.1,0])\nmove_to_joints([0.5,0])"
    if caught:
        code = "try:\n " + code.replace("\n", "\n ") + "\nexcept ValueError: pass"
    replies = iter([code, "move_to_joints([0.02,0])"])

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": next(replies)}}]}
        )

    policy = CapxPolicy(
        model="mock",
        base_url="http://mock/v1",
        assistance="none",
        max_queued_steps=20,
        transport=httpx.MockTransport(respond),
        env={},
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert len(chunk.actions) <= 20
        assert np.allclose(chunk.actions[-1].data, [0.02, 0, 1])
        assert all(action.data[0] <= 0.02 for action in chunk.actions)
        assert "discarded" in json.dumps(policy.transcript())
    finally:
        policy.close()


def test_python_error_rolls_back_cursor_and_all_queued_motions() -> None:
    replies = iter(["close_gripper()\nraise ValueError('broken')", "move_to_joints([0.01,0])"])
    policy = CapxPolicy(
        model="mock",
        base_url="http://mock/v1",
        assistance="none",
        env={},
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": next(replies)}}]},
            )
        ),
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert all(action.data[-1] == 1 for action in chunk.actions)
        assert "broken" in json.dumps(policy.transcript())
    finally:
        policy.close()
