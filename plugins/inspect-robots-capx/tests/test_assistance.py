"""Assistance gates the prompt and namespace without calling disabled services."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation
from inspect_robots_capx import CapxPolicy
from inspect_robots_capx.policy import CapxPolicyConfig


def _info() -> EmbodimentInfo:
    return EmbodimentInfo(
        name="joint-test",
        action_space=Box(
            shape=(3,),
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.ones(3),
            semantics=ActionSemantics(
                "joint_pos", gripper="continuous", dim_labels=("j0", "j1", "gripper")
            ),
        ),
        observation_space=ObservationSpace(
            cameras=(CameraSpec("front", 2, 2),), state=StateSpec((StateField("joint_pos", (3,)),))
        ),
        control_hz=30.0,
        is_simulated=True,
    )


def _observation() -> Observation:
    return Observation(
        images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
        state={"joint_pos": np.array([0.0, 0.0, 1.0])},
    )


def _transport(code: str, requests: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses", "disabled helper contacted a server"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": code}],
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


def test_none_only_installs_motion_and_uses_astra_responses() -> None:
    requests: list[dict[str, Any]] = []
    code = (
        "assert 'solve_ik' not in globals()\n"
        "assert 'segment' not in globals()\n"
        "assert 'plan_grasp' not in globals()\n"
        "move_to_configuration([0.1, 0.0, 0.8])"
    )
    policy = CapxPolicy(
        model="gpt-6-astra",
        base_url="http://test/v1",
        wire="responses",
        effort="low",
        assistance="none",
        env={},
        transport=_transport(code, requests),
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert np.allclose(chunk.actions[-1].data, [0.1, 0.0, 0.8])
        assert policy._servers is None
        assert isinstance(policy.config, CapxPolicyConfig)
        assert policy.config.ik_backend == "none"
        assert requests[0]["model"] == "gpt-6-astra"
        assert requests[0]["reasoning"]["effort"] == "low"
        prompt = json.dumps(requests[0]["input"][0])
        assert (
            "solve_ik(" not in prompt and "segment(" not in prompt and "plan_grasp(" not in prompt
        )
    finally:
        policy.close()


def test_local_ik_receives_arm_cursor_and_never_calls_perception() -> None:
    requests: list[dict[str, Any]] = []
    seeds: list[np.ndarray] = []

    def solve(position: np.ndarray, quat: np.ndarray, seed: np.ndarray) -> np.ndarray:
        seeds.append(seed.copy())
        assert np.array_equal(position, [0.4, 0.0, 0.3])
        assert np.array_equal(quat, [1.0, 0.0, 0.0, 0.0])
        return np.array([0.1, -0.2])

    code = (
        "assert 'segment' not in globals()\n"
        "assert 'plan_grasp' not in globals()\n"
        "move_to_joints([0.05, 0.0])\n"
        "q = solve_ik([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])\n"
        "move_to_joints(q)"
    )
    policy = CapxPolicy(
        model="gpt-6-astra",
        base_url="http://test/v1",
        wire="responses",
        effort="low",
        assistance="ik",
        ik_solver=solve,
        env={},
        transport=_transport(code, requests),
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert np.allclose(chunk.actions[-1].data, [0.1, -0.2, 1.0])
        assert np.array_equal(seeds[0], [0.05, 0.0])
        assert policy._servers is None
        assert isinstance(policy.config, CapxPolicyConfig)
        assert policy.config.ik_backend == "local"
        assert "solve_ik" in policy.config.enabled_helpers
        prompt = json.dumps(requests[0]["input"][0])
        assert "solve_ik(" in prompt and "segment(" not in prompt
    finally:
        policy.close()


@pytest.mark.parametrize("profile", ["full", "invalid"])
def test_unimplemented_levels_fail_before_constructing_clients(profile: str) -> None:
    with pytest.raises(ValueError, match="assistance must"):
        CapxPolicy(model="gpt-6-astra", assistance=profile, env={})


def test_none_rejects_injected_ik() -> None:
    with pytest.raises(ValueError, match="requires IK"):
        CapxPolicy(assistance="none", ik_solver=lambda p, q, seed: seed)


@pytest.mark.parametrize("bad_joints", [np.array([0.0]), np.array([np.nan, 0.0])])
def test_invalid_local_ik_result_becomes_execution_feedback(bad_joints: np.ndarray) -> None:
    replies = iter(["solve_ik([0.4, 0, 0.3], [1, 0, 0, 0])", "FINISH"])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": next(replies)}}]}
        )

    policy = CapxPolicy(
        model="gpt-6-astra",
        base_url="http://test/v1",
        assistance="ik",
        ik_solver=lambda p, q, seed: bad_joints,
        env={},
        transport=httpx.MockTransport(handler),
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert chunk.actions[0].meta["stop_reason"] == "FINISH"
        assert "IK solver must return" in json.dumps(policy.transcript())
    finally:
        policy.close()


@pytest.mark.parametrize("empty", [False, True])
def test_sam3_ik_exposes_only_segmentation_and_local_ik(empty: bool) -> None:
    import base64

    model_requests: list[dict[str, Any]] = []
    service_requests: list[httpx.Request] = []
    seeds: list[np.ndarray] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/segment"
        service_requests.append(request)
        assert json.loads(request.content)["text_prompt"] == "disk"
        result = {
            "shape": [2, 2],
            "mask_base64": base64.b64encode(bytes([1, 0, 0, 1])).decode(),
            "box": [0, 0, 2, 2],
            "score": 0.9,
            "label": "disk",
        }
        return httpx.Response(200, json={"results": [] if empty else [result]})

    def solve(position: np.ndarray, quaternion: np.ndarray, seed: np.ndarray) -> np.ndarray:
        seeds.append(seed.copy())
        return np.array([0.1, -0.1])

    code = (
        "assert 'plan_grasp' not in globals()\n"
        "matches = segment('disk')\n"
        f"assert len(matches) == {0 if empty else 1}\n"
        "move_to_joints(solve_ik([0, 0, 0.3], [1, 0, 0, 0]))"
    )
    policy = CapxPolicy(
        model="gpt-6-astra",
        base_url="http://test/v1",
        wire="responses",
        effort="low",
        assistance="sam3_ik",
        ik_solver=solve,
        sam3_url="http://sam3.test:8114",
        env={},
        transport=_transport(code, model_requests),
        service_transport=httpx.MockTransport(respond),
    )
    try:
        policy.bind(_info())
        policy.reset(Scene("s", "restore"))
        chunk = policy.act(_observation())
        assert np.allclose(chunk.actions[-1].data, [0.1, -0.1, 1.0])
        assert np.array_equal(seeds[0], [0.0, 0.0])
        assert len(service_requests) == 1
        assert policy._servers is None and policy._pyroki_client is None
        assert policy._sam3_client is not None
        prompt = json.dumps(model_requests[0]["input"][0])
        assert "segment(" in prompt and "solve_ik(" in prompt and "plan_grasp(" not in prompt
        assert isinstance(policy.config, CapxPolicyConfig)
        assert set(policy.config.enabled_helpers) == {
            "segment",
            "solve_ik",
            "move_to_joints",
            "move_to_configuration",
            "open_gripper",
            "close_gripper",
        }
        policy.reset(Scene("next", "restore again"))
        assert policy._sandbox is not None
        assert "matches" not in policy._sandbox._namespace
    finally:
        policy.close()
    assert policy._sam3_client._http.is_closed


def test_ik_http_profile_creates_only_pyroki_client() -> None:
    policy = CapxPolicy(model="gpt-6-astra", base_url="http://test/v1", assistance="ik", env={})
    try:
        assert policy._servers is None and policy._sam3_client is None
        assert policy._pyroki_client is not None
    finally:
        policy.close()
    assert policy._pyroki_client._http.is_closed
