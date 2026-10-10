"""Smoke-test all reset backends with real MuJoCo and scripted Astra responses.

No provider request is made. This validates wiring, IK and physical execution,
not autonomous Hanoi restoration or the reasoning ability of GPT Astra.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import run_reset


def _ik_code(robot: Any) -> str:
    raw = robot._env.unwrapped
    lie = importlib.import_module("ocbench.lie")
    position = raw.data.site_xpos[raw._pinch_site_id].copy()
    quaternion = lie.SO3.from_matrix(
        raw.data.site_xmat[raw._pinch_site_id].reshape(3, 3)
    ).wxyz.copy()
    seed = robot._joint_state()[:6].copy()
    before = [raw.data.qpos.copy(), raw.data.qvel.copy(), raw.data.ctrl.copy()]
    before_time = raw.data.time
    for offset in ([0, 0, 0], [0, 0, 0.002], [0.002, 0, 0.005]):
        solution = robot.solve_ik(position + offset, quaternion, seed)
        assert solution.shape == (6,) and np.all(np.isfinite(solution))
    try:
        robot.solve_ik(np.array([100.0, 100.0, 100.0]), quaternion, seed)
    except ValueError as error:
        assert "did not converge" in str(error)
    else:
        raise AssertionError("Unreachable pose must be rejected")
    assert raw.data.time == before_time
    for saved, actual in zip(before, [raw.data.qpos, raw.data.qvel, raw.data.ctrl], strict=True):
        assert np.array_equal(saved, actual), "IK changed the physical world"
    target = (position + np.array([0, 0, 0.002])).tolist()
    return f"q = solve_ik({target!r}, {quaternion.tolist()!r})\nmove_to_joints(q)"


def _smoke(
    backend: str,
    assistance: str | None,
    dataset_root: Path,
    output_root: Path,
    sam3_url: str | None = None,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    service_calls: list[str] = []
    robots: list[Any] = []
    original_factory = run_reset._build_policy
    module = importlib.import_module(f"inspect_robots_{backend}")
    policy_name = "LLMAgentPolicy" if backend == "agent" else "CapxPolicy"
    constructor = getattr(module, policy_name)
    original_checker = run_reset._check_sam3

    def service_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/segment", "Unrequested IK or grasp service"
        service_calls.append(request.url.path)
        body = json.loads(request.content)
        assert body["text_prompt"] == "disk"
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[80:120, 80:120] = 1
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "shape": [224, 224],
                        "mask_base64": base64.b64encode(mask.tobytes()).decode(),
                        "box": [80, 80, 120, 120],
                        "score": 0.9,
                        "label": "disk",
                    }
                ]
            },
        )

    def preflight(options: Any) -> dict[str, Any]:
        return original_checker(options, transport=httpx.MockTransport(service_response))

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses", "Unexpected model-server request"
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-astra"
        assert body["reasoning"]["effort"] == "low"
        calls.append(body)
        output: list[dict[str, Any]]
        if backend == "agent":
            name = "move_joints" if len(calls) == 1 else "done"
            arguments = (
                {
                    "targets": {"gripper": float(robots[0]._joint_state()[-1])},
                    "note": "Scripted integration check; hold current configuration.",
                }
                if len(calls) == 1
                else {}
            )
            output = [
                {
                    "id": f"fc_{len(calls)}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": f"call_{len(calls)}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            ]
        else:
            if len(calls) == 1:
                assert robots[0]._env is not None
                code = (
                    _ik_code(robots[0])
                    if assistance in ("ik", "sam3_ik")
                    else "assert 'solve_ik' not in globals()\n"
                    "move_to_configuration(obs['state']['joint_pos'])"
                )
                if assistance == "sam3_ik":
                    code = (
                        "assert 'plan_grasp' not in globals()\n"
                        "matches = segment('disk')\n"
                        "assert all(m['mask'].shape == (224, 224) for m in matches)\n"
                        "print('segmentation matches', len(matches))\n" + code
                    )
                content = "```python\n" + code + "\n```"
            else:
                content = "FINISH"
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ]
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "model": "gpt-6-astra",
                "output": output,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )

    def construct(**kwargs: Any) -> Any:
        if backend == "capx" and assistance == "sam3_ik":
            kwargs["service_transport"] = (
                httpx.HTTPTransport() if sam3_url else httpx.MockTransport(service_response)
            )
        policy = constructor(**kwargs, transport=httpx.MockTransport(respond), env={})
        if backend == "capx":
            assert policy._servers is None and policy._pyroki_client is None
            assert (policy._sam3_client is not None) == (assistance == "sam3_ik")
        return policy

    def factory(options: Any, robot: Any) -> Any:
        robots.append(robot)
        return original_factory(options, robot)

    output_dir = output_root / (backend + (f"-{assistance}" if assistance else ""))
    argv = [
        "run_reset.py",
        "--dataset-root",
        str(dataset_root),
        "--policy",
        backend,
        "--model",
        "gpt-6-astra",
        "--base-url",
        "http://scripted.test/v1",
        "--wire",
        "responses",
        "--effort",
        "low",
        "--images",
        "always",
        "--max-llm-calls",
        "8",
        "--num-scenes",
        "1",
        "--log-dir",
        str(output_dir),
    ]
    if assistance:
        argv += ["--assistance", assistance]
    if assistance == "sam3_ik":
        argv += ["--sam3-url", sam3_url or "http://scripted-sam3.test:8114"]
    with (
        patch.object(module, policy_name, construct),
        patch.object(run_reset, "_build_policy", factory),
        patch.object(sys, "argv", argv),
        patch.object(run_reset, "_check_sam3", preflight)
        if assistance == "sam3_ik" and not sam3_url
        else nullcontext(),
    ):
        assert run_reset.main() == 0
    summary_path = max(output_dir.glob("summary-*.json"), key=lambda path: path.stat().st_mtime)
    summary = json.loads(summary_path.read_text())
    assert summary["errored_trials"] == 0
    assert summary["total_agent_control_steps"] > 0
    assert len(calls) == 2
    assert summary["llm_calls"] == 2
    assert summary["llm_usage"][0]["input_tokens"] == 200
    assert summary["llm_usage"][0]["output_tokens"] == 20
    assert summary["estimated_cost_usd"] is not None and summary["estimated_cost_usd"] > 0
    assert summary["robot_diagnostics"][0]["control_steps"] > 0
    eval_log = max(output_dir.glob("ocbench*.json"), key=lambda path: path.stat().st_mtime)
    trial_metadata = json.loads(eval_log.read_text())["samples"][0]["trial_metadata"][0]
    assert "wire_capture" in trial_metadata and "transcript" in trial_metadata
    images = sum(
        part.get("type") == "input_image"
        for message in calls[0]["input"]
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
    if assistance == "sam3_ik" and not sam3_url:
        assert len(service_calls) == 2  # Startup probe plus the model-generated call.
    assert images == 6, "Expected three goal and three current cameras"
    return {
        "policy": backend,
        "assistance": assistance,
        "llm_calls": len(calls),
        "initial_images": images,
        "sam3_service": ("real" if sam3_url else "mock") if assistance == "sam3_ik" else None,
        "summary": str(summary_path),
        "infrastructure_passed": True,
        "usage_and_diagnostics_persisted": True,
        "cost_is_scripted": True,
        "api_requests": 0,
    }


def main() -> int:
    """Run paired infrastructure checks and save explicit non-benchmark evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-url", help="Use a real SAM3 service instead of scripted masks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for backend, assistance in [
        ("agent", None),
        ("capx", "none"),
        ("capx", "ik"),
        ("capx", "sam3_ik"),
    ]:
        results.append(
            _smoke(backend, assistance, args.dataset_root, args.output_dir, args.sam3_url)
        )
    report = {
        "kind": "scripted-infrastructure-smoke",
        "live_llm": False,
        "autonomous_reset_evaluated": False,
        "model_request": "gpt-6-astra",
        "results": results,
    }
    (args.output_dir / "assistance-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
