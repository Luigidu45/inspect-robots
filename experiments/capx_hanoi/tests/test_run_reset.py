"""Exercise the reset runner's textual goal wiring and summary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment

from inspect_robots.logging import JsonLogSink, RerunSink
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult


def test_reset_runner_counts_success_without_sending_reference_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default reset must count success and send only current cameras."""
    experiment = Path(__file__).parents[1]
    sys.path.insert(0, str(experiment))
    try:
        spec = importlib.util.spec_from_file_location(
            "hanoi_reset_run", experiment / "run_reset.py"
        )
        assert spec is not None and spec.loader is not None
        run_reset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_reset)
    finally:
        sys.path.remove(str(experiment))
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "id": "fc_0",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_0",
                        "name": "move_joints",
                        "arguments": json.dumps(
                            {
                                "targets": {"gripper": 0.8},
                                "note": "I open the gripper to verify native tool execution.",
                            }
                        ),
                    }
                ],
            },
        )

    class World(OCBenchHanoiResetEmbodiment):
        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            extra = (
                {"reference_images": dict.fromkeys(["front", "side", "wrist"], image)}
                if self._expose_reference_images
                else {}
            )
            return Observation(
                images=dict.fromkeys(["front", "side", "wrist"], image),
                state={"joint_pos": np.array([0, 0, 0, 0, 0, 0, 0.5])},
                instruction=scene.instruction,
                extra=extra,
            )

        def step(self, action: Action) -> StepResult:
            return StepResult(
                observation=self.reset(Scene("s", "reset")),
                terminated=True,
                termination_reason="success",
                info={"success": True, "reset_success": True},
            )

    def policy(_args: Any, _robot: Any) -> LLMAgentPolicy:
        return LLMAgentPolicy(
            model="scripted-test",
            base_url="http://test/v1",
            api_key_env="TEST_KEY",
            wire="responses",
            max_llm_calls=8,
            effort="low",
            images="always",
            depth="off",
            transport=httpx.MockTransport(respond),
            env={},
        )

    monkeypatch.setattr(run_reset, "OCBenchHanoiResetEmbodiment", World)
    monkeypatch.setattr(run_reset, "_build_policy", policy)
    monkeypatch.setattr(
        run_reset,
        "load_episode_actions",
        lambda _root, episodes: {episode: np.zeros((1, 7)) for episode in episodes},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reset.py",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--model",
            "scripted-test",
            "--base-url",
            "http://test/v1",
            "--num-scenes",
            "1",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert run_reset.main() == 0
    summary = json.loads(next((tmp_path / "logs").glob("summary-*.json")).read_text())
    assert summary["scheduled_scenes"] == summary["successful_scenes"] == 1
    assert summary["success_rate_all_scheduled"] == 1
    assert summary["mean_agent_steps_on_success"] == 1
    assert summary["policy"] == "agent"
    assert summary["max_llm_calls"] == 64
    assert summary["goal_mode"] == "text"
    assert summary["protocol"] == "ocbench-hanoi-text-reset-30hz-v1"
    assert summary["rerun"] is False and summary["rerun_port"] is None
    contents = requests[0]["input"][-1]["content"]
    assert sum(part["type"] == "input_image" for part in contents) == 3
    assert "Reference goal images" not in json.dumps(contents)
    assert {tool["name"] for tool in requests[0]["tools"]} == {
        "move_joints",
        "done",
        "give_up",
    }


@pytest.fixture
def runner() -> Any:
    """Load the standalone CLI without changing its package layout."""
    experiment = Path(__file__).parents[1]
    sys.path.insert(0, str(experiment))
    try:
        spec = importlib.util.spec_from_file_location(
            "hanoi_reset_cli", experiment / "run_reset.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(experiment))


@pytest.mark.parametrize(
    ("backend", "assistance"),
    [("agent", None), ("capx", None), ("capx", "none"), ("capx", "ik"), ("capx", "sam3_ik")],
)
def test_real_policy_factory_uses_same_astra_settings(
    runner: Any, monkeypatch: pytest.MonkeyPatch, backend: str, assistance: str | None
) -> None:
    """Native tools stay native; CaP-X assistance affects only its helpers."""
    import inspect_robots_agent
    import inspect_robots_capx

    original = (
        inspect_robots_agent.LLMAgentPolicy
        if backend == "agent"
        else inspect_robots_capx.CapxPolicy
    )
    captured: dict[str, Any] = {}

    def construct(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(
            **kwargs, env={}, transport=httpx.MockTransport(lambda _: httpx.Response(500))
        )

    module = inspect_robots_agent if backend == "agent" else inspect_robots_capx
    name = "LLMAgentPolicy" if backend == "agent" else "CapxPolicy"
    monkeypatch.setattr(module, name, construct)
    args = argparse.Namespace(
        policy=backend,
        assistance=assistance,
        model="gpt-6-astra",
        base_url="http://test/v1",
        api_key_env="OPENAI_API_KEY",
        wire="responses",
        max_llm_calls=8,
        prior_learnings=None,
        effort="low",
        images="always",
        live_transcript=True,
        sam3_url="http://sam3.test:8114" if assistance == "sam3_ik" else None,
        sam3_timeout_s=30.0,
    )
    robot = OCBenchHanoiResetEmbodiment({0: np.zeros((1, 7))})
    policy = runner._build_policy(args, robot)
    try:
        assert captured["model"] == "gpt-6-astra"
        assert captured["wire"] == "responses"
        assert captured["effort"] == "low"
        assert captured["max_llm_calls"] == 8
        assert captured["transcript_echo"] is True
        if backend == "agent":
            assert "assistance" not in captured and "ik_solver" not in captured
            assert captured["images"] == "always" and captured["depth"] == "off"
        else:
            assert captured["assistance"] == (assistance or "none")
            assert policy._servers is None
            if assistance in ("ik", "sam3_ik"):
                assert captured["ik_solver"] == robot.solve_ik
            else:
                assert captured["ik_solver"] is None
        policy.bind(robot.info)
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        robot.close()


def test_rerun_sink_keeps_the_canonical_json_log(runner: Any, tmp_path: Path) -> None:
    """Live visualization must not replace the persistent EvalLog."""
    sinks = runner._build_sinks(
        argparse.Namespace(rerun=True, rerun_port=9988, log_dir=str(tmp_path))
    )
    assert sinks is not None
    assert isinstance(sinks[0], JsonLogSink)
    assert isinstance(sinks[1], RerunSink)
    assert sinks[0].log_dir == tmp_path
    assert sinks[1].spawn is True
    assert sinks[1].recording_dir == str(tmp_path)
    assert sinks[1].spawn_port == 9988


def test_no_rerun_returns_default_json_sink(runner: Any, tmp_path: Path) -> None:
    """The regular runtime must stay independent of optional Rerun packages."""
    assert (
        runner._build_sinks(argparse.Namespace(rerun=False, rerun_port=None, log_dir=str(tmp_path)))
        is None
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--policy", "agent", "--assistance", "none"], "only supported"),
        (["--policy", "capx", "--assistance", "sam3_ik"], "requires --sam3-url"),
        (["--policy", "agent", "--sam3-url", "http://sam3.test"], "only supported"),
        (["--policy", "capx", "--assistance", "full"], "invalid choice"),
        (["--sam3-timeout-s", "nan"], "finite and positive"),
        (
            ["--policy", "capx", "--assistance", "sam3_ik", "--sam3-url", "gpu-box:8114"],
            "HTTP(S) base URL",
        ),
        (["--policy", "capx", "--images", "on_demand"], "requires --images always"),
        (["--max-output-tokens", "0"], "must be positive"),
        (["--max-cost-usd", "nan"], "finite and positive"),
        (["--max-cost-usd", "1", "--model", "unknown"], "known gpt-6-astra pricing"),
        (
            ["--max-output-tokens", "100", "--wire", "chat"],
            "gpt-6-astra tool calling requires",
        ),
        (["--wire", "chat"], "gpt-6-astra tool calling requires"),
        (["--max-llm-calls", "0"], "must be positive"),
        (["--base-url", "[https://api.openai.com/v1](https://api.openai.com/v1)"], "plain HTTP"),
        (["--rerun-port", "9988"], "requires --rerun"),
        (["--rerun", "--rerun-port", "0"], "must be in [1, 65535]"),
        (["--policy", "agent", "--capx-max-queued-steps", "90"], "requires --policy capx"),
        (["--policy", "capx", "--capx-max-queued-steps", "0"], "positive value"),
    ],
)
def test_invalid_cli_combinations_fail_before_loading_data(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    message: str,
) -> None:
    """Do not prepare a simulation or make API calls for unsupported comparisons."""

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid CLI must fail before loading episodes")

    monkeypatch.setattr(runner, "load_episode_actions", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_reset.py", "--dataset-root", "missing", "--base-url", "http://test/v1", *options],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_sam3_preflight_failure_stops_before_dataset_replay(
    runner: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing service must not consume model calls or prepare scenes."""

    def failure(_args: Any) -> Any:
        raise ConnectionError("connection refused")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Must stop before reading actions")

    monkeypatch.setattr(runner, "_check_sam3", failure)
    monkeypatch.setattr(runner, "load_episode_actions", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reset.py",
            "--dataset-root",
            "missing",
            "--base-url",
            "http://test/v1",
            "--policy",
            "capx",
            "--assistance",
            "sam3_ik",
            "--sam3-url",
            "http://sam3.test:8114",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "SAM3 preflight failed before scene replay" in capsys.readouterr().err


@pytest.mark.parametrize("bad_shape", [False, True])
def test_sam3_preflight_validates_wire_and_mask_size(runner: Any, bad_shape: bool) -> None:
    """The readiness probe must accept empty matches and reject wrong image geometry."""
    import base64

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://sam3.test:8114/segment"
        body = json.loads(request.content)
        assert body["text_prompt"] == "disk"
        assert base64.b64decode(body["image_base64"]).startswith(b"\x89PNG")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "shape": [2, 2],
                        "mask_base64": base64.b64encode(bytes(4)).decode(),
                        "box": [0, 0, 2, 2],
                        "score": 0.9,
                        "label": "disk",
                    }
                ]
                if bad_shape
                else []
            },
        )

    args = argparse.Namespace(sam3_url="http://sam3.test:8114", sam3_timeout_s=30.0)
    if bad_shape:
        with pytest.raises(ValueError, match="input image size"):
            runner._check_sam3(args, transport=httpx.MockTransport(respond))
    else:
        report = runner._check_sam3(args, transport=httpx.MockTransport(respond))
        assert report["status"] == "passed" and report["matches"] == 0
