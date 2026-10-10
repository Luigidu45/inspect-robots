"""Exercise the actual runner, CaP-X HTTP wire, scoring and error denominator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from inspect_robots_capx import CapxPolicy
from inspect_robots_ocbench import OCBenchEmbodiment

from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult


@pytest.mark.parametrize("failure,finish", [(False, False), (True, False), (False, True)])
def test_runner_counts_errors_in_scheduled_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: bool, finish: bool
) -> None:
    """A completed log and a failed trial must produce different task success rates."""
    spec = importlib.util.spec_from_file_location("hanoi_run", Path(__file__).parents[1] / "run.py")
    assert spec is not None and spec.loader is not None
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        code = "FINISH" if finish else "move_to_configuration([0, 0, 0, 0, 0, 0, 0.4])"
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": code}}]}
        )

    class World(OCBenchEmbodiment):
        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            if failure:
                raise ValueError("deliberate reset failure")
            return Observation(
                images={
                    name: np.zeros((224, 224, 3), dtype=np.uint8)
                    for name in ["front", "side", "wrist"]
                },
                state={"joint_pos": np.array([0, 0, 0, 0, 0, 0, 0.5])},
                instruction=scene.instruction,
            )

        def step(self, action: Action) -> StepResult:
            return StepResult(
                observation=self.reset(Scene("s", "solve")),
                terminated=not finish,
                termination_reason=None if finish else "success",
                info={"success": not finish},
            )

    def policy(**kwargs: Any) -> CapxPolicy:
        return CapxPolicy(**kwargs, transport=httpx.MockTransport(respond), env={})

    monkeypatch.setattr(run, "OCBenchEmbodiment", World)
    monkeypatch.setattr(run, "CapxPolicy", policy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "scripted-test",
            "--base-url",
            "http://test/v1",
            "--num-scenes",
            "1",
            "--log-dir",
            str(tmp_path),
        ],
    )
    exit_code = run.main()
    summary = json.loads(next(tmp_path.glob("summary-*.json")).read_text())
    assert summary["scheduled_scenes"] == 1
    assert summary["successful_scenes"] == int(not failure and not finish)
    assert summary["success_rate_all_scheduled"] == int(not failure and not finish)
    assert summary["errored_trials"] == int(failure)
    assert exit_code == int(failure)
    if not failure:
        assert summary["received_llm_replies"] == 1
        assert len(requests) == 1
        contents = requests[0]["messages"][-1]["content"]
        assert sum(part["type"] == "image_url" for part in contents) == 3
        assert "joint_pos" in json.dumps(contents)
        assert "privileged/" not in json.dumps(contents)
