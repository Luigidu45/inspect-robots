"""Exercise the reset runner's task wiring, reference images, and summary."""

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
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment

from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult


def test_reset_runner_counts_success_and_sends_reference_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful reset must count once and send goal plus current cameras."""
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
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "move_to_configuration([0, 0, 0, 0, 0, 0, 0.8])",
                        }
                    }
                ]
            },
        )

    class World(OCBenchHanoiResetEmbodiment):
        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            return Observation(
                images=dict.fromkeys(["front", "side", "wrist"], image),
                state={"joint_pos": np.array([0, 0, 0, 0, 0, 0, 0.5])},
                instruction=scene.instruction,
                extra={"reference_images": dict.fromkeys(["front", "side", "wrist"], image)},
            )

        def step(self, action: Action) -> StepResult:
            return StepResult(
                observation=self.reset(Scene("s", "reset")),
                terminated=True,
                termination_reason="success",
                info={"success": True, "reset_success": True},
            )

    def policy(**kwargs: Any) -> CapxPolicy:
        return CapxPolicy(**kwargs, transport=httpx.MockTransport(respond), env={})

    monkeypatch.setattr(run_reset, "OCBenchHanoiResetEmbodiment", World)
    monkeypatch.setattr(run_reset, "CapxPolicy", policy)
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
    contents = requests[0]["messages"][-1]["content"]
    assert sum(part["type"] == "image_url" for part in contents) == 6
    assert "Reference goal images" in json.dumps(contents)
