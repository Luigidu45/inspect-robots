"""Verify action conversion, information isolation and frozen scene identity without MuJoCo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_ocbench import OCBenchEmbodiment, hanoi_task
from inspect_robots_ocbench.embodiment import DELTA_SCALE


class FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.data = SimpleNamespace(
            qpos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]), qvel=np.zeros(7)
        )
        self.model = SimpleNamespace(
            actuator_ctrlrange=np.column_stack(
                [
                    -np.array([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831]),
                    np.array([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831]),
                ]
            )
        )
        self._arm_actuator_ids = np.arange(6)
        self._arm_joint_ids = np.arange(6)
        self._gripper_opening_joint_id = 6
        self._joint_action_delta = DELTA_SCALE.copy()
        self.command: Any = None
        self.seed = -1
        self.success = False
        self.failure = False
        self.healthy = True
        self.closed = False

    def reset(self, seed: int) -> tuple[None, dict[str, Any]]:
        self.seed = seed
        return None, {"dynamics_info": np.zeros(12)}

    def render(self, camera: str) -> Any:
        assert camera in {"front", "side", "ur5e/wrist"}
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def step(self, action: Any) -> tuple[None, int, bool, bool, dict[str, Any]]:
        self.command = action.copy()
        return (
            None,
            1,
            self.success,
            False,
            {
                "success": self.success,
                "failure": self.failure,
                "healthy": self.healthy,
                "privileged/peg_0_pos": np.ones(3),
            },
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def simulator(monkeypatch: pytest.MonkeyPatch) -> FakeEnv:
    env = FakeEnv()

    def make(env_id: str, **kwargs: Any) -> FakeEnv:
        assert env_id == "hanoi-cpu-triple-task2-v0"
        assert kwargs == {
            "control_timestep": 1 / 30,
            "physics_timestep": 1 / 600,
            "width": 224,
            "height": 224,
        }
        return env

    monkeypatch.setattr(
        "inspect_robots_ocbench.embodiment.importlib.import_module",
        lambda _: SimpleNamespace(make=make),
    )
    return env


def test_reset_seed_hash_observation_isolation_and_clock(simulator: FakeEnv) -> None:
    robot = OCBenchEmbodiment()
    digest = hashlib.sha256(
        simulator.data.qpos.tobytes() + simulator.data.qvel.tobytes() + np.zeros(12).tobytes()
    ).hexdigest()
    obs = robot.reset(
        Scene("scene", "solve", metadata={"reset_seed": 123, "initial_world_sha256": digest}),
        seed=456,
    )
    assert simulator.seed == 123
    assert robot.reset_provenance["framework_trial_seed"] == 456
    assert obs.extra == {} and set(obs.state) == {"joint_pos"}
    assert set(obs.images) == {"front", "side", "wrist"}
    assert obs.state["joint_pos"].shape == (7,)
    assert obs.state["joint_pos"].dtype == np.float32
    assert obs.instruction == "solve" and obs.state_time == 0
    result = robot.step(Action(np.array([0.09, -0.09, 0.18, 0.36, -0.36, 0.72, 0.44])))
    np.testing.assert_allclose(simulator.command, [0.5, -0.5, 1, 1, -1, 1, -0.5], atol=1e-6)
    assert result.observation.state_time == 1 / 30
    assert "privileged/peg_0_pos" not in result.info
    assert not result.info["success"]
    robot.close()
    robot.close()
    assert simulator.closed


@pytest.mark.parametrize(
    "failure,healthy,expected", [(False, True, True), (True, True, False), (False, False, False)]
)
def test_success_requires_healthy_without_failure(
    simulator: FakeEnv, failure: bool, healthy: bool, expected: bool
) -> None:
    robot = OCBenchEmbodiment()
    robot.reset(Scene("scene", "solve"), seed=0)
    simulator.success = True
    simulator.failure, simulator.healthy = failure, healthy
    result = robot.step(Action(np.zeros(7)))
    assert result.terminated and result.info["success"] is expected
    assert result.termination_reason == ("success" if expected else "failure")


def test_invalid_seed_hash_bounds_and_actions_fail_before_execution(simulator: FakeEnv) -> None:
    robot = OCBenchEmbodiment()
    with pytest.raises(RuntimeError):
        robot.step(Action(np.zeros(7)))
    with pytest.raises(ValueError, match="reset_seed"):
        robot.reset(Scene("s", "solve", metadata={"reset_seed": True}), seed=0)
    with pytest.raises(ValueError, match="world hash"):
        robot.reset(Scene("s", "solve", metadata={"initial_world_sha256": "wrong"}), seed=0)
    simulator.model.actuator_ctrlrange[0, 0] = -2
    with pytest.raises(ValueError, match="actuator limits"):
        robot.reset(Scene("s", "solve"), seed=0)
    for target in [np.zeros(6), np.full(7, np.nan)]:
        with pytest.raises(ValueError, match="seven finite"):
            robot.step(Action(target))
    assert simulator.command is None


def test_task_excludes_privileged_layout_and_preserves_all_scheduled_scenes(tmp_path: Path) -> None:
    path = tmp_path / "scenes.json"
    path.write_text(
        json.dumps(
            {
                "validation": {
                    "scenes": [
                        {
                            "scene_id": "v1",
                            "seed": 123,
                            "initial_world_sha256": "hash",
                            "peg_layout": [1, 2],
                        }
                    ]
                }
            }
        )
    )
    task = hanoi_task(str(path))
    assert task.max_steps == 7000 and len(task.scenes) == 1
    assert task.scenes[0].metadata == {"reset_seed": 123, "initial_world_sha256": "hash"}
    assert task.metadata["scheduled_scenes"] == 1
    with pytest.raises(ValueError):
        hanoi_task(str(path), split="train")
    with pytest.raises(ValueError):
        hanoi_task(str(path), num_scenes=2)
