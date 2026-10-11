"""Verify action conversion, information isolation and frozen scene identity without MuJoCo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_ocbench import (
    RESET_INSTRUCTION,
    OCBenchEmbodiment,
    OCBenchHanoiResetEmbodiment,
    hanoi_reset_task,
    hanoi_task,
)
from inspect_robots_ocbench.embodiment import (
    DELTA_SCALE,
    REFERENCE_IMAGES_KEY,
    RESET_STABILITY_STEPS,
    ocbench_embodiment,
)


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
        self.bad_floor_contact = False
        self.illegal_stack = False
        self.healthy = True
        self.closed = False
        self.make_kwargs: dict[str, Any] = {}
        self.reset_replay_behavior = False
        self.restore_on_step = False
        self.tower_on_initial_peg = True
        self.step_count = 0
        self.cur_task_info = {"init_peg": 1}

    def reset(self, seed: int) -> tuple[None, dict[str, Any]]:
        self.seed = seed
        self.step_count = 0
        self.tower_on_initial_peg = True
        return None, {"dynamics_info": np.zeros(12)}

    def render(self, camera: str) -> Any:
        assert camera in {"front", "side", "ur5e/wrist"}
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def step(self, action: Any) -> tuple[None, int, bool, bool, dict[str, Any]]:
        self.command = action.copy()
        self.step_count += 1
        if self.reset_replay_behavior:
            if self.step_count == 1:
                self.tower_on_initial_peg = False
            elif self.restore_on_step:
                self.tower_on_initial_peg = True
            self.success = not self.tower_on_initial_peg
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

    def _is_tower_on_peg(self, peg: int) -> bool:
        assert peg == 1
        return self.tower_on_initial_peg

    def close(self) -> None:
        self.closed = True

    def _has_bad_floor_contact(self) -> bool:
        return self.bad_floor_contact

    def _has_illegal_stack(self) -> bool:
        return self.illegal_stack


def test_registered_embodiment_declares_workspace_runtime_remediation() -> None:
    """Keep doctor aligned with the locked workspace installation path."""
    assert getattr(ocbench_embodiment, "RUNTIME_REQUIREMENTS", None) == {
        "ocbench": "uv sync --all-packages --extra ocbench",
        "pyarrow": "uv sync --all-packages --extra ocbench",
    }


@pytest.fixture
def simulator(monkeypatch: pytest.MonkeyPatch) -> FakeEnv:
    env = FakeEnv()

    def make(env_id: str, **kwargs: Any) -> FakeEnv:
        assert env_id == "hanoi-cpu-triple-task2-v0"
        env.make_kwargs = kwargs
        assert kwargs["control_timestep"] == 1 / 30
        assert kwargs["physics_timestep"] == 1 / 600
        assert kwargs["width"] == kwargs["height"] == 224
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


def test_reset_embodiment_replays_episode_and_scores_original_stack(simulator: FakeEnv) -> None:
    simulator.reset_replay_behavior = True
    robot = OCBenchHanoiResetEmbodiment(
        {7: np.zeros((1, 7), dtype=np.float32)}, expose_reference_images=True
    )
    scene = Scene(
        "reset-7",
        RESET_INSTRUCTION,
        metadata={
            "reset_seed": 123,
            "dataset_episode": 7,
            "initial_world_sha256": "cross-runtime-hash",
        },
    )

    observation = robot.reset(scene, seed=456)

    assert simulator.make_kwargs["terminate_at_success"] is False
    assert simulator.make_kwargs["max_episode_steps"] == 14_001
    assert set(observation.extra[REFERENCE_IMAGES_KEY]) == {"front", "side", "wrist"}
    assert robot.reset_provenance["dataset_episode"] == 7
    assert robot.reset_provenance["setup_replay_steps"] == 1
    assert robot.reset_provenance["initial_world_hash_matches"] is False
    assert not simulator.tower_on_initial_peg

    simulator.restore_on_step = True
    for expected_streak in range(1, RESET_STABILITY_STEPS + 1):
        result = robot.step(Action(np.array([0, 0, 0, 0, 0, 0, 0.5])))
        assert result.info["reset_stability_steps"] == expected_streak
        assert result.terminated is (expected_streak == RESET_STABILITY_STEPS)
    assert result.terminated and result.termination_reason == "success"
    assert result.info["reset_success"] is True
    assert result.info["source_task_success"] is False


def test_text_goal_reset_does_not_expose_reference_images(simulator: FakeEnv) -> None:
    """Text-goal trials must expose only current RGB and proprioception."""
    simulator.reset_replay_behavior = True
    robot = OCBenchHanoiResetEmbodiment({7: np.zeros((1, 7), dtype=np.float32)})
    observation = robot.reset(
        Scene(
            "reset-7",
            RESET_INSTRUCTION,
            metadata={"reset_seed": 123, "dataset_episode": 7},
        )
    )
    assert observation.extra == {}
    assert set(observation.images) == {"front", "side", "wrist"}
    assert robot.info.docs is not None
    assert "No goal image is provided" in robot.info.docs


def test_reset_embodiment_rejects_non_boolean_reference_mode() -> None:
    with pytest.raises(ValueError, match="expose_reference_images"):
        OCBenchHanoiResetEmbodiment(
            {0: np.zeros((1, 7), dtype=np.float32)},
            expose_reference_images=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "actions",
    [{}, {True: np.zeros((1, 7))}, {0: np.zeros((7,))}, {0: np.full((1, 7), np.nan)}],
)
def test_reset_embodiment_rejects_invalid_episode_actions(actions: dict[Any, Any]) -> None:
    with pytest.raises(ValueError):
        OCBenchHanoiResetEmbodiment(actions)


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


def test_reset_task_joins_split_to_episode_provenance(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    ledger_path = tmp_path / "ledger.json"
    split_path.write_text(
        json.dumps(
            {
                "dataset_repo": "owner/dataset",
                "dataset_revision": "revision",
                "train": [8],
                "validation": [7],
                "test": [9],
            }
        )
    )
    ledger_path.write_text(
        json.dumps(
            [
                {
                    "dataset_episode": 7,
                    "seed": 123,
                    "initial_world_sha256_reconstructed": "world-7",
                },
                {
                    "dataset_episode": 8,
                    "seed": 456,
                    "initial_world_sha256_reconstructed": "world-8",
                },
            ]
        )
    )

    task = hanoi_reset_task(str(split_path), str(ledger_path), num_scenes=1)

    assert task.name == "ocbench-hanoi-triple-task2-text-reset-30hz"
    assert task.scenes[0].instruction == RESET_INSTRUCTION
    assert task.scenes[0].metadata == {
        "reset_seed": 123,
        "initial_world_sha256": "world-7",
        "dataset_episode": 7,
    }
    assert [scorer.name for scorer in task.scorers] == ["success_at_end", "episode_length"]
    assert task.metadata["goal_mode"] == "text"
    assert task.metadata["protocol"] == "ocbench-hanoi-text-reset-30hz-v1"
    train = hanoi_reset_task(str(split_path), str(ledger_path), split="train", num_scenes=1)
    assert train.scenes[0].metadata["dataset_episode"] == 8
    reference = hanoi_reset_task(
        str(split_path), str(ledger_path), num_scenes=1, goal_mode="reference"
    )
    assert reference.name == "ocbench-hanoi-triple-task2-reference-reset-30hz"
    assert "reference images" in reference.scenes[0].instruction
    with pytest.raises(ValueError, match="goal_mode"):
        hanoi_reset_task(str(split_path), str(ledger_path), goal_mode="unknown")
    with pytest.raises(ValueError, match="missing"):
        hanoi_reset_task(str(split_path), str(ledger_path), split="test", num_scenes=1)


@pytest.mark.parametrize(
    ("position", "quat", "seed", "message"),
    [
        ([0, 0], [1, 0, 0, 0], [0] * 6, "expects XYZ"),
        ([0, 0, 0], [1, 0, 0], [0] * 6, "expects XYZ"),
        ([0, 0, 0], [1, 0, 0, 0], [0] * 7, "expects XYZ"),
        ([float("nan"), 0, 0], [1, 0, 0, 0], [0] * 6, "finite"),
        ([0, 0, 0], [0, 0, 0, 0], [0] * 6, "unit WXYZ"),
        ([0, 0, 0], [2, 0, 0, 0], [0] * 6, "unit WXYZ"),
        ([0, 0, 0], [1, 0, 0, 0], [100] * 6, "joint limits"),
    ],
)
def test_ik_rejects_bad_inputs_before_solver(
    simulator: FakeEnv, position: list[float], quat: list[float], seed: list[float], message: str
) -> None:
    robot = OCBenchEmbodiment()
    robot.reset(Scene("s", "restore"))
    with pytest.raises(ValueError, match=message):
        robot.solve_ik(np.array(position), np.array(quat), np.array(seed))
    assert simulator.command is None
    robot.close()


def test_ik_requires_initialized_robot() -> None:
    robot = OCBenchEmbodiment()
    with pytest.raises(RuntimeError, match="reset before"):
        robot.solve_ik(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(6))


def test_fk_observation_is_optional_and_contains_only_robot_pose(
    simulator: FakeEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = OCBenchEmbodiment(expose_robot_fk=True)
    poses: list[npt.NDArray[np.float64]] = []

    def fk(
        joints: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        poses.append(joints.copy())
        return np.array([0.4, 0.1, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(robot, "forward_kinematics", fk)
    observation = robot.reset(Scene("s", "restore"))
    assert set(observation.state) == {"joint_pos", "pinch_pos", "pinch_quat_wxyz"}
    assert observation.state["pinch_pos"].dtype == np.float32
    assert np.allclose(observation.state["pinch_pos"], [0.4, 0.1, 0.3])
    assert np.array_equal(poses[0], observation.state["joint_pos"][:6])
    assert observation.extra == {}
    assert robot.info.observation_space.state is not None
    assert {field.key for field in robot.info.observation_space.state.fields} == set(
        observation.state
    )
    assert robot.info.docs is not None and "pinch_quat_wxyz" in robot.info.docs
    robot.close()


@pytest.mark.parametrize("joints", [[0] * 5, [0] * 7, [float("nan")] * 6])
def test_fk_rejects_bad_joint_vectors_before_computing(
    simulator: FakeEnv,
    joints: list[float],
) -> None:
    robot = OCBenchEmbodiment()
    robot.reset(Scene("s", "restore"))
    with pytest.raises(ValueError, match="six finite"):
        robot.forward_kinematics(np.array(joints))
    assert simulator.command is None


def test_fk_requires_initialized_robot_and_boolean_option() -> None:
    robot = OCBenchEmbodiment()
    with pytest.raises(RuntimeError, match="reset before"):
        robot.forward_kinematics(np.zeros(6))
    with pytest.raises(ValueError, match="expose_robot_fk"):
        OCBenchEmbodiment(expose_robot_fk=1)  # type: ignore[arg-type]


def test_reset_docs_retain_gripper_timing_and_frame_conventions() -> None:
    robot = OCBenchHanoiResetEmbodiment({0: np.zeros((1, 7))})
    assert robot.info.docs is not None
    assert "0 means fully CLOSED" in robot.info.docs
    assert "1 means fully OPEN" in robot.info.docs
    assert "30 Hz" in robot.info.docs
    assert "world XYZ" in robot.info.docs and "WXYZ" in robot.info.docs


def test_tracking_and_failure_evidence_stay_out_of_policy_observations(simulator: FakeEnv) -> None:
    robot = OCBenchEmbodiment()
    robot.reset(Scene("s", "restore"))
    simulator.failure = True
    simulator.bad_floor_contact = True
    result = robot.step(Action(np.array([0.1, 0, 0, 0, 0, 0, 1.0])))
    assert result.info["failure_reasons"] == ["bad_floor_contact"]
    assert result.observation.extra == {}
    assert robot.control_diagnostics["last_joint_tracking_error_rad"][0] == pytest.approx(0.1)
    assert robot.control_diagnostics["last_gripper_tracking_error"] == pytest.approx(0.5)
    saved = robot.control_diagnostics
    saved["failure_reasons"].clear()
    assert robot.control_diagnostics["failure_reasons"] == ["bad_floor_contact"]
    robot.reset(Scene("next", "restore"))
    assert robot.control_diagnostics == {}
