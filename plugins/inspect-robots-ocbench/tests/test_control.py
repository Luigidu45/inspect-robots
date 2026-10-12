"""Exercise bounded physical settling without a simulator or an LLM service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots.controller import _INFER_KEY
from inspect_robots.errors import EmbodimentFault
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.types import Action, ActionChunk, Observation, StepResult
from inspect_robots_ocbench import OCBenchEmbodiment
from inspect_robots_ocbench.control import MotionFeedbackController, SettlingConfig


class ScriptedPolicy(PolicyBase):
    def __init__(self, target: float = 0.1, steps: int = 2) -> None:
        self.observations: list[Observation] = []
        self.info = PolicyInfo(name="scripted", action_space=OCBenchEmbodiment().info.action_space)
        self.target = np.array([target, 0, 0, 0, 0, 0, 0.4])
        self.steps = steps

    def act(self, observation: Observation) -> ActionChunk:
        self.observations.append(observation)
        return ActionChunk([Action(self.target.copy()) for _ in range(self.steps)])


def obs(q: float, approvals: list[dict[str, Any]] | None = None) -> Observation:
    return Observation(
        state={"joint_pos": np.array([q, 0, 0, 0, 0, 0, 0.4])},
        extra={"approvals": approvals or []},
    )


def test_settling_reaches_target_without_additional_policy_calls() -> None:
    policy = ScriptedPolicy()
    controller = MotionFeedbackController(SettlingConfig(max_steps=8, stable_steps=2))
    store: dict[str, Any] = {}
    first = controller.next_action(policy, obs(0), 0, store)
    assert first.meta["motion_requested_configuration"] == policy.target.tolist()
    controller.next_action(policy, obs(0.02), 1, store)
    hold = controller.next_action(policy, obs(0.05), 2, store)
    assert hold.meta["motion_phase"] == "settling"
    controller.next_action(policy, obs(0.1), 3, store)
    assert len(policy.observations) == 1
    controller.next_action(policy, obs(0.1), 4, store)
    report = controller.reports(store)[-1]
    assert len(policy.observations) == 2
    assert len(store[_INFER_KEY]) == 2
    assert report["execution"] == "complete"
    assert report["settling"] == "converged"
    assert report["planned_steps"] == report["played_steps"] == 2
    assert report["hold_steps"] == 2
    assert report["arm_error_rad"] == 0
    assert report["pinch_position_error_m"] is None


def test_blocked_target_times_out_and_reports_interventions() -> None:
    policy = ScriptedPolicy()
    controller = MotionFeedbackController(SettlingConfig(max_steps=3))
    store: dict[str, Any] = {}
    for t in range(6):
        controller.next_action(policy, obs(0, [{"t": 1, "detail": "delta_clamped"}]), t, store)
    assert len(policy.observations) == 2
    report = controller.reports(store)[-1]
    assert report["settling"] == "timeout"
    assert report["hold_steps"] == 3
    assert report["target_reached"] is False
    assert report["approver_modified_steps"] == 1
    assert report["approver_reasons"] == {"delta_clamped": 1}
    assert report["arm_error_rad"] == pytest.approx(0.1)
    assert report["outside_tolerance"] == ["arm"]
    assert len(report["summary"]) <= 320


def test_feedback_only_does_not_change_horizon_and_stores_are_trial_local() -> None:
    policy = ScriptedPolicy()
    controller = MotionFeedbackController()
    store: dict[str, Any] = {}
    for t in range(3):
        controller.next_action(policy, obs(0), t, store)
    report = controller.reports(store)[-1]
    assert report["hold_steps"] == 0 and report["settling"] == "disabled"
    assert report["gripper_error_normalized"] == 0
    controller.next_action(policy, obs(0), 0, {})
    assert "motion_feedback" not in policy.observations[-1].extra


def test_early_trial_end_records_partial_playout_without_inventing_steps() -> None:
    policy = ScriptedPolicy(steps=5)
    controller = MotionFeedbackController(SettlingConfig(max_steps=10))
    store: dict[str, Any] = {}
    controller.next_action(policy, obs(0), 0, store)
    report = controller.finish(store, obs(0.01), 1, "max_steps")
    assert report is not None
    assert report["execution"] == "partial"
    assert report["played_steps"] == 1
    assert report["hold_steps"] == 0
    assert report["end_reason"] == "max_steps"
    assert controller.finish(store, obs(0.01), 1, "max_steps") is None


def test_stop_is_forwarded_without_settling() -> None:
    class StopPolicy(ScriptedPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            self.observations.append(observation)
            return ActionChunk([Action(self.target, meta={"request_stop": True})])

    controller = MotionFeedbackController(SettlingConfig(max_steps=30))
    store: dict[str, Any] = {}
    action = controller.next_action(StopPolicy(), obs(0), 0, store)
    assert action.meta["request_stop"]
    assert controller.finish(store, obs(0), 1, "done") is None


def test_invalid_measured_state_cannot_be_reported_as_converged() -> None:
    controller = MotionFeedbackController()
    store: dict[str, Any] = {}
    policy = ScriptedPolicy(steps=1)
    controller.next_action(policy, obs(0), 0, store)
    with pytest.raises(EmbodimentFault):
        controller.next_action(policy, obs(float("nan")), 1, store)


@pytest.mark.parametrize(
    "pose",
    [
        (np.full(3, np.nan), np.array([1.0, 0, 0, 0])),
        (np.zeros(3), np.zeros(4)),
        (np.zeros(3), np.array([float("nan"), 0, 0, 0])),
        (np.zeros(2), np.array([1.0, 0, 0, 0])),
    ],
)
def test_invalid_fk_target_fails_before_issuing_motion(pose: Any) -> None:
    controller = MotionFeedbackController(forward_kinematics=lambda _joints: pose)
    with pytest.raises(EmbodimentFault, match="pinch pose"):
        controller.next_action(ScriptedPolicy(), obs(0), 0, {})


def test_final_sensor_fault_remains_unknown_without_a_second_hook_failure() -> None:
    controller = MotionFeedbackController()
    store: dict[str, Any] = {}
    action = controller.next_action(ScriptedPolicy(), obs(0), 0, store)
    record = TrialRecord(scene_id="fault", epoch=0, seed=0, status="error", error="sensor fault")
    record.steps.append(
        StepRecord(
            t=0,
            observation=obs(0),
            action=action,
            result=StepResult(observation=obs(float("nan"))),
        )
    )
    controller.finalize_trial(record)
    report = record.metadata["motion_feedback"][0]
    assert report["target_reached"] is None
    assert report["measurement_error"] is not None
    assert "measurement_unavailable" in report["summary"]
    assert record.error == "sensor fault"


def test_fk_errors_use_world_distance_and_sign_invariant_quaternions() -> None:
    def fk(joints: Any) -> tuple[Any, Any]:
        return np.array([joints[0], 0, 0]), np.array([1, 0, 0, 0])

    policy = ScriptedPolicy(steps=1)
    controller = MotionFeedbackController(forward_kinematics=fk)
    store: dict[str, Any] = {}
    controller.next_action(policy, obs(0), 0, store)
    current = obs(0.09)
    current = Observation(
        state={
            **current.state,
            "pinch_pos": np.array([0.09, 0, 0]),
            "pinch_quat_wxyz": np.array([-1, 0, 0, 0]),
        },
        extra={
            "execution_step": {
                "approved_configuration": [0.08] * 7,
                "normalized_delta": [0.5] * 7,
                "actuator_target_configuration": [0.07] * 7,
            }
        },
    )
    controller.next_action(policy, current, 1, store)
    report = controller.reports(store)[-1]
    assert report["pinch_position_error_m"] == pytest.approx(0.01)
    assert report["pinch_orientation_error_rad"] == 0
    assert report["approved_configuration"] == [0.08] * 7
    assert report["normalized_delta"] == [0.5] * 7
    assert report["actuator_target_configuration"] == [0.07] * 7
    assert report["target_reached"] is False
    assert "execution_step" not in policy.observations[-1].extra
    assert set(policy.observations[-1].extra["motion_feedback"]) == {"summary", "version"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": -1},
        {"max_steps": True},
        {"stable_steps": 0},
        {"joint_tolerance_rad": float("nan")},
        {"gripper_tolerance": 0},
    ],
)
def test_invalid_settling_settings_fail_before_execution(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SettlingConfig(**kwargs)
