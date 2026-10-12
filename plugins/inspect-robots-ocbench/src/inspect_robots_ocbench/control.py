"""Measure joint-target chunks and optionally settle them using bounded control steps.

No task geometry, object state, simulator stepping, or provider calls belong here.
The rollout executes every returned action through its usual approver and budget.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from inspect_robots.controller import _INFER_KEY
from inspect_robots.errors import EmbodimentFault
from inspect_robots.policy import Policy
from inspect_robots.types import Action, Observation

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

_KEY = "ocbench:motion_feedback"
FK = Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]


@dataclass(frozen=True)
class SettlingConfig:
    """Bound extra steps and require separate arm, pinch and opening tolerances."""

    max_steps: int = 0
    stable_steps: int = 3
    joint_tolerance_rad: float = 0.01
    gripper_tolerance: float = 0.02
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        for name, minimum in (("max_steps", 0), ("stable_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "joint_tolerance_rad",
            "gripper_tolerance",
            "position_tolerance_m",
            "orientation_tolerance_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class _Motion:
    start: int
    planned: int
    target: npt.NDArray[np.float64]
    buffer: deque[Action]
    target_pose: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
    streak: int = 0


class MotionFeedbackController:
    """Play full chunks, then optionally hold their endpoint without another inference.

    Convergence means target agreement for consecutive measured samples, not a
    successful grasp or stable objects. ``max_steps=0`` preserves the chunk horizon.
    Only seven-dimensional absolute arm-plus-gripper targets are supported.
    """

    def __init__(
        self, settling: SettlingConfig | None = None, *, forward_kinematics: FK | None = None
    ) -> None:
        self.settling = settling or SettlingConfig()
        self._fk = forward_kinematics
        self._trial_store: dict[str, Any] | None = None

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Use observed convergence to schedule bounded holds and the next inference."""
        self._trial_store = store
        state = store.setdefault(_KEY, {"motion": None, "reports": []})
        motion: _Motion | None = state["motion"]
        if motion is not None:
            if motion.buffer:
                return self._tag(motion.buffer.popleft(), "playout")
            reached = self._errors(motion, observation)["target_reached"]
            motion.streak = motion.streak + 1 if reached else 0
            holds = t - motion.start - motion.planned
            if (
                self.settling.max_steps
                and holds < self.settling.max_steps
                and (motion.streak < self.settling.stable_steps)
            ):
                return self._tag(Action(motion.target.copy()), "settling")
            report = self._report(motion, observation, t)
            state["reports"].append(report)
            state["motion"] = None
            # Both policy formats receive the same bounded summary. Per-step
            # command arrays stay in simulator logs, including CaP-X's obs.
            extra = {
                key: value for key, value in observation.extra.items() if key != "execution_step"
            }
            observation = replace(
                observation,
                extra={
                    **extra,
                    "motion_feedback": {
                        "summary": report["summary"],
                        "version": report["version"],
                    },
                },
            )

        chunk = policy.act(observation)
        store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(chunk)))
        if any(action.meta.get("request_stop") for action in chunk.actions):
            # A stop should be its own single-step chunk, never followed by holds.
            if len(chunk) != 1:
                raise ValueError("stop requests must occupy a single-action chunk")
            return chunk.actions[0]
        target = np.asarray(chunk.actions[-1].data, dtype=np.float64).copy()
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("motion feedback requires seven finite absolute joint targets")
        pose = self._validated_pose(*self._fk(target[:6])) if self._fk is not None else None
        motion = _Motion(t, len(chunk), target, deque(chunk.actions), pose)
        state["motion"] = motion
        return self._tag(motion.buffer.popleft(), "playout")

    @staticmethod
    def _validated_pose(
        position: Any, quaternion: Any
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.all(np.isfinite(pos))
            or not np.all(np.isfinite(quat))
            or not math.isfinite(norm)
            or norm == 0
        ):
            raise EmbodimentFault("motion feedback requires a finite world pinch pose")
        return pos.copy(), quat / norm

    @staticmethod
    def _tag(action: Action, phase: str) -> Action:
        return replace(
            action,
            meta={
                **action.meta,
                "motion_phase": phase,
                "motion_requested_configuration": np.asarray(action.data).tolist(),
            },
        )

    def _errors(self, motion: _Motion, observation: Observation) -> dict[str, Any]:
        measured = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise EmbodimentFault("motion feedback requires seven finite measured joint positions")
        arm = float(np.max(np.abs(motion.target[:6] - measured[:6])))
        grip = float(abs(motion.target[6] - measured[6]))
        position: float | None = None
        orientation: float | None = None
        if motion.target_pose is not None:
            pos, quat = self._validated_pose(
                observation.state["pinch_pos"], observation.state["pinch_quat_wxyz"]
            )
            target_pos, target_quat = motion.target_pose
            position = float(np.linalg.norm(target_pos - pos))
            dot = abs(float(np.dot(target_quat, quat)))
            orientation = float(2 * np.arccos(np.clip(dot, 0, 1)))
        config = self.settling
        outside = [
            name
            for name, value, tolerance in (
                ("arm", arm, config.joint_tolerance_rad),
                ("gripper", grip, config.gripper_tolerance),
                ("pinch_position", position, config.position_tolerance_m),
                ("pinch_orientation", orientation, config.orientation_tolerance_rad),
            )
            if value is not None and value > tolerance
        ]
        return {
            "arm_error_rad": arm,
            "gripper_error_normalized": grip,
            "gripper_opening_measured": float(measured[6]),
            "pinch_position_error_m": position,
            "pinch_orientation_error_rad": orientation,
            "target_reached": not outside,
            "outside_tolerance": outside,
            "observed_configuration": measured.tolist(),
        }

    def _report(
        self, motion: _Motion, observation: Observation, t: int, end_reason: str | None = None
    ) -> dict[str, Any]:
        elapsed = max(0, t - motion.start)
        played = min(elapsed, motion.planned)
        held = max(0, elapsed - motion.planned)
        measurement_error: str | None = None
        try:
            errors = self._errors(motion, observation)
        except EmbodimentFault as exc:
            if end_reason is None:
                raise
            # Preserve the original sensor fault at trial end; missing measurements
            # must not create a second hook failure or imply target agreement.
            measurement_error = str(exc)
            errors = dict.fromkeys(
                (
                    "arm_error_rad",
                    "gripper_error_normalized",
                    "gripper_opening_measured",
                    "pinch_position_error_m",
                    "pinch_orientation_error_rad",
                    "observed_configuration",
                    "target_reached",
                )
            )
            errors["outside_tolerance"] = ["measurement_unavailable"]
        settling = (
            "disabled"
            if not self.settling.max_steps
            else "interrupted"
            if end_reason is not None
            else "converged"
            if motion.streak >= self.settling.stable_steps
            else "timeout"
        )
        approvals = [
            a
            for a in observation.extra.get("approvals", [])
            if isinstance(a, dict) and motion.start <= a.get("t", -1) < t
        ]
        trace = observation.extra.get("execution_step", {})
        report = {
            "version": "motion-feedback-v1",
            "execution": "complete" if played == motion.planned else "partial",
            "settling": settling,
            "end_reason": end_reason,
            "measurement_error": measurement_error,
            "planned_steps": motion.planned,
            "played_steps": played,
            "hold_steps": held,
            "total_steps": elapsed,
            "simulation_time_s": elapsed / 30,
            "requested_configuration": motion.target.tolist(),
            "last_step_requested_configuration": trace.get("requested_configuration"),
            "approved_configuration": trace.get("approved_configuration"),
            "normalized_delta": trace.get("normalized_delta"),
            "actuator_target_configuration": trace.get("actuator_target_configuration"),
            "approver_modified_steps": len(approvals),
            "approver_reasons": dict(
                Counter(str(a.get("detail") or "modified") for a in approvals)
            ),
            **errors,
        }
        if measurement_error is not None:
            report["summary"] = (
                f"playout={report['execution']} {played}/{motion.planned}; "
                f"hold={held}/{self.settling.max_steps} {settling}; "
                "measurement_unavailable; target_reached=unknown"
            )
            return report
        spatial = (
            ""
            if errors["pinch_position_error_m"] is None
            else (
                f"; pinch_e={errors['pinch_position_error_m']:.4g}m/"
                f"{errors['pinch_orientation_error_rad']:.4g}rad"
            )
        )
        report["summary"] = (
            f"playout={report['execution']} {played}/{motion.planned}; "
            f"hold={held}/{self.settling.max_steps} "
            f"{settling}; target_reached={errors['target_reached']}; "
            f"outside={','.join(errors['outside_tolerance']) or 'none'}; "
            f"arm_e={errors['arm_error_rad']:.4g}rad; "
            f"grip_e={errors['gripper_error_normalized']:.4g} "
            f"opening={errors['gripper_opening_measured']:.4g}; "
            f"approver_modified={len(approvals)}{spatial}"
        )
        return report

    def finish(
        self,
        store: dict[str, Any],
        observation: Observation,
        executed_steps: int,
        end_reason: str | None,
    ) -> dict[str, Any] | None:
        """Record final or interrupted execution without claiming unexecuted holds."""
        state = store.get(_KEY)
        if state is None or state["motion"] is None:
            return None
        report = self._report(state["motion"], observation, executed_steps, end_reason)
        state["reports"].append(report)
        state["motion"] = None
        return report

    @staticmethod
    def reports(store: dict[str, Any]) -> list[dict[str, Any]]:
        """Return reports retained in this trial's controller store."""
        return deepcopy(store.get(_KEY, {}).get("reports", []))

    def finalize_trial(self, record: TrialRecord) -> None:
        """Persist audit reports and release the store after a sequential runner trial."""
        store = self._trial_store
        self._trial_store = None
        if store is None:
            record.metadata["motion_feedback"] = []
            return
        if record.steps:
            observation = record.steps[-1].result.observation
            approvals = [
                {"t": event.t, "detail": event.data.get("detail")}
                for event in record.events
                if event.kind == "approval" and event.data.get("modified")
            ]
            self.finish(
                store,
                replace(observation, extra={**observation.extra, "approvals": approvals}),
                len(record.steps),
                record.termination_reason or record.status,
            )
        record.metadata["motion_feedback"] = self.reports(store)
