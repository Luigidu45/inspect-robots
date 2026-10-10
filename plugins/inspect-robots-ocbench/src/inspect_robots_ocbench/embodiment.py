"""Translate absolute UR5e targets into OCBench's measured-state delta commands."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.embodiment import (
    PRIVILEGED_SUCCESS,
    RENDERABLE,
    RESETTABLE,
    SEEDABLE,
    EmbodimentInfo,
)
from inspect_robots.registry import embodiment
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

ENV_ID = "hanoi-cpu-triple-task2-v0"
DELTA_SCALE = np.array([0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12])
_ARM_HIGH = np.array([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831])
_CAMERAS = {"front": "front", "side": "side", "wrist": "ur5e/wrist"}


class OCBenchEmbodiment:
    """Expose the frozen 30 Hz, non-lite Hanoi task as RGB plus seven joint positions.

    Construction imports no OCBench or MuJoCo. Install the reviewed OCBench
    checkout in the evaluation interpreter. Manifest reset seeds override the
    framework's trial seed; a supplied world hash must match before any action.
    Privileged task geometry is never included in policy observations.
    """

    def __init__(self) -> None:
        self._env: Any = None
        self._instruction = ""
        self._steps = 0
        self.reset_provenance: dict[str, Any] = {}
        self.info = EmbodimentInfo(
            name="ocbench",
            action_space=Box(
                shape=(7,),
                low=np.concatenate([-_ARM_HIGH, [0.0]]),
                high=np.concatenate([_ARM_HIGH, [1.0]]),
                semantics=ActionSemantics(
                    "joint_pos",
                    gripper="continuous",
                    dim_labels=(
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow",
                        "wrist_1",
                        "wrist_2",
                        "wrist_3",
                        "gripper",
                    ),
                    max_step=tuple(float(value) for value in DELTA_SCALE),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 224, 224) for name in _CAMERAS),
                state=StateSpec((StateField("joint_pos", (7,), "rad+normalized", "float32"),)),
            ),
            control_hz=30.0,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, PRIVILEGED_SUCCESS, RENDERABLE}),
            environment_id=ENV_ID,
            environment_revision="95e3d4e16756f864b58e32a6a661286288ed3be8:30hz:20substeps",
            docs=(
                "UR5e, six arm targets in radians then gripper opening: 0 closed, 1 open. "
                "Use move_to_configuration for simultaneous arm and pinching motions. "
                "Only RGB and joint_pos are available; depth, calibrated extrinsics and "
                "privileged object poses are unavailable. Solve the three-disk Tower of "
                "Hanoi starting on the middle peg, ending on either outer peg."
            ),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset at the scheduled manifest seed and verify simulator bounds and world hash."""
        reset_seed = scene.metadata.get("reset_seed", seed if seed is not None else 0)
        if (
            isinstance(reset_seed, bool)
            or not isinstance(reset_seed, int)
            or not 0 <= reset_seed < 2**32
        ):
            raise ValueError("reset_seed must be an integer in [0, 2**32)")
        if self._env is None:
            try:
                ocbench = importlib.import_module("ocbench")
            except ImportError as exc:
                raise ImportError(
                    "Install the reviewed OCBench checkout in this interpreter first."
                ) from exc
            self._env = ocbench.make(
                ENV_ID,
                control_timestep=1 / 30,
                physics_timestep=1 / 600,
                width=224,
                height=224,
            )
        _, info = self._env.reset(seed=reset_seed)
        raw = self._env.unwrapped
        ranges = raw.model.actuator_ctrlrange[raw._arm_actuator_ids]
        if not np.allclose(ranges, np.column_stack([-_ARM_HIGH, _ARM_HIGH]), rtol=0, atol=1e-10):
            raise ValueError("OCBench arm actuator limits differ from the frozen UR5e contract")
        if not np.allclose(raw._joint_action_delta, DELTA_SCALE, rtol=0, atol=1e-12):
            raise ValueError("OCBench action delta scale differs from the frozen protocol")
        world = raw.data.qpos.tobytes() + raw.data.qvel.tobytes() + info["dynamics_info"].tobytes()
        actual_hash = hashlib.sha256(world).hexdigest()
        if scene.metadata.get("initial_world_sha256", actual_hash) != actual_hash:
            raise ValueError("OCBench reset world hash differs from the scheduled manifest")
        self.reset_provenance = {
            "reset_seed": reset_seed,
            "framework_trial_seed": seed,
            "initial_world_sha256": actual_hash,
        }
        self._instruction = scene.instruction
        self._steps = 0
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Execute one 1/30-second control step, bounding targets and normalized deltas."""
        if self._env is None:
            raise RuntimeError("Call reset before step")
        target = np.asarray(action.data, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("OCBench expects seven finite absolute target values")
        box = self.info.action_space
        target = np.clip(target, box.low, box.high)
        # Match the exporter/replay's float32 measured state, including clipped
        # gripper opening. OCBench applies the resulting delta to its raw qpos.
        current = self._joint_state()
        normalized = np.clip((target - current) / DELTA_SCALE, -1, 1).astype(np.float32)
        _, reward, terminated, truncated, info = self._env.step(normalized)
        self._steps += 1
        success = bool(info["success"] and info["healthy"] and not info["failure"])
        failure = bool(info["failure"] or not info["healthy"])
        return StepResult(
            observation=self._observe(),
            reward=float(reward),
            terminated=bool(terminated or success or failure),
            termination_reason="success" if success else "failure" if failure else None,
            truncated=bool(truncated),
            info={
                "success": success,
                "failure": failure,
                "healthy": bool(info["healthy"]),
                "reset_provenance": dict(self.reset_provenance),
            },
        )

    def close(self) -> None:
        """Release graphics resources; repeated calls are harmless."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def _joint_state(self) -> npt.NDArray[np.float32]:
        raw = self._env.unwrapped
        opening = np.clip(raw.data.qpos[raw._gripper_opening_joint_id] / 0.8, 0, 1)
        return np.concatenate([raw.data.qpos[raw._arm_joint_ids], [opening]]).astype(np.float32)

    def _observe(self) -> Observation:
        timestamp = self._steps / 30
        return Observation(
            images={
                name: self._env.unwrapped.render(camera=camera).copy()
                for name, camera in _CAMERAS.items()
            },
            state={"joint_pos": self._joint_state()},
            instruction=self._instruction,
            state_time=timestamp,
            image_times=dict.fromkeys(_CAMERAS, timestamp),
        )


@embodiment("ocbench")
def ocbench_embodiment() -> OCBenchEmbodiment:
    """Construct the dataset-matched Hanoi embodiment without loading the simulator."""
    return OCBenchEmbodiment()
