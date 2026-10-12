"""Translate absolute UR5e targets into OCBench's measured-state delta commands."""

from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy
from dataclasses import replace
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
REFERENCE_IMAGES_KEY = "reference_images"
RESET_STABILITY_STEPS = 5
_ROBOT_DOCS = (
    "UR5e with six absolute arm targets in radians, followed by normalized gripper opening. "
    "Gripper 0 means fully CLOSED; 1 means fully OPEN. Measured opening can lag or be "
    "blocked by object contact. Control runs at 30 Hz (one action is 1/30 second), with "
    "600 Hz physics. Per-action target deltas are limited to "
    "[0.18, 0.18, 0.18, 0.36, 0.36, 0.36] rad and 0.12 gripper opening. "
    "These are command limits, not a guarantee of physical convergence. A requested "
    "configuration is not evidence that the robot reached it; inspect measured joint_pos "
    "and camera images after execution. RGB and joint_pos are exposed, without "
    "depth, calibrated extrinsics or privileged object poses. When IK assistance is enabled, "
    "its target is the "
    "Robotiq PINCH POINT in world XYZ meters with a unit WXYZ quaternion. Generic UR5 "
    "DH coordinates or flange coordinates must not be treated as this world pinch frame."
)
_FK_DOCS = (
    " Robot FK is enabled: state['pinch_pos'] is the current Robotiq pinch XYZ in "
    "world meters, and state['pinch_quat_wxyz'] is its unit WXYZ quaternion. These "
    "are computed from measured arm joints and robot geometry, without object poses. "
    "They use the same pinch frame as solve_ik. Use this current pose as a spatial "
    "reference instead of searching inverse IK to reconstruct it. FK does not locate "
    "disks or pegs, establish a grasp, or validate a trajectory."
)


class OCBenchEmbodiment:
    """Expose the frozen 30 Hz, non-lite Hanoi task as RGB plus seven joint positions.

    Construction imports no OCBench or MuJoCo. Install the reviewed OCBench
    checkout in the evaluation interpreter. Manifest reset seeds override the
    framework's trial seed; a supplied world hash must match before any action.
    Privileged task geometry is never included in policy observations.
    """

    def __init__(self, *, expose_robot_fk: bool = False) -> None:
        if not isinstance(expose_robot_fk, bool):
            raise ValueError("expose_robot_fk must be a bool")
        self._env: Any = None
        self._fk_data: Any = None
        self._expose_robot_fk = expose_robot_fk
        self._instruction = ""
        self._steps = 0
        self._control_diagnostics: dict[str, Any] = {}
        self._last_execution_step: dict[str, Any] = {}
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
                state=StateSpec(
                    (StateField("joint_pos", (7,), "rad+normalized", "float32"),)
                    + (
                        (
                            StateField("pinch_pos", (3,), "m", "float32"),
                            StateField("pinch_quat_wxyz", (4,), "unit quaternion WXYZ", "float32"),
                        )
                        if expose_robot_fk
                        else ()
                    )
                ),
            ),
            control_hz=30.0,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, PRIVILEGED_SUCCESS, RENDERABLE}),
            environment_id=ENV_ID,
            environment_revision="95e3d4e16756f864b58e32a6a661286288ed3be8:30hz:20substeps",
            docs=(
                _ROBOT_DOCS + " Solve the three-disk Tower of "
                "Hanoi starting on the middle peg, ending on either outer peg."
                + (_FK_DOCS if expose_robot_fk else "")
            ),
        )

    def _make_kwargs(self) -> dict[str, Any]:
        """Return simulator construction arguments for this evaluation mode."""
        return {
            "control_timestep": 1 / 30,
            "physics_timestep": 1 / 600,
            "width": 224,
            "height": 224,
        }

    def _requires_exact_world_hash(self) -> bool:
        """Require bitwise reset identity for the standard evaluation protocol."""
        return True

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
                **self._make_kwargs(),
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
        expected_hash = scene.metadata.get("initial_world_sha256", actual_hash)
        hash_matches = expected_hash == actual_hash
        if not hash_matches and self._requires_exact_world_hash():
            raise ValueError("OCBench reset world hash differs from the scheduled manifest")
        self.reset_provenance = {
            "reset_seed": reset_seed,
            "framework_trial_seed": seed,
            "initial_world_sha256": actual_hash,
            "expected_initial_world_sha256": expected_hash,
            "initial_world_hash_matches": hash_matches,
        }
        self._instruction = scene.instruction
        self._steps = 0
        self._control_diagnostics = {}
        self._last_execution_step = {}
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Execute one 1/30-second control step, bounding targets and normalized deltas."""
        if self._env is None:
            raise RuntimeError("Call reset before step")
        normalized = self._normalized_action(action.data)
        command = self._execution_command(action, normalized)
        _, reward, terminated, truncated, info = self._env.step(normalized)
        self._steps += 1
        success = bool(info["success"] and info["healthy"] and not info["failure"])
        failure = bool(info["failure"] or not info["healthy"])
        diagnostics = self._record_control(action, info, command)
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
                **diagnostics,
            },
        )

    @property
    def control_diagnostics(self) -> dict[str, Any]:
        """Return evaluation-only tracking evidence, excluding object geometry."""
        return deepcopy(self._control_diagnostics)

    def _execution_command(
        self, action: Action, normalized: npt.NDArray[np.float32]
    ) -> dict[str, Any]:
        """Snapshot pre-review intent and the bounded actuator target before stepping."""
        raw = self._env.unwrapped
        current = np.concatenate(
            [
                raw.data.qpos[raw._arm_joint_ids],
                [raw.data.qpos[raw._gripper_opening_joint_id] / 0.8],
            ]
        )
        target = current + normalized * raw._joint_action_delta
        target = np.clip(target, self.info.action_space.low, self.info.action_space.high)
        return {
            "requested_configuration": action.meta.get("motion_requested_configuration"),
            "approved_configuration": np.asarray(action.data, dtype=np.float64).tolist(),
            "normalized_delta": normalized.tolist(),
            "actuator_target_configuration": target.tolist(),
            "phase": action.meta.get("motion_phase"),
        }

    def _record_control(
        self, action: Action, info: dict[str, Any], command: dict[str, Any]
    ) -> dict[str, Any]:
        observed = self._joint_state().astype(np.float64)
        approved = np.asarray(action.data, dtype=np.float64)
        error = approved - observed
        self._last_execution_step = {
            **deepcopy(command),
            "observed_configuration": observed.tolist(),
        }
        reasons = []
        if not info["healthy"]:
            reasons.append("outside_workspace")
        if info["failure"] or not info["healthy"]:
            for method, reason in (
                ("_has_bad_floor_contact", "bad_floor_contact"),
                ("_has_illegal_stack", "illegal_stack"),
            ):
                check = getattr(self._env.unwrapped, method, None)
                if callable(check) and check():
                    reasons.append(reason)
            if not reasons:
                reasons.append("unspecified_physical_failure")
        previous = self._control_diagnostics
        self._control_diagnostics = {
            "control_steps": self._steps,
            "control_hz": 30,
            "max_joint_tracking_error_rad": max(
                float(np.max(np.abs(error[:6]))),
                previous.get("max_joint_tracking_error_rad", 0.0),
            ),
            "max_gripper_tracking_error": max(
                float(abs(error[6])), previous.get("max_gripper_tracking_error", 0.0)
            ),
            "last_joint_tracking_error_rad": error[:6].tolist(),
            "last_gripper_tracking_error": float(error[6]),
            "last_approved_configuration": approved.tolist(),
            "last_execution_command": deepcopy(self._last_execution_step),
            "last_observed_configuration": observed.tolist(),
            "failure_reasons": sorted(set(previous.get("failure_reasons", [])) | set(reasons)),
        }
        return {
            "joint_tracking_error_rad": error[:6].tolist(),
            "gripper_tracking_error": float(error[6]),
            "failure_reasons": reasons,
            "execution_command": deepcopy(self._last_execution_step),
        }

    def _normalized_action(self, data: Any) -> npt.NDArray[np.float32]:
        """Convert one finite absolute target into OCBench's bounded delta action."""
        target = np.asarray(data, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("OCBench expects seven finite absolute target values")
        box = self.info.action_space
        target = np.clip(target, box.low, box.high)
        # Match the exporter/replay's float32 measured state, including clipped
        # gripper opening. OCBench applies the resulting delta to its raw qpos.
        current = self._joint_state()
        normalized = np.asarray(np.clip((target - current) / DELTA_SCALE, -1, 1), dtype=np.float32)
        return normalized

    def close(self) -> None:
        """Release graphics resources; repeated calls are harmless."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self._fk_data = None

    def forward_kinematics(
        self,
        joint_pos: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return world XYZ and unit WXYZ for the robot's Robotiq pinch frame.

        Only the supplied six arm joints, isolated robot model, and fixed tool
        transform are used. Neither the physical world nor the IK workspace is
        modified. This computes a pose, not reachability or collision safety.
        """
        if self._env is None:
            raise RuntimeError("Call reset before computing FK")
        joints = np.asarray(joint_pos, dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("FK expects six finite arm joints")
        raw = self._env.unwrapped
        model = raw._ik._model
        mujoco = importlib.import_module("mujoco")
        lie = importlib.import_module("ocbench.lie")
        if self._fk_data is None:
            self._fk_data = mujoco.MjData(model)
        self._fk_data.qpos[:] = joints
        mujoco.mj_kinematics(model, self._fk_data)
        site_id = model.site("attachment_site").id
        attach = lie.SE3.from_rotation_and_translation(
            lie.SO3.from_matrix(self._fk_data.site_xmat[site_id].reshape(3, 3)),
            self._fk_data.site_xpos[site_id].copy(),
        )
        pinch = attach @ raw._T_pa.inverse()
        return (
            np.asarray(pinch.translation(), dtype=np.float64).copy(),
            np.asarray(pinch.rotation().wxyz, dtype=np.float64).copy(),
        )

    def solve_ik(
        self,
        position: npt.NDArray[np.float64],
        quaternion_wxyz: npt.NDArray[np.float64],
        joint_seed: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solve a world-frame Robotiq pinch pose using isolated UR5e kinematics.

        Uses only robot geometry and the supplied six-joint seed. No object
        poses are read and no simulator step or physical state write occurs.
        Rejects nonconverged or out-of-bounds solutions instead of clipping them.
        This solver does not perform collision or trajectory planning.
        """
        if self._env is None:
            raise RuntimeError("Call reset before solving IK")
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64)
        seed = np.asarray(joint_seed, dtype=np.float64)
        if pos.shape != (3,) or quat.shape != (4,) or seed.shape != (6,):
            raise ValueError("IK expects XYZ (3,), WXYZ (4,), and six arm joints (6,)")
        if not all(np.all(np.isfinite(value)) for value in (pos, quat, seed)):
            raise ValueError("IK inputs must be finite")
        norm = float(np.linalg.norm(quat))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("IK orientation must be a unit WXYZ quaternion")
        box = self.info.action_space
        assert box.low is not None and box.high is not None
        if np.any(seed < box.low[:6]) or np.any(seed > box.high[:6]):
            raise ValueError("IK seed is outside UR5e joint limits")

        raw = self._env.unwrapped
        lie = importlib.import_module("ocbench.lie")
        mujoco = importlib.import_module("mujoco")
        target = lie.SE3.from_rotation_and_translation(lie.SO3(quat / norm), pos)
        attach_target = target @ raw._T_pa
        solver = raw._ik
        joints = np.asarray(
            solver.solve(
                pos=attach_target.translation(),
                quat=attach_target.rotation().wxyz,
                curr_qpos=seed.copy(),
                max_iters=100,
            ),
            dtype=np.float64,
        )
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("IK returned an invalid UR5e solution")
        if np.any(joints < box.low[:6]) or np.any(joints > box.high[:6]):
            raise ValueError("IK solution is outside UR5e joint limits")
        # Refresh FK after the final solver update; the scratch model contains
        # only the robot, not the current Hanoi objects or peg layout.
        mujoco.mj_forward(solver._model, solver._data)
        site_id = solver._model.site("attachment_site").id
        achieved_attach = lie.SE3.from_rotation_and_translation(
            lie.SO3.from_matrix(solver._data.site_xmat[site_id].reshape(3, 3)),
            solver._data.site_xpos[site_id].copy(),
        )
        achieved = achieved_attach @ raw._T_pa.inverse()
        position_error = float(np.linalg.norm(achieved.translation() - pos))
        rotation_error = float(
            np.linalg.norm((target.rotation().inverse() @ achieved.rotation()).log())
        )
        if (
            not np.isfinite(position_error)
            or not np.isfinite(rotation_error)
            or position_error > 1e-3
            or rotation_error > 1e-2
        ):
            raise ValueError(
                f"IK did not converge: position_error_m={position_error:.6g}, "
                f"orientation_error_rad={rotation_error:.6g}"
            )
        return joints.copy()

    def _joint_state(self) -> npt.NDArray[np.float32]:
        raw = self._env.unwrapped
        opening = np.clip(raw.data.qpos[raw._gripper_opening_joint_id] / 0.8, 0, 1)
        return np.asarray(
            np.concatenate([raw.data.qpos[raw._arm_joint_ids], [opening]]), dtype=np.float32
        )

    def _observe(self) -> Observation:
        timestamp = self._steps / 30
        state = {"joint_pos": self._joint_state()}
        if self._expose_robot_fk:
            position, quaternion = self.forward_kinematics(
                state["joint_pos"][:6].astype(np.float64)
            )
            state.update(
                pinch_pos=position.astype(np.float32),
                pinch_quat_wxyz=quaternion.astype(np.float32),
            )
        return Observation(
            images={
                name: self._env.unwrapped.render(camera=camera).copy()
                for name, camera in _CAMERAS.items()
            },
            state=state,
            instruction=self._instruction,
            state_time=timestamp,
            image_times=dict.fromkeys(_CAMERAS, timestamp),
            extra={"execution_step": deepcopy(self._last_execution_step)}
            if self._last_execution_step.get("requested_configuration") is not None
            else {},
        )


@embodiment("ocbench")
def ocbench_embodiment() -> OCBenchEmbodiment:
    """Construct the dataset-matched Hanoi embodiment without loading the simulator."""
    return OCBenchEmbodiment()


# Kept on the registered factory so ``inspect-robots doctor --embodiment ocbench``
# can diagnose a fresh environment without importing MuJoCo or constructing an
# environment. The command installs the workspace-locked extra, not an ad-hoc
# requirements file.
ocbench_embodiment.RUNTIME_REQUIREMENTS = {  # type: ignore[attr-defined]
    "ocbench": "uv sync --all-packages --extra ocbench",
    "pyarrow": "uv sync --all-packages --extra ocbench",
}


class OCBenchHanoiResetEmbodiment(OCBenchEmbodiment):
    """Start from recorded solved episodes and score restoration of the original stack.

    Episode actions are replayed privately during ``reset``. The policy sees the
    resulting scene and robot proprioception. Goal images captured before replay
    are optional. The policy never receives object poses or demonstration actions.
    """

    def __init__(
        self,
        episode_actions: dict[int, npt.NDArray[np.floating[Any]]],
        *,
        expose_reference_images: bool = False,
        expose_robot_fk: bool = False,
    ) -> None:
        super().__init__(expose_robot_fk=expose_robot_fk)
        if not episode_actions:
            raise ValueError("episode_actions must contain at least one episode")
        self._episode_actions: dict[int, npt.NDArray[np.float32]] = {}
        for episode, actions in episode_actions.items():
            if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
                raise ValueError("episode action keys must be non-negative integers")
            array = np.asarray(actions, dtype=np.float32)
            if array.ndim != 2 or array.shape[1:] != (7,) or len(array) == 0:
                raise ValueError(f"episode {episode} actions must have shape (N, 7), N > 0")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"episode {episode} actions contain non-finite values")
            self._episode_actions[episode] = array.copy()
        if not isinstance(expose_reference_images, bool):
            raise ValueError("expose_reference_images must be a bool")
        self._expose_reference_images = expose_reference_images
        self._reference_images: dict[str, npt.NDArray[np.uint8]] = {}
        self._reset_success = False
        self._restore_streak = 0
        self.info = replace(
            self.info,
            name="ocbench-hanoi-reset",
            docs=(
                _ROBOT_DOCS + " Restore the three-disk stack on its original middle peg. "
                "The largest disk belongs on the bottom and the "
                "smallest on top. Release the disks and leave them stable. The arm "
                "may finish in any safe pose."
                + (
                    " Goal images are included in the observation."
                    if expose_reference_images
                    else " No goal image is provided; follow the textual task definition."
                )
                + (_FK_DOCS if expose_robot_fk else "")
            ),
        )

    def _make_kwargs(self) -> dict[str, Any]:
        """Disable source-task success termination and cover setup plus reset horizons."""
        return {
            **super()._make_kwargs(),
            "terminate_at_success": False,
            "max_episode_steps": 14_001,
        }

    def _requires_exact_world_hash(self) -> bool:
        """Use successful full replay as the reset protocol's cross-runtime identity gate."""
        return False

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Replay one solved episode, then expose the disturbed reset scene."""
        self._reference_images = {}
        initial = super().reset(scene, seed=seed)
        episode = scene.metadata.get("dataset_episode")
        if isinstance(episode, bool) or not isinstance(episode, int):
            raise ValueError("reset scenes require an integer dataset_episode")
        try:
            actions = self._episode_actions[episode]
        except KeyError as exc:
            raise ValueError(f"no actions loaded for dataset episode {episode}") from exc

        if self._expose_reference_images:
            self._reference_images = {name: image.copy() for name, image in initial.images.items()}
        final_info: dict[str, Any] | None = None
        for target in actions:
            _, _, _, truncated, info = self._env.step(self._normalized_action(target))
            final_info = info
            if bool(info["failure"] or not info["healthy"]):
                raise ValueError(f"dataset episode {episode} became unhealthy during replay")
            if truncated:
                raise ValueError(f"dataset episode {episode} exceeded the setup horizon")
        raw = self._env.unwrapped
        if final_info is None or not bool(final_info["success"]):
            raise ValueError(f"dataset episode {episode} does not end in OCBench success")
        if bool(raw._is_tower_on_peg(raw.cur_task_info["init_peg"])):
            raise ValueError(f"dataset episode {episode} did not move the stack from its origin")

        self._steps = 0
        self._reset_success = False
        self._restore_streak = 0
        self.reset_provenance = {
            **self.reset_provenance,
            "dataset_episode": episode,
            "setup_replay_steps": len(actions),
            "setup_source_success": True,
        }
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance reset control and terminate only on restoration or physical failure."""
        if self._env is None:
            raise RuntimeError("Call reset before step")
        normalized = self._normalized_action(action.data)
        command = self._execution_command(action, normalized)
        _, _, _, truncated, info = self._env.step(normalized)
        self._steps += 1
        raw = self._env.unwrapped
        failure = bool(info["failure"] or not info["healthy"])
        diagnostics = self._record_control(action, info, command)
        restored = bool(raw._is_tower_on_peg(raw.cur_task_info["init_peg"]))
        released = bool(self._joint_state()[-1] >= 0.5)
        self._restore_streak = self._restore_streak + 1 if restored and released else 0
        self._reset_success = bool(self._restore_streak >= RESET_STABILITY_STEPS and not failure)
        return StepResult(
            observation=self._observe(),
            reward=1.0 if self._reset_success else 0.0,
            terminated=bool(self._reset_success or failure),
            termination_reason="success" if self._reset_success else "failure" if failure else None,
            truncated=bool(truncated),
            info={
                "success": self._reset_success,
                "reset_success": self._reset_success,
                "source_task_success": bool(info["success"]),
                "reset_stability_steps": self._restore_streak,
                "failure": failure,
                "healthy": bool(info["healthy"]),
                "reset_provenance": dict(self.reset_provenance),
                **diagnostics,
            },
        )

    def _observe(self) -> Observation:
        """Optionally attach immutable goal images without exposing object state."""
        observation = super()._observe()
        if not self._reference_images:
            return observation
        references = {name: image.copy() for name, image in self._reference_images.items()}
        return replace(observation, extra={REFERENCE_IMAGES_KEY: references})
