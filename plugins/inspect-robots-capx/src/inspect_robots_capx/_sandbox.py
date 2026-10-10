"""In-process execution boundary for model-generated Python code.

Model output executes with the evaluator's process privileges. This module is
an integration surface, not a security sandbox; untrusted models require an
external container or equivalent isolation.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.types import Observation
from inspect_robots_capx._assistance import IKSolver, assistance_helpers
from inspect_robots_capx._motion import MotionQueue
from inspect_robots_capx._servers import CapxServerClients, PyrokiClient, Sam3Client


@dataclass(frozen=True)
class ExecutionResult:
    """Captured feedback from one model-code turn without propagating its exception."""

    stdout: str
    stderr: str
    raised: bool


class _TurnView(dict[str, Any]):
    """Turn-scoped ``obs`` mapping that resolves zero-arg callable values on access.

    Embodiments may provide bulk ``observation.extra`` entries (depth,
    intrinsics, extrinsics) as zero-argument callables to keep trial records
    small; resolving them here keeps the documented ``obs[...]`` idioms
    working unchanged for both forms.
    """

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        return value() if callable(value) else value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the resolved value for ``key``, or ``default`` when absent.

        Membership decides absence, so a ``KeyError`` raised inside an
        embodiment-provided thunk propagates instead of masquerading as a
        missing key.
        """
        if key not in self:
            return default
        return self[key]


class CodeSandbox:
    """Persistent per-trial Python namespace with observation-bound robot helpers."""

    def __init__(
        self,
        *,
        servers: CapxServerClients | None,
        motion: MotionQueue,
        camera: str,
        state_key: str,
        depth_key: str = "depth",
        intrinsics_key: str = "intrinsics",
        extrinsics_key: str = "extrinsics",
        assistance: str = "default",
        ik_solver: IKSolver | None = None,
        sam3_client: Sam3Client | None = None,
        pyroki_client: PyrokiClient | None = None,
    ) -> None:
        self._helpers = assistance_helpers(assistance)
        self._ik_solver = ik_solver
        self._sam3_client = sam3_client
        self._pyroki_client = pyroki_client
        self._servers = servers
        self._motion = motion
        self._camera = camera
        self._state_key = state_key
        self._depth_key = depth_key
        self._intrinsics_key = intrinsics_key
        self._extrinsics_key = extrinsics_key
        self._observation: Observation | None = None
        self._namespace: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Start a fresh trial namespace with only the documented helpers bound."""
        self._observation = None
        self._motion.reset()
        helpers = {
            "segment": self._segment,
            "plan_grasp": self._plan_grasp,
            "solve_ik": self._solve_ik,
            "move_to_joints": self._motion.move_to_joints,
            "move_to_configuration": self._motion.move_to_configuration,
            "open_gripper": self._motion.open_gripper,
            "close_gripper": self._motion.close_gripper,
        }
        self._namespace = {"__builtins__": __builtins__}
        self._namespace.update({name: helpers[name] for name in self._helpers})

    def set_observation(self, observation: Observation) -> None:
        """Expose one turn's observation and reseed motion from its full state field."""
        try:
            state = observation.state[self._state_key]
        except KeyError as exc:
            raise ValueError(
                f"observation.state is missing bound proprioceptive key {self._state_key!r}"
            ) from exc
        self._observation = observation
        self._motion.begin_turn(state)
        obs = _TurnView(
            {
                "images": observation.images,
                "state": observation.state,
            }
        )
        obs.update(observation.extra)
        self._namespace["obs"] = obs

    def execute(self, code: str) -> ExecutionResult:
        """Execute one turn and return stdout, stderr, and whether it raised."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        raised = False
        cursor = self._motion.cursor
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(code, self._namespace, self._namespace)
                if self._motion.limit_exceeded:
                    raise ValueError(
                        "Motion queue limit exceeded, even though code caught the error"
                    )
            except Exception:
                raised = True
                traceback.print_exc()
                if cursor is not None:
                    self._motion.begin_turn(cursor)
                else:
                    self._motion.reset()
                print("All queued motions from failed code were discarded.", file=stderr)
        return ExecutionResult(stdout=stdout.getvalue(), stderr=stderr.getvalue(), raised=raised)

    def _require_observation(self) -> Observation:
        if self._observation is None:
            raise RuntimeError("no observation is bound for this code turn")
        return self._observation

    def _extra_array(self, key: str) -> npt.NDArray[Any]:
        observation = self._require_observation()
        if key not in observation.extra:
            raise KeyError(
                f"observation.extra is missing {key!r}; embodiments must provide "
                f"extra[{key!r}] as an array or a zero-argument callable returning one"
            )
        value = observation.extra[key]
        if callable(value):
            value = value()
        return np.asarray(value)

    def _segment(self, text: str) -> list[dict[str, Any]]:
        observation = self._require_observation()
        try:
            image = observation.images[self._camera]
        except KeyError as exc:
            raise KeyError(f"observation has no configured camera {self._camera!r}") from exc
        sam3 = self._sam3_client or (self._servers.sam3 if self._servers is not None else None)
        if sam3 is None:
            raise RuntimeError("SAM3 is unavailable in this assistance profile")
        results = sam3.segment(image, text)
        if any(result["mask"].shape != image.shape[:2] for result in results):
            raise ValueError("SAM3 mask size does not match the selected RGB camera")
        return results

    def _plan_grasp(self, mask: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        if self._servers is None:
            raise RuntimeError("grasp planning is unavailable in this assistance profile")
        return self._servers.graspnet.plan(
            self._extra_array(self._depth_key),
            self._extra_array(self._intrinsics_key),
            np.asarray(mask),
        )

    def _solve_ik(
        self,
        position: npt.NDArray[Any],
        quaternion_wxyz: npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        if self._ik_solver is not None:
            joints = np.asarray(
                self._ik_solver(
                    np.asarray(position, dtype=np.float64),
                    np.asarray(quaternion_wxyz, dtype=np.float64),
                    self._motion.arm_cursor,
                ),
                dtype=np.float64,
            )
            if joints.shape != (self._motion.arm_dof,) or not np.all(np.isfinite(joints)):
                raise ValueError("IK solver must return a finite arm-joint vector")
            return joints.copy()
        pyroki = self._pyroki_client or (
            self._servers.pyroki if self._servers is not None else None
        )
        if pyroki is None:
            raise RuntimeError("IK is unavailable in this assistance profile")
        return pyroki.solve_ik(
            position,
            quaternion_wxyz,
            self._motion.arm_dof,
        )
