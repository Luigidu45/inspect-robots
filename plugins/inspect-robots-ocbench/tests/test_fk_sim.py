"""Check robot-only FK against physical MuJoCo frames when the runtime is installed."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation
from inspect_robots_ocbench import OCBenchEmbodiment

pytest.importorskip("mujoco")
lie = pytest.importorskip("ocbench.lie")


class _NoRenderRobot(OCBenchEmbodiment):
    def _observe(self) -> Observation:
        return Observation(state={"joint_pos": self._joint_state()})


@pytest.mark.sim
def test_command_trace_matches_actual_mujoco_actuator_targets() -> None:
    robot = _NoRenderRobot()
    try:
        robot.reset(Scene("command-check", "check control targets"), seed=123)
        raw: Any = robot._env.unwrapped
        target = robot._joint_state().astype(np.float64)
        target[5] += 0.5
        result = robot.step(
            Action(target, meta={"motion_requested_configuration": target.tolist()})
        )
        trace = result.info["execution_command"]
        assert trace["normalized_delta"][5] == 1
        assert np.allclose(
            trace["actuator_target_configuration"][:6],
            raw.data.ctrl[raw._arm_actuator_ids],
            rtol=0,
            atol=1e-10,
        )
        assert np.allclose(
            trace["actuator_target_configuration"][6],
            raw.data.ctrl[raw._gripper_actuator_ids] / 255,
            rtol=0,
            atol=1e-10,
        )
        assert np.array_equal(trace["observed_configuration"], robot._joint_state())
    finally:
        robot.close()


@pytest.mark.sim
def test_fk_matches_pinch_frame_without_changing_world_or_ik_state() -> None:
    robot = _NoRenderRobot()
    try:
        robot.reset(Scene("fk-check", "check robot kinematics"), seed=123)
        raw: Any = robot._env.unwrapped
        before = [raw.data.qpos.copy(), raw.data.qvel.copy(), raw.data.ctrl.copy()]
        before_time = raw.data.time
        ik_before = raw._ik._data.qpos.copy()
        joints = robot._joint_state()[:6].astype(np.float64)
        position, quat = robot.forward_kinematics(joints)
        assert np.array_equal(raw._ik._data.qpos, ik_before)
        actual_position = raw.data.site_xpos[raw._pinch_site_id]
        actual_rotation = raw.data.site_xmat[raw._pinch_site_id].reshape(3, 3)
        assert np.allclose(position, actual_position, atol=1e-6, rtol=0)
        assert np.allclose(lie.SO3(quat).as_matrix(), actual_rotation, atol=1e-6, rtol=0)
        assert abs(np.linalg.norm(quat) - 1) < 1e-10
        for offset in (0.0, 0.01, -0.02):
            candidate = joints.copy()
            candidate[5] += offset
            target_pos, target_quat = robot.forward_kinematics(candidate)
            solution = robot.solve_ik(target_pos, target_quat, joints)
            achieved_pos, achieved_quat = robot.forward_kinematics(solution)
            assert np.linalg.norm(achieved_pos - target_pos) <= 1e-3
            assert (
                np.linalg.norm((lie.SO3(target_quat).inverse() @ lie.SO3(achieved_quat)).log())
                <= 1e-2
            )
        assert raw.data.time == before_time
        for expected, actual in zip(
            before, [raw.data.qpos, raw.data.qvel, raw.data.ctrl], strict=True
        ):
            assert np.array_equal(expected, actual)
        # FK uses its own scratch data, independent of the solver's workspace.
        ik_after = raw._ik._data.qpos.copy()
        robot.forward_kinematics(joints + 0.001)
        assert np.array_equal(raw._ik._data.qpos, ik_after)
    finally:
        robot.close()
