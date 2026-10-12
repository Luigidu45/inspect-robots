"""Compare physical feedback variants using mocked agent replies and no API access.

The scripted wrist offset probes generic control tracking. It is not a Hanoi
reset strategy and is never installed as a production policy or capability.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_capx import CapxPolicy
from inspect_robots_ocbench import (
    MotionFeedbackController,
    OCBenchHanoiResetEmbodiment,
    SettlingConfig,
    hanoi_reset_task,
)
from reset_data import load_episode_actions

from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.controller import DefaultController
from inspect_robots.logging import NullSink
from inspect_robots.rollout import rollout
from inspect_robots.types import Observation


class _ProprioceptiveRobot(OCBenchHanoiResetEmbodiment):
    def _observe(self) -> Observation:
        state = {"joint_pos": self._joint_state()}
        position, quat = self.forward_kinematics(state["joint_pos"][:6].astype(np.float64))
        state.update(pinch_pos=position.astype(np.float32), pinch_quat_wxyz=quat.astype(np.float32))
        return Observation(
            state=state,
            instruction=self._instruction,
            state_time=self._steps / 30,
            extra={"execution_step": dict(self._last_execution_step)},
        )


def _transport(
    mode: str, wrist_target: float, requests: list[dict[str, Any]]
) -> httpx.MockTransport:
    """Return deterministic replies bound to one probe, without network access."""

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        output: list[dict[str, Any]]
        if mode == "agent":
            output = [
                {
                    "type": "function_call",
                    "status": "completed",
                    "id": "fc",
                    "call_id": f"call{len(requests)}",
                    "name": "move_joints" if first else "give_up",
                    "arguments": json.dumps(
                        {
                            "targets": {"wrist_3": wrist_target},
                            "note": "Generic tracking probe",
                        }
                        if first
                        else {"reason": "Probe finished"}
                    ),
                }
            ]
        else:
            code = "q = obs['state']['joint_pos'].copy()\nq[5] += 0.05\nmove_to_configuration(q)"
            if mode == "capx-ik":
                code = (
                    "solve_ik(obs['state']['pinch_pos'], obs['state']['pinch_quat_wxyz'])\n" + code
                )
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": code if first else "GIVE_UP"}],
                }
            ]
        return httpx.Response(200, json={"status": "completed", "output": output})

    return httpx.MockTransport(respond)


def measure(dataset_root: Path) -> dict[str, Any]:
    """Exercise all three policy interfaces on identical measured-state wrist targets."""
    config = Path(__file__).resolve().parent / "config"
    task = hanoi_reset_task(
        str(config / "split_manifest.json"),
        str(config / "episode_provenance.json"),
        "validation",
        1,
    )
    scene = task.scenes[0]
    episode = int(scene.metadata["dataset_episode"])
    robot = _ProprioceptiveRobot(
        load_episode_actions(dataset_root, [episode]), expose_robot_fk=True
    )
    rows = []
    try:
        for mode in ("agent", "capx-none", "capx-ik"):
            for variant, hold in (("baseline", 0), ("feedback", 0), ("feedback-settle", 30)):
                initial = robot.reset(scene)
                target = initial.state["joint_pos"].astype(np.float64)
                target[5] += 0.05
                requests: list[dict[str, Any]] = []

                common: dict[str, Any] = {
                    "model": "scripted-test",
                    "base_url": "http://mock/v1",
                    "wire": "responses",
                    "max_llm_calls": 2,
                    "effort": "low",
                    "transport": _transport(mode, float(target[5]), requests),
                    "env": {},
                }
                policy: Any
                if mode == "agent":
                    policy = LLMAgentPolicy(**common, images="always")
                else:
                    policy = CapxPolicy(
                        **common,
                        assistance="ik" if mode == "capx-ik" else "none",
                        ik_solver=robot.solve_ik if mode == "capx-ik" else None,
                        camera="front",
                        max_queued_steps=90,
                    )
                controller = (
                    MotionFeedbackController(
                        SettlingConfig(max_steps=hold), forward_kinematics=robot.forward_kinematics
                    )
                    if variant != "baseline"
                    else None
                )
                try:
                    policy.bind(robot.info)
                    record = rollout(
                        policy,
                        robot,
                        scene,
                        controller=controller or DefaultController(),
                        approver=ChainApprover(
                            ClampApprover(robot.info.action_space),
                            DeltaLimitApprover(robot.info.action_space),
                        ),
                        sink=NullSink(),
                        max_steps=120,
                        seed=0,
                        epoch=0,
                    )
                    if controller is not None:
                        controller.finalize_trial(record)
                    report = record.metadata.get("motion_feedback", [None])[0]
                    # The last action is the explicit stop, not a tracking sample.
                    sample = record.steps[-2].result.observation
                    error = float(np.max(np.abs(sample.state["joint_pos"][:6] - target[:6])))
                    feedback_texts = [
                        part["text"]
                        for message in requests[-1]["input"]
                        for part in message.get("content", [])
                        if isinstance(part, dict)
                        and str(part.get("text", "")).startswith("Motion feedback: ")
                    ]
                    assert len(requests) == 2
                    assert len(feedback_texts) == (0 if variant == "baseline" else 1)
                    rows.append(
                        {
                            "mode": mode,
                            "variant": variant,
                            "provider_api_requests": 0,
                            "mocked_model_calls": len(requests),
                            "control_steps": len(record.steps),
                            "final_arm_error_rad": error,
                            "motion_report": report,
                            "feedback_characters": sum(map(len, feedback_texts)),
                        }
                    )
                finally:
                    close = getattr(policy, "close", None)
                    if callable(close):
                        close()
        for mode in ("agent", "capx-none", "capx-ik"):
            group = [row for row in rows if row["mode"] == mode]
            assert group[0]["control_steps"] == group[1]["control_steps"]
            assert group[2]["final_arm_error_rad"] < group[0]["final_arm_error_rad"]
            assert group[2]["motion_report"]["hold_steps"] <= 30
    finally:
        robot.close()
    return {
        "api_requests": 0,
        "autonomous_reset_evaluated": False,
        "dataset_episode": episode,
        "target_change": "wrist_3 +0.05 rad",
        "observations": "proprioception and FK; no rendered images",
        "trials": rows,
    }


def main() -> int:
    """Save physical tracking evidence with outbound socket connections disabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Feedback validation forbids network connections")

    socket.socket.connect = no_network  # type: ignore[method-assign]
    socket.socket.connect_ex = no_network  # type: ignore[method-assign]
    report = measure(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "api_requests": 0,
                "trials": [
                    {key: value for key, value in row.items() if key != "motion_report"}
                    for row in report["trials"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
