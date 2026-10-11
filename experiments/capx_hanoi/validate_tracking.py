"""Measure physical target tracking during Hanoi reset preparation without an LLM."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment, hanoi_reset_task
from reset_data import load_episode_actions

from inspect_robots.types import Action


def measure(dataset_root: Path) -> dict[str, Any]:
    """Compare one-step and repeated targets using identical recorded initial scenes."""
    experiment = Path(__file__).resolve().parent
    config = experiment / "config"
    task = hanoi_reset_task(
        str(config / "split_manifest.json"),
        str(config / "episode_provenance.json"),
        "validation",
        1,
    )
    scene = task.scenes[0]
    episode = int(scene.metadata["dataset_episode"])
    robot = OCBenchHanoiResetEmbodiment(load_episode_actions(dataset_root, [episode]))
    trials = []
    try:
        for steps in (1, 3, 30):
            observation = robot.reset(scene)
            target = observation.state["joint_pos"].astype(np.float64)
            target[5] += 0.01
            samples = []
            start_time = robot._env.unwrapped.data.time
            for _ in range(steps):
                result = robot.step(Action(target.copy()))
                samples.append(
                    {
                        "joint_error_rad": result.info["joint_tracking_error_rad"],
                        "gripper_error": result.info["gripper_tracking_error"],
                    }
                )
                if result.terminated:
                    break
            elapsed = robot._env.unwrapped.data.time - start_time
            assert abs(elapsed - len(samples) / 30) < 1e-8
            assert set(result.observation.extra) == {"reference_images"}
            trials.append(
                {
                    "requested_steps": steps,
                    "executed_steps": len(samples),
                    "simulation_time_s": elapsed,
                    "samples": samples,
                    "diagnostics": robot.control_diagnostics,
                    "termination_reason": result.termination_reason,
                }
            )
    finally:
        robot.close()
    assert all(not t["diagnostics"]["failure_reasons"] for t in trials)
    assert all(t["requested_steps"] == t["executed_steps"] for t in trials)
    initial_wrist_error = abs(trials[0]["samples"][-1]["joint_error_rad"][5])
    final_wrist_error = abs(trials[-1]["samples"][-1]["joint_error_rad"][5])
    assert final_wrist_error < initial_wrist_error and final_wrist_error < 1e-3
    return {
        "kind": "deterministic-physical-tracking",
        "llm_calls": 0,
        "api_requests": 0,
        "autonomous_reset_evaluated": False,
        "tracking_check_passed": True,
        "target_change": "wrist_3 +0.01 rad, observed gripper held",
        "dataset_episode": episode,
        "trials": trials,
    }


def main() -> int:
    """Save tracking evidence with outbound socket connections disabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Tracking validation forbids network connections")

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
                    {
                        "steps": t["executed_steps"],
                        "final_joint_error": t["samples"][-1]["joint_error_rad"],
                        "failure": t["diagnostics"]["failure_reasons"],
                    }
                    for t in report["trials"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
