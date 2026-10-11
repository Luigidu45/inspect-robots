"""Preflight the native Inspect Robots agent reset path without an LLM request.

Exit 0 means the local runtime and API-key configuration are ready. Exit 2
means local checks passed but the key is absent. Exit 1 means that the dataset,
simulator, manifests, or native-agent plugin is unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from importlib.metadata import entry_points, version
from pathlib import Path
from typing import Any

import numpy as np
import run_reset
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment, hanoi_reset_task
from reset_data import load_episode_actions


def audit(dataset_root: Path, api_key_env: str) -> dict[str, Any]:
    """Check the exact dependency chain used by ``run_reset --policy agent``."""
    experiment = Path(__file__).resolve().parent
    config = experiment / "config"
    protocol = json.loads((config / "protocol.json").read_text())
    names = ("numpy", "mujoco", "dm-control", "gymnasium", "pyarrow", "ocbench")
    versions = {name: version(name) for name in names}
    expected = protocol["source"]["versions"]
    for name in names:
        if versions[name] != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]}, installed {versions[name]}")

    policy_names = {entry.name for entry in entry_points(group="inspect_robots.policies")}
    task_names = {entry.name for entry in entry_points(group="inspect_robots.tasks")}
    if "agent" not in policy_names:
        raise ValueError("native policy 'agent' is not installed")
    if "ocbench-hanoi-reset" not in task_names:
        raise ValueError("task 'ocbench-hanoi-reset' is not installed")

    task = hanoi_reset_task(
        str(config / "split_manifest.json"),
        str(config / "episode_provenance.json"),
        "validation",
        1,
    )
    scene = task.scenes[0]
    episode = int(scene.metadata["dataset_episode"])
    actions = load_episode_actions(dataset_root, [episode])
    dataset_info = json.loads((dataset_root / "meta" / "info.json").read_text())
    if dataset_info["total_episodes"] != protocol["dataset"]["episodes"]:
        raise ValueError("Dataset episode count differs from the fixed protocol")

    robot = OCBenchHanoiResetEmbodiment(actions)
    try:
        observation = robot.reset(scene)
        if set(observation.images) != {"front", "side", "wrist"}:
            raise ValueError("reset observation does not expose the three expected cameras")
        if observation.extra:
            raise ValueError("text-goal reset unexpectedly exposes privileged or goal data")
        if not all(image.shape == (224, 224, 3) for image in observation.images.values()):
            raise ValueError("camera shape differs from the frozen protocol")
        if (
            observation.state["joint_pos"].shape != (7,)
            or not np.isfinite(observation.state["joint_pos"]).all()
        ):
            raise ValueError("joint observation is not a finite seven-dimensional vector")
        if robot.info.control_hz != 30:
            raise ValueError("control rate differs from the frozen 30 Hz protocol")
        if abs(robot._env.unwrapped.model.opt.timestep - 1 / 600) >= 1e-12:
            raise ValueError("physics timestep differs from the frozen 600 Hz protocol")

        options = argparse.Namespace(
            policy="agent",
            model="gpt-6-astra",
            base_url="https://api.openai.com/v1",
            api_key_env=api_key_env,
            wire="responses",
            max_llm_calls=64,
            max_output_tokens=None,
            max_cost_usd=None,
            effort="medium",
            images="always",
            live_transcript=True,
            prior_learnings=None,
        )
        policy = run_reset._build_policy(options, robot)
        try:
            policy.bind(robot.info)
            policy.reset(scene)
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        provenance = dict(robot.reset_provenance)
    finally:
        robot.close()

    pending = []
    if not os.environ.get(api_key_env):
        pending.append(f"Set {api_key_env} in the environment; its value is never recorded")
    modules = ("inspect_robots", "inspect_robots_agent", "inspect_robots_ocbench", "ocbench")
    return {
        "local_runtime_ready": True,
        "external_configuration_ready": not pending,
        "python": sys.executable,
        "versions": versions,
        "module_paths": {name: importlib.import_module(name).__file__ for name in modules},
        "dataset_root": str(dataset_root.resolve()),
        "dataset_episodes": dataset_info["total_episodes"],
        "prepared_episode": episode,
        "reset_preparation": provenance,
        "control_hz": 30,
        "physics_hz": 600,
        "policy": "agent",
        "tools": ["move_joints", "done", "give_up"],
        "pending": pending,
        "live_llm_evaluated": False,
        "autonomous_reset_evaluated": False,
    }


def main() -> int:
    """Persist a native-runtime report without requesting an LLM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.dataset_root, args.api_key_env)
        status = 0 if report["external_configuration_ready"] else 2
    except Exception as error:
        report = {
            "local_runtime_ready": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "live_llm_evaluated": False,
        }
        status = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
