"""Audit the installed reset runtime without making paid LLM requests.

Exit 0 means local checks and external configuration are ready. Exit 2 means
local checks passed but credentials or the SAM3 endpoint remain pending.
Exit 1 means an environment, dataset, simulation or service check failed.
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
from validate_assistance import _ik_code


def audit(dataset_root: Path, sam3_url: str | None, api_key_env: str) -> dict[str, Any]:
    """Check real scene preparation, kinematics, plugins and external prerequisites."""
    experiment = Path(__file__).resolve().parent
    protocol = json.loads((experiment / "protocol.json").read_text())
    names = ("numpy", "mujoco", "dm-control", "gymnasium", "pyarrow", "ocbench")
    versions = {name: version(name) for name in names}
    expected = protocol["source"]["versions"]
    for name in names:
        if versions[name] != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]}, installed {versions[name]}")
    policy_names = {entry.name for entry in entry_points(group="inspect_robots.policies")}
    task_names = {entry.name for entry in entry_points(group="inspect_robots.tasks")}
    assert {"agent", "capx"} <= policy_names
    assert "ocbench-hanoi-reset" in task_names
    task = hanoi_reset_task(
        str(experiment / "split_manifest.json"),
        str(experiment / "episode_provenance.json"),
        "validation",
        1,
    )
    scene = task.scenes[0]
    episode = int(scene.metadata["dataset_episode"])
    split_manifest = json.loads((experiment / "split_manifest.json").read_text())
    evaluation_episodes = sorted(split_manifest["validation"] + split_manifest["test"])
    actions = load_episode_actions(dataset_root, evaluation_episodes)
    dataset_info = json.loads((dataset_root / "meta" / "info.json").read_text())
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    parquet_rows = sum(
        parquet.ParquetFile(path).metadata.num_rows
        for path in (dataset_root / "data").rglob("*.parquet")
    )
    if parquet_rows != protocol["dataset"]["frames"]:
        raise ValueError(f"Expected {protocol['dataset']['frames']} frames, found {parquet_rows}")
    if dataset_info["total_episodes"] != protocol["dataset"]["episodes"]:
        raise ValueError("Dataset episode count differs from the fixed protocol")
    robot = OCBenchHanoiResetEmbodiment(actions)
    configurations = []
    try:
        observation = robot.reset(scene)
        assert set(observation.images) == {"front", "side", "wrist"}
        assert set(observation.extra) == {"reference_images"}
        assert all(image.shape == (224, 224, 3) for image in observation.images.values())
        assert observation.state["joint_pos"].shape == (7,)
        assert np.all(np.isfinite(observation.state["joint_pos"]))
        assert robot.info.control_hz == 30
        assert abs(robot._env.unwrapped.model.opt.timestep - 1 / 600) < 1e-12
        _ik_code(robot)  # Only robot geometry; checks three poses and rejection.
        for backend, assistance in [
            ("agent", None),
            ("capx", "none"),
            ("capx", "ik"),
            ("capx", "sam3_ik"),
        ]:
            options = argparse.Namespace(
                policy=backend,
                assistance=assistance,
                model="gpt-6-astra",
                base_url="https://api.openai.com/v1",
                api_key_env=api_key_env,
                wire="responses",
                max_llm_calls=8,
                effort="low",
                images="always",
                prior_learnings=None,
                sam3_url=sam3_url or "http://pending-sam3.invalid:8114",
                sam3_timeout_s=30.0,
            )
            policy = run_reset._build_policy(options, robot)
            try:
                policy.bind(robot.info)
                policy.reset(scene)
                if backend == "capx":
                    assert policy._servers is None and policy._pyroki_client is None
                    helpers = list(policy.config.enabled_helpers)
                    assert "plan_grasp" not in helpers
                    assert ("segment" in helpers) == (assistance == "sam3_ik")
                    assert ("solve_ik" in helpers) == (assistance in ("ik", "sam3_ik"))
                else:
                    helpers = ["move_joints", "done", "give_up"]
                configurations.append(
                    {
                        "policy": backend,
                        "assistance": assistance,
                        "helpers": helpers,
                        "construction_and_binding": "passed",
                    }
                )
            finally:
                close = getattr(policy, "close", None)
                if callable(close):
                    close()
        provenance = dict(robot.reset_provenance)
    finally:
        robot.close()
    sam3 = (
        run_reset._check_sam3(argparse.Namespace(sam3_url=sam3_url, sam3_timeout_s=30.0))
        if sam3_url
        else {"status": "pending", "reason": "SAM3 URL not configured"}
    )
    pending = []
    if not os.environ.get(api_key_env):
        pending.append(f"Set {api_key_env} in the environment; its value is never recorded")
    if not sam3_url:
        pending.append("Provide a real SAM3 endpoint for sam3_ik")
    modules = (
        "inspect_robots",
        "inspect_robots_agent",
        "inspect_robots_capx",
        "inspect_robots_ocbench",
        "ocbench",
    )
    return {
        "local_runtime_ready": True,
        "external_configuration_ready": not pending,
        "python": sys.executable,
        "versions": versions,
        "module_paths": {name: importlib.import_module(name).__file__ for name in modules},
        "dataset_root": str(dataset_root.resolve()),
        "dataset_frames": parquet_rows,
        "dataset_episodes": dataset_info["total_episodes"],
        "evaluation_episodes_validated": len(evaluation_episodes),
        "prepared_episode": episode,
        "reset_preparation": provenance,
        "control_hz": 30,
        "physics_hz": 600,
        "local_ik": "passed",
        "configurations": configurations,
        "sam3": sam3,
        "pending": pending,
        "live_llm_evaluated": False,
        "autonomous_reset_evaluated": False,
    }


def main() -> int:
    """Persist a readiness report whose exit code distinguishes pending setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam3-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.dataset_root, args.sam3_url, args.api_key_env)
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
