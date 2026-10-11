"""Build held-out Hanoi tasks from the frozen, independently generated scene manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_robots.registry import task
from inspect_robots.scene import Scene
from inspect_robots.task import Task

RESET_INSTRUCTION = (
    "Reset the Tower of Hanoi scene to its canonical starting state. Move all three "
    "disks onto the middle peg as one legal stack: largest disk on the bottom, medium "
    "disk in the middle, and smallest disk on top. Never place a larger disk on a "
    "smaller disk. Manipulate only one disk at a time, release the final stack, and "
    "leave it stable. The robot arm may finish in any safe pose."
)

REFERENCE_RESET_INSTRUCTION = (
    "Reset the Tower of Hanoi scene to the goal arrangement shown in the reference "
    "images. Move all three disks onto the middle peg as one legal stack: largest disk "
    "on the bottom, medium disk in the middle, and smallest disk on top. Never place a "
    "larger disk on a smaller disk. Manipulate only one disk at a time, release the "
    "final stack, and leave it stable. The robot arm may finish in any safe pose."
)


@task("ocbench-hanoi")
def hanoi_task(
    manifest_path: str, split: str = "validation", num_scenes: int | None = None
) -> Task:
    """Schedule manifest scenes exactly once, without filtering unsuccessful resets."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    manifest = Path(manifest_path).read_bytes()
    rows = json.loads(manifest)[split]["scenes"]
    if not rows:
        raise ValueError("The scheduled scene manifest must not be empty")
    if num_scenes is not None:
        if (
            isinstance(num_scenes, bool)
            or not isinstance(num_scenes, int)
            or not 1 <= num_scenes <= len(rows)
        ):
            raise ValueError("num_scenes must be a positive integer within the manifest")
        rows = rows[:num_scenes]
    scenes = [
        Scene(
            id=row["scene_id"],
            instruction="Solve the randomized three-disk Tower of Hanoi.",
            metadata={
                "reset_seed": row["seed"],
                "initial_world_sha256": row["initial_world_sha256"],
            },
        )
        for row in rows
    ]
    return Task(
        name="ocbench-hanoi-triple-task2-rgb-proprio-30hz",
        scenes=scenes,
        scorer="success_at_end",
        max_steps=7000,
        metadata={
            "manifest_path": manifest_path,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "split": split,
            "scheduled_scenes": len(scenes),
            "protocol": "capx-hanoi-triple-task2-rgb-proprio-30hz-v1",
        },
    )


@task("ocbench-hanoi-reset")
def hanoi_reset_task(
    split_manifest_path: str,
    episode_ledger_path: str,
    split: str = "validation",
    num_scenes: int | None = None,
    goal_mode: str = "text",
    *,
    robot_fk: bool = False,
) -> Task:
    """Restore recorded solved episodes to their per-episode original stack."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if goal_mode not in {"text", "reference"}:
        raise ValueError("goal_mode must be text or reference")
    if not isinstance(robot_fk, bool):
        raise ValueError("robot_fk must be a bool")
    variant = "-fk" if robot_fk else ""
    split_bytes = Path(split_manifest_path).read_bytes()
    ledger_bytes = Path(episode_ledger_path).read_bytes()
    split_manifest = json.loads(split_bytes)
    episode_ids = split_manifest[split]
    ledger = {row["dataset_episode"]: row for row in json.loads(ledger_bytes)}
    if not episode_ids:
        raise ValueError("The reset split must not be empty")
    if num_scenes is not None:
        if (
            isinstance(num_scenes, bool)
            or not isinstance(num_scenes, int)
            or not 1 <= num_scenes <= len(episode_ids)
        ):
            raise ValueError("num_scenes must be a positive integer within the reset split")
        episode_ids = episode_ids[:num_scenes]

    scenes = []
    for episode in episode_ids:
        if episode not in ledger:
            raise ValueError(f"dataset episode {episode} is missing from the provenance ledger")
        row = ledger[episode]
        scenes.append(
            Scene(
                id=f"{split}-reset-{episode:03d}",
                instruction=(
                    RESET_INSTRUCTION if goal_mode == "text" else REFERENCE_RESET_INSTRUCTION
                ),
                metadata={
                    "reset_seed": row["seed"],
                    "initial_world_sha256": row["initial_world_sha256_reconstructed"],
                    "dataset_episode": episode,
                },
            )
        )
    return Task(
        name=f"ocbench-hanoi-triple-task2-{goal_mode}-reset-30hz{variant}",
        scenes=scenes,
        scorer=["success_at_end", "episode_length"],
        max_steps=7000,
        metadata={
            "split_manifest_path": split_manifest_path,
            "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "episode_ledger_path": episode_ledger_path,
            "episode_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "dataset_repo": split_manifest.get("dataset_repo"),
            "dataset_revision": split_manifest.get("dataset_revision"),
            "split": split,
            "scheduled_scenes": len(scenes),
            "goal_mode": goal_mode,
            "robot_fk": robot_fk,
            "protocol": f"ocbench-hanoi-{goal_mode}-reset-30hz{variant}-v1",
        },
    )
