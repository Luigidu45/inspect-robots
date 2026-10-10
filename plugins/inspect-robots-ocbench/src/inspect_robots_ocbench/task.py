"""Build held-out Hanoi tasks from the frozen, independently generated scene manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_robots.registry import task
from inspect_robots.scene import Scene
from inspect_robots.task import Task

RESET_INSTRUCTION = (
    "Restore the three disks to the original middle-peg stack shown in the reference "
    "images using an efficient sequence of safe moves. Place the largest disk on the "
    "bottom, the medium disk in the middle, and the smallest disk on top. Release the "
    "disks and leave the stack stable. The robot arm may finish in any safe pose."
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
) -> Task:
    """Restore recorded solved episodes to their per-episode original stack."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
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
                instruction=RESET_INSTRUCTION,
                metadata={
                    "reset_seed": row["seed"],
                    "initial_world_sha256": row["initial_world_sha256_reconstructed"],
                    "dataset_episode": episode,
                },
            )
        )
    return Task(
        name="ocbench-hanoi-triple-task2-reference-reset-30hz",
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
            "protocol": "capx-hanoi-reference-reset-30hz-v1",
        },
    )
