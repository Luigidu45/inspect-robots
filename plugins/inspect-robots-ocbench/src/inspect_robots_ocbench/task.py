"""Build held-out Hanoi tasks from the frozen, independently generated scene manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_robots.registry import task
from inspect_robots.scene import Scene
from inspect_robots.task import Task


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
