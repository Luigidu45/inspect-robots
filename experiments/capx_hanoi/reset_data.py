"""Load only the recorded action streams needed to prepare Hanoi reset scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def load_episode_actions(
    dataset_root: Path, episode_ids: list[int]
) -> dict[int, npt.NDArray[np.float32]]:
    """Read ordered absolute targets for selected LeRobot episodes."""
    if not episode_ids or any(
        isinstance(episode, bool) or not isinstance(episode, int) or episode < 0
        for episode in episode_ids
    ):
        raise ValueError("episode_ids must contain non-negative integers")
    info_path = dataset_root / "meta" / "info.json"
    data_root = dataset_root / "data"
    if not info_path.is_file() or not data_root.is_dir():
        raise ValueError(f"{dataset_root} is not a complete LeRobot dataset root")
    info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("fps") != 30:
        raise ValueError(f"reset protocol requires a 30 Hz dataset, got {info.get('fps')!r}")
    feature = info.get("features", {}).get("action", {})
    if feature.get("shape") != [7]:
        raise ValueError(f"reset protocol requires seven-dimensional actions, got {feature!r}")

    try:
        import pyarrow.dataset as pads  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading the Hanoi dataset requires pyarrow; install it in the OCBench runtime"
        ) from exc
    files = sorted(data_root.rglob("*.parquet"))
    if not files:
        raise ValueError(f"no Parquet data files found below {data_root}")
    dataset = pads.dataset([str(path) for path in files], format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=pads.field("episode_index").isin(episode_ids),
    )
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    result: dict[int, npt.NDArray[np.float32]] = {}
    for episode in episode_ids:
        indices = np.flatnonzero(episodes == episode)
        if len(indices) == 0:
            raise ValueError(f"dataset episode {episode} has no action records")
        order = indices[np.argsort(frames[indices])]
        expected = np.arange(len(order))
        if not np.array_equal(frames[order], expected):
            raise ValueError(f"dataset episode {episode} has missing or duplicate frame indices")
        selected = actions[order]
        if selected.shape != (len(order), 7) or not np.all(np.isfinite(selected)):
            raise ValueError(f"dataset episode {episode} contains invalid actions")
        result[episode] = selected
    return result
