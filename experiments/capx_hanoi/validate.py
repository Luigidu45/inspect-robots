"""Audit a pinned Hanoi LeRobot snapshot without modifying it or either OCBench checkout.

Run with the OCBench environment that has NumPy, PyArrow, MuJoCo and PyAV.
Seed reconstruction is a hypothesis checked against every initial robot state;
only explicitly selected episodes receive a complete simulation replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CAMERAS = ("front", "side", "wrist")
SCALE = np.array([0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12])


def write_json(path: Path, value: Any) -> None:
    """Write reviewable evidence with strict JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def episode_metadata(root: Path) -> list[dict[str, Any]]:
    """Read episode records in their persisted LeRobot order."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: row["episode_index"])


def audit_numeric(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check every numeric row, episode boundary and camera timestamp interval."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    metadata = episode_metadata(root)
    table = pq.read_table(sorted((root / "data").rglob("*.parquet")))
    data = {
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
        "episode": table["episode_index"].to_numpy(),
        "frame": table["frame_index"].to_numpy(),
        "timestamp": table["timestamp"].to_numpy(),
        "index": table["index"].to_numpy(),
    }
    for key in ("success", "done", "reward"):
        data[key] = np.asarray(table[f"next.{key}"].to_pylist()).reshape(-1)
    checks: dict[str, bool] = {
        "row_count": table.num_rows == info["total_frames"],
        "episode_count": len(metadata) == info["total_episodes"],
        "state_shape": data["state"].shape == (table.num_rows, 7),
        "action_shape": data["action"].shape == (table.num_rows, 7),
        "finite": all(
            bool(np.isfinite(data[key]).all()) for key in ("state", "action", "timestamp", "reward")
        ),
        "global_index": bool(np.array_equal(data["index"], np.arange(table.num_rows))),
        "gripper_range": all(
            bool(((data[key][:, 6] >= 0) & (data[key][:, 6] <= 1)).all())
            for key in ("state", "action")
        ),
        "control_delta_bounds": bool(
            (np.abs(data["action"] - data["state"]) <= SCALE + 1e-6).all()
        ),
        "timestamps": bool((np.abs(data["timestamp"] - data["frame"] / info["fps"]) <= 1e-5).all()),
        "joint_names": info["features"]["action"]["names"]
        == info["features"]["observation.state"]["names"],
    }
    boundaries_ok = terminal_ok = success_ok = video_intervals_ok = True
    first_rows = []
    expected_start = 0
    for episode_id, row in enumerate(metadata):
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        length = row["length"]
        first_rows.append(start)
        boundaries_ok &= (
            row["episode_index"] == episode_id
            and start == expected_start
            and end - start == length
            and bool((data["episode"][start:end] == episode_id).all())
            and bool(np.array_equal(data["frame"][start:end], np.arange(length)))
        )
        terminal_ok &= bool(np.array_equal(np.flatnonzero(data["done"][start:end]), [length - 1]))
        success_ok &= bool(np.array_equal(np.flatnonzero(data["success"][start:end]), [length - 1]))
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            duration = row[f"{prefix}/to_timestamp"] - row[f"{prefix}/from_timestamp"]
            video_intervals_ok &= abs(duration * info["fps"] - length) <= 1e-5
        expected_start = end
    checks.update(
        episode_boundaries=bool(boundaries_ok and expected_start == table.num_rows),
        done_only_at_episode_end=bool(terminal_ok),
        success_only_at_episode_end=bool(success_ok),
        camera_intervals=bool(video_intervals_ok),
    )
    data["first"] = np.asarray(first_rows)
    report = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "frames": table.num_rows,
        "episodes": len(metadata),
        "fps": info["fps"],
        "episode_length_min": min(row["length"] for row in metadata),
        "episode_length_max": max(row["length"] for row in metadata),
        "max_timestamp_error_s": float(
            np.max(np.abs(data["timestamp"] - data["frame"] / info["fps"]))
        ),
        "max_target_minus_state": np.max(np.abs(data["action"] - data["state"]), axis=0).tolist(),
        "sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in (".json", ".parquet")
        },
    }
    return report, data


def load_generator(path: Path) -> Any:
    """Import the user-reviewed generator so its actuator conversion is tested directly."""
    spec = importlib.util.spec_from_file_location("hanoi_generator", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recover_seeds(generator: Any, env: Any, data: dict[str, Any], root_seed: int) -> list:
    """Recover unique reset candidates without assuming asynchronous episode order."""
    states = data["state"][data["first"]]
    episode_ids = data["episode"][data["first"]]
    remaining = set(range(len(states)))
    matches: list[dict[str, Any]] = []
    for attempt in range(10):
        for job, seed in generator._episode_jobs(len(states), root_seed):
            candidate = (seed + attempt * 1_000_003) % (2**32)
            np.random.seed(candidate)
            _, info = env.reset(seed=candidate)
            difference = np.max(np.abs(states - generator._joint_state(env)), axis=1)
            found = [int(i) for i in np.flatnonzero(difference <= 1e-6) if int(i) in remaining]
            if len(found) > 1:
                raise ValueError(f"Ambiguous seed candidate: {candidate}")
            for index in found:
                dynamics = info["dynamics_info"].copy()
                world = env.unwrapped.data.qpos.tobytes() + env.unwrapped.data.qvel.tobytes()
                matches.append(
                    {
                        "dataset_episode": int(episode_ids[index]),
                        "generation_job": job,
                        "seed": candidate,
                        "attempt": attempt + 1,
                        "initial_state_max_error": float(difference[index]),
                        "initial_world_sha256_reconstructed": hashlib.sha256(
                            world + dynamics.tobytes()
                        ).hexdigest(),
                        "peg_layout_reconstructed": dynamics.tolist(),
                    }
                )
                remaining.remove(index)
            if not remaining:
                break
        print(f"Seed candidates: attempt {attempt + 1}, matched {len(matches)}", flush=True)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Reset seeds not recovered for {len(remaining)} episodes")
    return sorted(matches, key=lambda row: row["dataset_episode"])


def check_actuator_targets(generator: Any, env: Any) -> dict[str, Any]:
    """Compare the saved target conversion with MuJoCo's actual actuator commands."""
    env.reset(seed=123)
    rng = np.random.default_rng(123)
    errors = []
    for action in [np.full(7, -1.0), np.zeros(7), np.ones(7), *rng.uniform(-1, 1, (100, 7))]:
        target = generator._absolute_action(env, action)
        env.unwrapped.set_control(action)
        actual = np.concatenate(
            [
                env.unwrapped.data.ctrl[env.unwrapped._arm_actuator_ids],
                env.unwrapped.data.ctrl[env.unwrapped._gripper_actuator_ids] / 255.0,
            ]
        )
        errors.append(float(np.max(np.abs(target - actual))))
    return {"cases": len(errors), "max_error": max(errors), "passed": max(errors) < 1e-6}


def replay_episodes(
    generator: Any,
    env: Any,
    data: dict[str, Any],
    ledger: list[dict[str, Any]],
    episode_ids: list[int],
) -> list[dict[str, Any]]:
    """Replay recorded absolute targets; report terminal outcomes and numerical drift."""
    results = []
    lookup = {row["dataset_episode"]: row for row in ledger}
    for episode_id in episode_ids:
        seed = lookup[episode_id]["seed"]
        np.random.seed(seed)
        _, info = env.reset(seed=seed)
        rows = np.flatnonzero(data["episode"] == episode_id)
        max_error = sum_squared_error = 0.0
        terminated = truncated = False
        executed_steps = 0
        for row in rows:
            executed_steps += 1
            current = generator._joint_state(env)
            difference = current - data["state"][row]
            max_error = max(max_error, float(np.max(np.abs(difference))))
            sum_squared_error += float(np.mean(difference**2))
            normalized = data["action"][row].astype(np.float64) - current.astype(np.float64)
            normalized /= env.unwrapped._joint_action_delta
            _, _, terminated, truncated, info = env.step(
                np.clip(normalized, -1, 1).astype(np.float32)
            )
            if terminated or truncated:
                break
        result = {
            "episode": episode_id,
            "seed": seed,
            "recorded_length": len(rows),
            "executed_steps": executed_steps,
            "success": bool(info["success"]),
            "failure": bool(info["failure"]),
            "healthy": bool(info["healthy"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "max_state_error": max_error,
            "state_rmse": float(np.sqrt(sum_squared_error / executed_steps)),
        }
        print("Replay:", json.dumps(result), flush=True)
        results.append(result)
    return results


def read_remote_video(url: str, reader: Callable[[Any], Any]) -> Any:
    """Retry transient remote demux/read failures without accepting an unchecked video."""
    import av

    for attempt in range(3):
        try:
            with av.open(url, timeout=(15, 15)) as container:
                return reader(container)
        except (av.FFmpegError, TimeoutError):
            if attempt == 2:
                raise
            print(f"Retrying video read, attempt {attempt + 2}/3", flush=True)
            time.sleep(2**attempt)
    raise AssertionError("Unreachable retry exit")


def probe_videos(root: Path, repo_id: str, revision: str) -> list[dict[str, Any]]:
    """Check each referenced MP4 header and decode its first frame, without full downloads."""
    names = set()
    for row in episode_metadata(root):
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            names.add(
                f"{prefix}/chunk-{row[f'{prefix}/chunk_index']:03d}/"
                f"file-{row[f'{prefix}/file_index']:03d}.mp4"
            )
    results = []

    def header(container: Any) -> dict[str, Any]:
        """Read stream metadata and require one decodable RGB frame."""
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        return {
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
            "duration_s": float(stream.duration * stream.time_base),
            "declared_frames": stream.frames,
            "first_frame_pts_s": float(frame.pts * stream.time_base),
            "decoded_shape": list(frame.to_ndarray(format="rgb24").shape),
        }

    for name in sorted(names):
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{name}"
        results.append({"file": name, **read_remote_video(url, header)})
    info = json.loads((root / "meta/info.json").read_text())
    by_name = {row["file"]: row for row in results}
    for camera in CAMERAS:
        prefix = f"videos/observation.images.{camera}"
        records = [row for row in results if row["file"].startswith(prefix + "/")]
        if sum(row["declared_frames"] for row in records) != info["total_frames"]:
            raise ValueError(f"Wrong total video frame count for {camera}")
        for row in records:
            if (
                row["codec"] != "h264"
                or row["fps"] != info["fps"]
                or row["decoded_shape"] != info["features"][f"observation.images.{camera}"]["shape"]
            ):
                raise ValueError(f"Video header does not match dataset schema: {row['file']}")
        for episode in episode_metadata(root):
            name = (
                f"{prefix}/chunk-{episode[f'{prefix}/chunk_index']:03d}/"
                f"file-{episode[f'{prefix}/file_index']:03d}.mp4"
            )
            if not (
                0
                <= episode[f"{prefix}/from_timestamp"]
                < episode[f"{prefix}/to_timestamp"]
                <= by_name[name]["duration_s"] + 1e-6
            ):
                raise ValueError(f"Episode video reference lies outside {name}")
    return results


def check_initial_rgb(
    generator: Any,
    env: Any,
    root: Path,
    ledger: list[dict[str, Any]],
    repo_id: str,
    revision: str,
) -> list[dict[str, Any]]:
    """Compare sampled initial video frames to rendered reset candidates, allowing compression."""
    metadata = {row["episode_index"]: row for row in episode_metadata(root)}
    lookup = {row["dataset_episode"]: row for row in ledger}
    results = []
    for episode_id in (0, len(ledger) // 2, len(ledger) - 1):
        seed = lookup[episode_id]["seed"]
        np.random.seed(seed)
        env.reset(seed=seed)
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            row = metadata[episode_id]
            name = (
                f"{prefix}/chunk-{row[f'{prefix}/chunk_index']:03d}/"
                f"file-{row[f'{prefix}/file_index']:03d}.mp4"
            )
            timestamp = row[f"{prefix}/from_timestamp"]
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{name}"

            def decoded_frame(
                container: Any, timestamp: float = timestamp, name: str = name
            ) -> tuple[np.ndarray, float]:
                """Seek the current episode's initial frame and retain its actual timestamp."""
                stream = container.streams.video[0]
                container.seek(
                    round(timestamp / float(stream.time_base)), stream=stream, backward=True
                )
                frame = None
                pts = -1.0
                for candidate in container.decode(stream):
                    pts = float(candidate.pts * stream.time_base)
                    if pts >= timestamp - 1 / 60:
                        frame = candidate
                        break
                if frame is None:
                    raise ValueError(f"No video frame at {timestamp} in {name}")
                return frame.to_ndarray(format="rgb24"), pts

            decoded, pts = read_remote_video(url, decoded_frame)
            render_name = generator.CAMERA_RENDER_NAMES.get(camera, camera)
            reference = env.unwrapped.render(camera=render_name)
            difference = reference.astype(float) - decoded.astype(float)
            mse = float(np.mean(difference**2))
            results.append(
                {
                    "episode": episode_id,
                    "camera": camera,
                    "expected_timestamp_s": timestamp,
                    "decoded_timestamp_s": pts,
                    "timestamp_error_s": abs(pts - timestamp),
                    "pixel_mae_0_255": float(np.mean(np.abs(difference))),
                    "psnr_db": float(10 * np.log10(255**2 / max(mse, 1e-12))),
                }
            )
    return results


def main() -> int:
    """Write audit evidence separately from the immutable input snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--ocbench-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root-seed-candidate", type=int, default=0)
    parser.add_argument("--probe-videos", action="store_true")
    parser.add_argument("--repo-id", default="NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2")
    parser.add_argument("--revision", default="c5f40db6f68a71147bef8224273dfeeb9d06bd81")
    parser.add_argument("--replay-episodes", nargs="+", type=int, default=[0, 56, 112, 168, 224])
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("Audit output must be outside the input dataset")
    report, data = audit_numeric(args.dataset_root)
    write_json(args.output_dir / "numeric_validation.json", report)
    if not report["all_checks_passed"]:
        return 1
    sys.path.insert(0, str(args.ocbench_root))
    path = args.ocbench_root / "scripts/generate_hanoi_lerobot.py"
    generator = load_generator(path)
    generator._init_worker(
        generator.WorkerConfig("/tmp", report["fps"], 20, 7000, CAMERAS, 224, 224, True, 10)
    )
    env = generator._WORKER_ENV
    try:
        targets = check_actuator_targets(generator, env)
        write_json(args.output_dir / "actuator_validation.json", targets)
        if not targets["passed"]:
            return 1
        ledger = recover_seeds(generator, env, data, args.root_seed_candidate)
        write_json(args.output_dir / "episode_provenance.json", ledger)
        if args.probe_videos:
            videos = probe_videos(args.dataset_root, args.repo_id, args.revision)
            write_json(args.output_dir / "video_probe.json", videos)
            alignment = check_initial_rgb(
                generator, env, args.dataset_root, ledger, args.repo_id, args.revision
            )
            write_json(args.output_dir / "initial_rgb_alignment.json", alignment)
        results = replay_episodes(generator, env, data, ledger, args.replay_episodes)
        write_json(args.output_dir / "replay.json", results)
    finally:
        env.close()
    return 0 if all(row["success"] and row["healthy"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
