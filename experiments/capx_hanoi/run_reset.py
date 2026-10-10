"""Evaluate CaP-X restoring solved OCBench Hanoi episodes to their reference state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_robots_capx import CapxPolicy
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment, hanoi_reset_task
from reset_data import load_episode_actions

from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover


def main() -> int:
    """Prepare recorded final states, run the reset agent, and retain evidence."""
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Model identifier accepted by the endpoint")
    parser.add_argument(
        "--base-url", required=True, help="Model API base URL, including /v1 if needed"
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing the key, never a literal key",
    )
    parser.add_argument("--wire", choices=["chat", "responses"], default="chat")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=experiment / "split_manifest.json",
    )
    parser.add_argument(
        "--episode-ledger",
        type=Path,
        default=experiment / "episode_provenance.json",
    )
    parser.add_argument(
        "--prior-learnings", help="Optional UTF-8 operating notes without privileged geometry"
    )
    parser.add_argument("--log-dir", default="logs/capx-hanoi-reset")
    args = parser.parse_args()

    task = hanoi_reset_task(
        str(args.split_manifest),
        str(args.episode_ledger),
        args.split,
        args.num_scenes,
    )
    episode_ids = [int(scene.metadata["dataset_episode"]) for scene in task.scenes]
    episode_actions = load_episode_actions(args.dataset_root, episode_ids)
    robot = OCBenchHanoiResetEmbodiment(episode_actions)
    policy = CapxPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        wire=args.wire,
        camera="front",
        max_llm_calls=args.max_llm_calls,
        prior_learnings=args.prior_learnings,
    )
    guard = ChainApprover(
        ClampApprover(robot.info.action_space), DeltaLimitApprover(robot.info.action_space)
    )
    log = None
    try:
        log = eval(
            task,
            policy,
            robot,
            approver=guard,
            log_dir=args.log_dir,
            store_frames=True,
            store_actions=True,
            fail_on_error=False,
        )[0]
        successful = [sample for sample in log.samples if sample.reduced.get("success_at_end") == 1]
        successful_lengths = []
        for sample in successful:
            episode_length = sample.reduced.get("episode_length")
            if episode_length is not None:
                successful_lengths.append(episode_length)
        summary = {
            "evaluation_status": log.status,
            "scheduled_scenes": len(task.scenes),
            "successful_scenes": len(successful),
            "success_rate_all_scheduled": len(successful) / len(task.scenes),
            "mean_agent_steps_on_success": (
                sum(successful_lengths) / len(successful_lengths) if successful_lengths else None
            ),
            "errored_trials": log.results.errored_trials,
            "total_agent_control_steps": log.stats.total_steps,
            "evaluation_wall_time_s": log.stats.duration_s,
            "mean_inference_latency_s": log.stats.mean_inference_latency_s,
            "model": args.model,
            "base_url": args.base_url,
            "wire": args.wire,
            "split": args.split,
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_episodes": episode_ids,
            "protocol": "capx-hanoi-reference-reset-30hz-v1",
        }
        text = json.dumps(summary, indent=2) + "\n"
        output = Path(args.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        stamp = log.eval.created.replace(":", "-")
        (output / f"summary-{args.split}-{stamp}.json").write_text(text, encoding="utf-8")
        print(text)
    finally:
        policy.close()
        robot.close()
    return 0 if log is not None and log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
