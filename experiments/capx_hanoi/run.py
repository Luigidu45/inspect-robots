"""Evaluate a configured vision-capable LLM through CaP-X on frozen Hanoi scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_robots_capx import CapxPolicy
from inspect_robots_ocbench import OCBenchEmbodiment, hanoi_task

from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover


def main() -> int:
    """Run scheduled scenes with action guardrails and retain evaluation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier accepted by the endpoint")
    parser.add_argument(
        "--base-url", required=True, help="Model API base URL, including /v1 if needed"
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name, never a literal key",
    )
    parser.add_argument("--wire", choices=["chat", "responses", "anthropic"], default="chat")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument(
        "--prior-learnings", help="Optional UTF-8 operating notes, not privileged scene geometry"
    )
    parser.add_argument("--log-dir", default="logs/capx-hanoi")
    args = parser.parse_args()
    task = hanoi_task(
        str(Path(__file__).with_name("evaluation_scenes.json")), args.split, args.num_scenes
    )
    robot = OCBenchEmbodiment()
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
    try:
        log = eval(
            task,
            policy,
            robot,
            approver=guard,
            log_dir=args.log_dir,
            store_frames=True,
            fail_on_error=False,
        )[0]
        successful = sum(sample.reduced.get("success_at_end") == 1 for sample in log.samples)
        summary = {
            "evaluation_status": log.status,
            "scheduled_scenes": len(task.scenes),
            "successful_scenes": successful,
            "success_rate_all_scheduled": successful / len(task.scenes),
            "errored_trials": log.results.errored_trials,
            "completed_trials": log.results.total_trials - log.results.errored_trials,
            "total_control_steps": log.stats.total_steps,
            "evaluation_wall_time_s": log.stats.duration_s,
            "mean_inference_latency_s": log.stats.mean_inference_latency_s,
            "received_llm_replies": sum(
                message.get("role") == "assistant"
                for sample in log.samples
                for transcript in sample.policy_transcripts
                if isinstance(transcript, list)
                for message in transcript
                if isinstance(message, dict)
            ),
            "model": args.model,
            "base_url": args.base_url,
            "wire": args.wire,
            "split": args.split,
            "protocol": "capx-hanoi-triple-task2-rgb-proprio-30hz-v1",
        }
        text = json.dumps(summary, indent=2) + "\n"
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        Path(
            args.log_dir, f"summary-{args.split}-{log.eval.created.replace(chr(58), chr(45))}.json"
        ).write_text(text)
        print(text)
    finally:
        policy.close()
        robot.close()
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
