"""Evaluate a native or CaP-X agent restoring solved OCBench Hanoi episodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
from inspect_robots_ocbench import OCBenchHanoiResetEmbodiment, hanoi_reset_task
from reset_data import load_episode_actions

from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover

if TYPE_CHECKING:
    import httpx


def _build_policy(args: argparse.Namespace, robot: OCBenchHanoiResetEmbodiment) -> Any:
    """Construct only the selected optional policy backend.

    The native ``agent`` path never imports CaP-X, so it needs neither its
    package nor its perception/IK servers.  ``capx`` remains available as an
    explicit comparative baseline.
    """
    common = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "wire": args.wire,
        "max_llm_calls": args.max_llm_calls,
        "prior_learnings": args.prior_learnings,
        "transcript_echo": args.live_transcript,
    }
    for name in ("max_output_tokens", "max_cost_usd"):
        value = getattr(args, name, None)
        if value is not None:
            common[name] = value
    if args.policy == "agent":
        try:
            from inspect_robots_agent import LLMAgentPolicy
        except ImportError as exc:
            raise ImportError(
                "The native reset agent needs inspect-robots-agent. "
                "Install it in this interpreter before running --policy agent."
            ) from exc
        return LLMAgentPolicy(
            **common,
            effort=args.effort,
            images=args.images,
            depth="off",
        )

    try:
        from inspect_robots_capx import CapxPolicy
    except ImportError as exc:
        raise ImportError(
            "--policy capx needs inspect-robots-capx. Install that optional plugin "
            "in this interpreter. The none and ik levels need no model servers."
        ) from exc
    queued_steps = getattr(args, "capx_max_queued_steps", None)
    return CapxPolicy(
        **common,
        camera="front",
        effort=args.effort,
        assistance=args.assistance or "none",
        ik_solver=robot.solve_ik if args.assistance in ("ik", "sam3_ik") else None,
        sam3_url=args.sam3_url or "http://127.0.0.1:8114",
        request_timeout_s=args.sam3_timeout_s,
        max_queued_steps=90 if queued_steps is None else queued_steps,
    )


def _attach_diagnostics(policy: Any, robot: OCBenchHanoiResetEmbodiment) -> None:
    """Add evaluation evidence without changing the native agent's control behavior."""
    original = getattr(policy, "on_trial_end", None)

    def finish(record: Any, log_dir: str, run_id: str) -> None:
        try:
            if callable(original):
                original(record, log_dir, run_id)
        finally:
            record.metadata["robot_diagnostics"] = robot.control_diagnostics

    policy.on_trial_end = finish


def _build_sinks(args: argparse.Namespace) -> list[Any] | None:
    """Return the optional live viewer plus the canonical JSON log sink.

    Keeping this lazy makes the normal reset path independent of Rerun and its
    desktop dependencies.  Supplying sinks to ``eval`` replaces its default
    JSON sink, so the explicit ``JsonLogSink`` is required whenever Rerun is
    enabled.
    """
    if not args.rerun:
        return None
    from inspect_robots.logging import JsonLogSink, RerunSink

    viewer_kwargs: dict[str, Any] = {
        "spawn": True,
        "recording_dir": args.log_dir,
    }
    if args.rerun_port is not None:
        viewer_kwargs["spawn_port"] = args.rerun_port
    return [JsonLogSink(args.log_dir), RerunSink(**viewer_kwargs)]


def _check_sam3(
    args: argparse.Namespace, *, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Require a working segmentation protocol before preparing any reset scene."""
    from inspect_robots_capx._servers import Sam3Client

    start = time.monotonic()
    client = Sam3Client(args.sam3_url, request_timeout_s=args.sam3_timeout_s, transport=transport)
    try:
        matches = client.segment(np.zeros((224, 224, 3), dtype=np.uint8), "disk")
        if any(match["mask"].shape != (224, 224) for match in matches):
            raise ValueError("SAM3 preflight masks must match the input image size")
    finally:
        client.close()
    return {
        "status": "passed",
        "image": "blank-rgb-224x224",
        "matches": len(matches),
        "duration_s": time.monotonic() - start,
    }


def main() -> int:
    """Prepare recorded final states, run the reset agent, and retain evidence."""
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--model", default="gpt-6-astra", help="Model identifier accepted by the endpoint"
    )
    parser.add_argument(
        "--base-url", required=True, help="Model API base URL, including /v1 if needed"
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing the key, never a literal key",
    )
    parser.add_argument(
        "--policy",
        choices=["agent", "capx"],
        default="agent",
        help="Native function-calling agent (default) or the optional CaP-X code agent.",
    )
    parser.add_argument("--wire", choices=["chat", "responses"], default="responses")
    parser.add_argument(
        "--goal-mode",
        choices=["text", "reference"],
        default="text",
        help="Describe the canonical reset in text (default) or also expose goal images.",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument(
        "--robot-fk",
        action="store_true",
        help="All policies: expose robot-only world pinch XYZ/WXYZ as a separate variant.",
    )
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=64,
        help="Per-trial reasoning/action calls (default: 64; this is not a spend limit).",
    )
    parser.add_argument("--max-output-tokens", type=int, help="Responses output plus reasoning cap")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        help="Per-trial estimated Astra spend guard; checked between requests, may overshoot once",
    )
    parser.add_argument(
        "--capx-max-queued-steps",
        type=int,
        help="CaP-X only: maximum steps queued by one code block (default: 90 = 3 s)",
    )
    parser.add_argument(
        "--assistance",
        choices=["none", "ik", "sam3_ik"],
        help="CaP-X only: motion, local UR5e/Robotiq IK, or SAM3 plus local IK.",
    )
    parser.add_argument("--sam3-url", help="SAM3 base URL, required only for sam3_ik")
    parser.add_argument(
        "--sam3-timeout-s",
        type=float,
        default=30.0,
        help="SAM3 request deadline, including retries (default: 30 s)",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        help="Reasoning effort for the selected LLM policy.",
    )
    parser.add_argument(
        "--images",
        choices=["always", "on_demand"],
        default="always",
        help="Current-camera delivery; goal frames depend on --goal-mode.",
    )
    parser.add_argument(
        "--live-transcript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print observations, usage, tool calls, and results while the rollout runs.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Open Rerun with live cameras, robot state, actions, and transcript.",
    )
    parser.add_argument(
        "--rerun-port",
        type=int,
        help="Local Rerun viewer port; requires --rerun (default: 9876).",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=experiment / "config" / "split_manifest.json",
    )
    parser.add_argument(
        "--episode-ledger",
        type=Path,
        default=experiment / "config" / "episode_provenance.json",
    )
    parser.add_argument(
        "--prior-learnings", help="Optional UTF-8 operating notes without privileged geometry"
    )
    parser.add_argument("--log-dir", default="logs/ocbench-hanoi-reset")
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be positive")
    if args.rerun_port is not None and not args.rerun:
        parser.error("--rerun-port requires --rerun")
    if args.rerun_port is not None and not 1 <= args.rerun_port <= 65535:
        parser.error("--rerun-port must be in [1, 65535]")
    try:
        api_endpoint = urlsplit(args.base_url)
        valid_api_url = api_endpoint.scheme in ("http", "https") and bool(api_endpoint.netloc)
    except ValueError:
        valid_api_url = False
    if not valid_api_url:
        parser.error("--base-url must be a plain HTTP(S) base URL, not Markdown")
    if args.model.removeprefix("openai/") == "gpt-6-astra" and args.wire != "responses":
        parser.error("gpt-6-astra tool calling requires --wire responses")
    if args.max_cost_usd is not None and (
        not np.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be finite and positive")
    if args.max_cost_usd is not None and args.model.removeprefix("openai/") != "gpt-6-astra":
        parser.error("--max-cost-usd requires known gpt-6-astra pricing")
    if args.wire != "responses" and (
        args.max_output_tokens is not None or args.max_cost_usd is not None
    ):
        parser.error("output and cost limits require --wire responses")
    if args.capx_max_queued_steps is not None and (
        args.policy != "capx" or args.capx_max_queued_steps < 1
    ):
        parser.error("--capx-max-queued-steps requires --policy capx and a positive value")
    if args.policy == "agent" and args.assistance is not None:
        parser.error("--assistance is only supported with --policy capx")
    if args.policy == "capx" and args.images != "always":
        parser.error("--policy capx currently requires --images always")
    if args.assistance == "sam3_ik" and not args.sam3_url:
        parser.error("--assistance sam3_ik requires --sam3-url; configure a real SAM3 endpoint")
    if args.sam3_url and not (args.policy == "capx" and args.assistance == "sam3_ik"):
        parser.error("--sam3-url is only supported with --policy capx --assistance sam3_ik")
    if args.sam3_url:
        try:
            endpoint = urlsplit(args.sam3_url)
            valid_url = endpoint.scheme in ("http", "https") and bool(endpoint.netloc)
        except ValueError:
            valid_url = False
        if not valid_url:
            parser.error("--sam3-url must be an HTTP(S) base URL")
    if not np.isfinite(args.sam3_timeout_s) or args.sam3_timeout_s <= 0:
        parser.error("--sam3-timeout-s must be finite and positive")
    sam3_preflight = None
    if args.assistance == "sam3_ik":
        try:
            sam3_preflight = _check_sam3(args)
        except (ConnectionError, TimeoutError, ValueError, KeyError) as exc:
            parser.error(f"SAM3 preflight failed before scene replay: {exc}")

    task = hanoi_reset_task(
        str(args.split_manifest),
        str(args.episode_ledger),
        args.split,
        args.num_scenes,
        args.goal_mode,
        robot_fk=args.robot_fk,
    )
    episode_ids = [int(scene.metadata["dataset_episode"]) for scene in task.scenes]
    episode_actions = load_episode_actions(args.dataset_root, episode_ids)
    robot = OCBenchHanoiResetEmbodiment(
        episode_actions,
        expose_reference_images=args.goal_mode == "reference",
        expose_robot_fk=args.robot_fk,
    )
    policy = _build_policy(args, robot)
    _attach_diagnostics(policy, robot)
    sinks = _build_sinks(args)
    if args.rerun:
        suffix = f" on port {args.rerun_port}" if args.rerun_port is not None else ""
        print(f"rerun: opening live viewer{suffix} and recording .rrd in {args.log_dir}")
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
            sinks=sinks,
            log_dir=args.log_dir,
            store_frames=True,
            store_actions=True,
            fail_on_error=False,
        )[0]
        successful = [sample for sample in log.samples if sample.reduced.get("success_at_end") == 1]
        successful_lengths = []
        trial_evidence = [metadata for sample in log.samples for metadata in sample.trial_metadata]
        usage_reports = [metadata.get("llm_usage", {}) for metadata in trial_evidence]
        known_costs = [report.get("estimated_cost_usd") for report in usage_reports]
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
            "policy": args.policy,
            "assistance": (args.assistance or "none") if args.policy == "capx" else None,
            "ik_backend": "ocbench-local"
            if args.policy == "capx" and args.assistance in ("ik", "sam3_ik")
            else None,
            "sam3_url": args.sam3_url if args.assistance == "sam3_ik" else None,
            "sam3_camera": "front" if args.assistance == "sam3_ik" else None,
            "sam3_preflight": sam3_preflight,
            "base_url": args.base_url,
            "wire": args.wire,
            "max_llm_calls": args.max_llm_calls,
            "effort": args.effort,
            "images": args.images,
            "live_transcript": args.live_transcript,
            "rerun": args.rerun,
            "rerun_port": args.rerun_port,
            "goal_mode": args.goal_mode,
            "robot_fk": args.robot_fk,
            "split": args.split,
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_episodes": episode_ids,
            "protocol": task.metadata["protocol"],
            "task_success": len(successful) == len(task.scenes),
            "llm_calls": sum(report.get("llm_calls", 0) for report in usage_reports),
            "llm_usage": usage_reports,
            "estimated_cost_usd": sum(known_costs)
            if known_costs and all(cost is not None for cost in known_costs)
            else None,
            "robot_diagnostics": [
                metadata.get("robot_diagnostics", {}) for metadata in trial_evidence
            ],
            "termination_reasons": [
                reason for sample in log.samples for reason in sample.termination_reasons
            ],
            "errors": [sample.error for sample in log.samples if sample.error],
            "max_output_tokens": args.max_output_tokens,
            "max_cost_usd_per_trial": args.max_cost_usd,
            "capx_max_queued_steps": (args.capx_max_queued_steps or 90)
            if args.policy == "capx"
            else None,
        }
        text = json.dumps(summary, indent=2) + "\n"
        output = Path(args.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        stamp = log.eval.created.replace(":", "-")
        (output / f"summary-{args.split}-{stamp}.json").write_text(text, encoding="utf-8")
        print(text)
    finally:
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
        robot.close()
    return 0 if log is not None and log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
