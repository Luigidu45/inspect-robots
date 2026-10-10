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
    }
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
    return CapxPolicy(
        **common,
        camera="front",
        effort=args.effort,
        assistance=args.assistance or "none",
        ik_solver=robot.solve_ik if args.assistance in ("ik", "sam3_ik") else None,
        sam3_url=args.sam3_url or "http://127.0.0.1:8114",
        request_timeout_s=args.sam3_timeout_s,
    )


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
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--max-llm-calls", type=int, default=8)
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
        default="low",
        help="Reasoning effort for the selected LLM policy.",
    )
    parser.add_argument(
        "--images",
        choices=["always", "on_demand"],
        default="always",
        help="Current-camera delivery; reference frames are always sent once.",
    )
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
    parser.add_argument("--log-dir", default="logs/ocbench-hanoi-reset")
    args = parser.parse_args()
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
    )
    episode_ids = [int(scene.metadata["dataset_episode"]) for scene in task.scenes]
    episode_actions = load_episode_actions(args.dataset_root, episode_ids)
    robot = OCBenchHanoiResetEmbodiment(episode_actions)
    policy = _build_policy(args, robot)
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
            "split": args.split,
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_episodes": episode_ids,
            "protocol": "ocbench-hanoi-reference-reset-30hz-v1",
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
