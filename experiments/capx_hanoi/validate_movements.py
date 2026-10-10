"""Replay recorded targets through real CaP-X code execution and OCBench physics.

This is a movement integration test with scripted model responses, not an LLM
benchmark. It deliberately retains CaP-X interpolation and action guardrails.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from inspect_robots_capx.policy import CapxPolicy
from inspect_robots_ocbench import OCBenchEmbodiment

from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.scene import Scene
from inspect_robots.types import Observation


class _ReplayEmbodiment(OCBenchEmbodiment):
    """Cache RGB within a chunk while retaining fresh proprioception at every step."""

    def __init__(self) -> None:
        super().__init__()
        self._initial_observation: Observation | None = None

    def _observe(self) -> Observation:
        if self._initial_observation is None:
            self._initial_observation = super()._observe()
        return replace(
            self._initial_observation,
            state={"joint_pos": self._joint_state()},
            state_time=self._steps / 30,
        )

    def refresh(self) -> Observation:
        """Render all cameras before the next scripted model turn."""
        self._initial_observation = super()._observe()
        return self._initial_observation


def replay(
    numeric: Path, ledger_path: Path, episodes: list[int], chunk_size: int
) -> list[dict[str, Any]]:
    """Measure full replay outcomes through policy.act, helpers, guards and simulation."""
    data = np.load(numeric)
    ledger = {row["dataset_episode"]: row for row in json.loads(ledger_path.read_text())}
    results = []
    for episode in episodes:
        row = ledger[episode]
        targets = data["action"][data["episode"] == episode]
        codes = [
            "\n".join(
                f"move_to_configuration({target.tolist()!r})"
                for target in targets[start : start + chunk_size]
            )
            for start in range(0, len(targets), chunk_size)
        ]
        calls = 0

        def respond(request: httpx.Request, codes: list[str] = codes) -> httpx.Response:
            nonlocal calls
            if request.url.path != "/v1/chat/completions":
                raise AssertionError("Recorded joint code must not call perception or IK servers")
            calls += 1
            content = f"```python\n{codes[calls - 1]}\n```" if calls <= len(codes) else "FINISH"
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
            )

        robot = _ReplayEmbodiment()
        policy = CapxPolicy(
            model="scripted-motion-validation",
            base_url="http://scripted.test/v1",
            camera="front",
            transport=httpx.MockTransport(respond),
            env={},
            max_llm_calls=len(codes) + 1,
        )
        scene = Scene(
            id=f"recorded-{episode}",
            instruction="Solve the randomized three-disk Tower of Hanoi.",
            metadata={
                "reset_seed": row["seed"],
                "initial_world_sha256": row["initial_world_sha256_reconstructed"],
            },
        )
        steps = modified = 0
        outcome = None
        try:
            policy.bind(robot.info)
            policy.reset(scene)
            obs = robot.reset(scene, seed=0)
            guard = ChainApprover(
                ClampApprover(robot.info.action_space), DeltaLimitApprover(robot.info.action_space)
            )
            store: dict[str, Any] = {}
            queued = 0
            print(f"Episode {episode}: {len(targets)} targets, chunk size {chunk_size}", flush=True)
            for _ in codes:
                chunk = policy.act(obs)
                queued += len(chunk.actions)
                for proposed in chunk.actions[: 7000 - steps]:
                    approved = guard.review(proposed, store)
                    modified += int(approved is not proposed)
                    outcome = robot.step(approved)
                    steps += 1
                    if outcome.terminated or outcome.truncated:
                        break
                if steps >= 7000 or (outcome and (outcome.terminated or outcome.truncated)):
                    break
                obs = robot.refresh()
            result = {
                "episode": episode,
                "reset_seed": row["seed"],
                "recorded_steps": len(targets),
                "queued_steps": queued,
                "executed_steps": steps,
                "guardrail_modifications": modified,
                "scripted_model_calls": calls,
                "chunk_size": chunk_size,
                "success": bool(outcome and outcome.info["success"]),
                "termination_reason": outcome.termination_reason if outcome else None,
                "horizon_exhausted": steps >= 7000,
                "playback_exhausted": steps == queued,
                "validation_kind": "scripted Python via CapxPolicy; not LLM task reasoning",
                "rendering": "fresh RGB each model turn and final; cached images within each chunk",
            }
            final = OCBenchEmbodiment._observe(robot)
            assert all(image.shape == (224, 224, 3) for image in final.images.values())
            print(json.dumps(result), flush=True)
            results.append(result)
        finally:
            policy.close()
            robot.close()
    return results


def main() -> int:
    """Run requested recorded episodes and return nonzero if any does not solve Hanoi."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument(
        "--ledger", type=Path, default=Path(__file__).with_name("episode_provenance.json")
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    results = replay(args.numeric, args.ledger, args.episodes, args.chunk_size)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["success"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
