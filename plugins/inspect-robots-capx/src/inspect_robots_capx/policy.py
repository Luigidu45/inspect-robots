"""Code-generation policy loop for the CaP-X integration.

Model output is executed in-process with the evaluator's privileges. This is
the policy under evaluation, not a security boundary. Robot actions still pass
through the rollout's approver chain, but untrusted models must run inside an
external container or equivalent isolation.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_agent import (
    ENV_MODEL,
    AssistantMessage,
    ChatClient,
    ResponsesClient,
    png_data_url,
    resolve_provider,
)
from inspect_robots_agent._capture import WireCapture, _safe
from inspect_robots_capx._assistance import IKSolver, assistance_helpers
from inspect_robots_capx._motion import MotionQueue
from inspect_robots_capx._sandbox import CodeSandbox, ExecutionResult
from inspect_robots_capx._servers import CapxServerClients, PyrokiClient, Sam3Client

_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_WIRE_FORMATS = frozenset({"chat", "responses"})


# Duplicated from inspect_robots_agent/policy.py; keep the effort contract in sync.
# The CLI -P parser coerces the literal `none` to Python None, so the constructor
# needs this marker to tell "operator asked for none" from "operator said nothing".
class _Unset:
    """Marker type distinguishing an omitted effort from an explicit one."""


_UNSET: Final = _Unset()
_EXECUTION_REPORT_CHAR_LIMIT = 16_000
_REPORT_TRUNCATION_MARKER = "[execution report truncated; tail follows]\n"
_REFERENCE_IMAGES_KEY = "reference_images"

# Duplicated in inspect_robots_agent/policy.py; keep both limits in sync.
_PRIOR_LEARNINGS_TEXT_LIMIT = 32 * 1024

_FENCED_CODE = re.compile(r"^```(?:python)?[ \t]*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_FENCED_ANYWHERE = re.compile(r"```(?:python)?[ \t]*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_CONTROL_WORD = re.compile(r"(FINISH|GIVE_UP)[.!]?", re.IGNORECASE)

_HELPER_DOCS = """Helpers available in the persistent namespace:

segment(text: str) -> list[dict]
    Segment the configured camera with SAM3. Each result has a boolean `mask`,
    `box`, `score`, and `label`.
plan_grasp(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    Return (K, 4, 4) grasp poses in the camera frame and (K,) scores. K may be
    zero. Use `obs["{extrinsics_key}"] @ pose` to transform a pose into the
    robot-base frame. World and robot base are treated as the same frame.
solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray
    Return the arm-joint vector expected by move_to_joints.
move_to_joints(joints: np.ndarray) -> None
    Queue a speed-limited arm move while holding the gripper.
move_to_configuration(configuration: np.ndarray) -> None
    Queue a synchronized speed-limited full action vector, including the
    gripper in its declared dimension. Use fractional openings when needed.
open_gripper() / close_gripper() -> None
    Queue a speed-limited gripper ramp while holding the arm.

`obs` is the current turn's dict with `images`, `state`, and every
observation.extra entry. Depth (`{depth_key}`), intrinsics (`{intrinsics_key}`),
and extrinsics (`{extrinsics_key}`) may be provided by the embodiment as
arrays or zero-argument callables; `obs[...]` access and the helper functions
resolve callable values automatically.
"""

_SYSTEM_TEMPLATE = """You generate Python code to directly solve a robot manipulation task.
Write raw Python with no Markdown fences. Helpers are already bound. Import
numpy explicitly when you need it. Variables persist across turns in this
trial. Perception within a turn uses the initial observation; queued motions
execute after your code returns, and you verify their effect next turn.

After each execution you receive the code, stdout, and stderr. Respond with
FINISH when the goal is complete, GIVE_UP when it cannot be completed, or
REGENERATE followed by corrected Python. You may also return raw corrected
Python directly. You have {budget} LLM calls for the whole trial.

Embodiment: {name}
Action profile: {action_summary}

{helper_docs}"""

_MOTION_HELPER_DOCS = """Helpers available in the persistent namespace:
move_to_joints(joints: np.ndarray) -> None
    Queue arm joints in radians while holding the gripper.
move_to_configuration(configuration: np.ndarray) -> None
    Queue the full action vector, including fractional gripper opening.
open_gripper() / close_gripper() -> None
    Queue gripper ramps while holding the arm.
`obs` contains the current images, state, and observation.extra entries.
Only the documented helpers below are installed.
"""
_IK_HELPER_DOCS = """solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray
    Solve for arm joints using the configured robot kinematics. Position is
    XYZ in meters in the documented robot frame; orientation is unit WXYZ.
    For OCBench, the target is the Robotiq pinch point in world coordinates.
    Local IK seeds from the queued arm cursor; HTTP IK uses its server seed.
    Solving does not move the robot.
    An IK exception means no valid solution was accepted; revise the target.
"""
_SAM3_HELPER_DOCS = """segment(text: str) -> list[dict]
    Segment the current RGB image from the configured camera using SAM3.
    Each result has a boolean (H, W) mask, a pixel-space XYXY box, score and label.
    The list can be empty. No match is perception uncertainty, not task failure.
    Masks describe the initial observation of this turn, before queued movement.
    They do not provide depth or a world-space pose; never pass pixel coordinates
    directly to solve_ik. No grasp-planning helper is installed in this profile.
"""


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an image-free deep copy suitable for persistence and visualization."""
    sanitized = copy.deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[index] = {
                    "type": "text",
                    "text": "[image omitted: streamed camera frame]",
                }
    return sanitized


@dataclass(frozen=True)
class CapxPolicyConfig(PolicyConfig):
    """Inference, server, motion, and transcript settings persisted in eval logs."""

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    wire: str = "chat"
    #: Resolved effort level; ``None`` means the field is omitted and the
    #: provider default applies. The recorded value is the normalized one, so
    #: an operator ``none`` reads back as ``"none"`` rather than a bare omission.
    effort: str | None = None
    assistance: str = "default"
    enabled_helpers: tuple[str, ...] = ()
    ik_backend: str = "pyroki"
    sam3_url: str = "http://127.0.0.1:8114"
    graspnet_url: str = "http://127.0.0.1:8115"
    pyroki_url: str = "http://127.0.0.1:8116"
    camera: str | None = None
    depth_key: str = "depth"
    intrinsics_key: str = "intrinsics"
    extrinsics_key: str = "extrinsics"
    max_llm_calls: int = 100
    max_code_failures: int = 3
    max_speed_frac: float = 0.1
    request_timeout_s: float = 120.0
    gripper_open_is_high: bool = True
    transcript_echo: bool = False
    #: Resolved absolute path to the injected prior-learnings file.
    prior_learnings: str | None = None
    #: SHA-256 hexdigest of the injected prior-learnings text.
    prior_learnings_sha256: str | None = None
    max_queued_steps: int | None = None
    max_output_tokens: int | None = None
    max_cost_usd: float | None = None
    wire_capture: bool = True


class CapxPolicy(PolicyBase):
    """Runs a persistent CaP-X-style codegen conversation over a bound joint arm.

    ``prior_learnings`` optionally loads a UTF-8 notes file once at construction
    and appends its text to every trial's system prompt.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        wire: str = "chat",
        max_llm_calls: int = 100,
        max_code_failures: int = 3,
        temperature: float | None = None,
        effort: str | None | _Unset = _UNSET,
        sam3_url: str = "http://127.0.0.1:8114",
        graspnet_url: str = "http://127.0.0.1:8115",
        pyroki_url: str = "http://127.0.0.1:8116",
        camera: str | None = None,
        depth_key: str = "depth",
        intrinsics_key: str = "intrinsics",
        extrinsics_key: str = "extrinsics",
        max_speed_frac: float = 0.1,
        request_timeout_s: float = 120.0,
        gripper_open_is_high: bool = True,
        transcript_echo: bool = False,
        prior_learnings: str | None = None,
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
        assistance: str = "default",
        ik_solver: IKSolver | None = None,
        service_transport: httpx.BaseTransport | None = None,
        max_queued_steps: int | None = None,
        max_output_tokens: int | None = None,
        max_cost_usd: float | None = None,
        wire_capture: bool = True,
    ) -> None:
        if max_queued_steps is not None and (
            isinstance(max_queued_steps, bool)
            or not isinstance(max_queued_steps, int)
            or max_queued_steps < 1
        ):
            raise ValueError("max_queued_steps must be an int >= 1")
        if wire != "responses" and (max_output_tokens is not None or max_cost_usd is not None):
            raise ConfigError("max_output_tokens and max_cost_usd require wire='responses'")
        self._max_queued_steps = max_queued_steps
        self._capture = WireCapture() if wire_capture else None
        self._usage_totals: dict[str, int] = {}
        enabled_helpers = assistance_helpers(assistance)
        if ik_solver is not None and assistance == "none":
            raise ValueError("ik_solver requires IK assistance")
        prior_learnings_path: str | None = None
        prior_learnings_text: str | None = None
        prior_learnings_sha256: str | None = None
        if prior_learnings is not None:
            if not isinstance(prior_learnings, str) or not prior_learnings:
                raise ConfigError(
                    "prior_learnings must be a non-empty filesystem path string, "
                    f"got {prior_learnings!r}.\n"
                    "fix: the -P parser coerces unquoted values; pass "
                    "-P 'prior_learnings=\"path/to/learnings.md\"'"
                )
            path = Path(prior_learnings)
            try:
                prior_learnings_text = path.read_text(encoding="utf-8")
                prior_learnings_path = str(path.resolve())
            except (OSError, UnicodeDecodeError) as exc:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} could not be read as UTF-8 "
                    f"({exc}).\n"
                    "fix: pass the path to a readable UTF-8 learnings file"
                ) from exc
            if not prior_learnings_text.strip():
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} is empty or whitespace-only.\n"
                    "fix: add concise notes to the file or omit -P prior_learnings="
                )
            if len(prior_learnings_text) > _PRIOR_LEARNINGS_TEXT_LIMIT:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} has "
                    f"{len(prior_learnings_text)} characters; the limit is "
                    f"{_PRIOR_LEARNINGS_TEXT_LIMIT}.\n"
                    "fix: summarize it first and pass the path to the shorter learnings file"
                )
            prior_learnings_sha256 = hashlib.sha256(
                prior_learnings_text.encode("utf-8")
            ).hexdigest()

        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        if max_code_failures < 1:
            raise ValueError("max_code_failures must be >= 1")
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
        if not np.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and > 0")
        # Unset leaves effort omitted so the provider default applies. A supplied
        # value normalizes the CLI-coerced None to the level "none" and is then
        # validated; ConfigError (not ValueError) so the CLI renders a guided
        # message instead of a traceback (#168).
        resolved_effort: str | None = None
        if not isinstance(effort, _Unset):
            resolved_effort = "none" if effort is None else effort
            if resolved_effort not in _EFFORT_LEVELS:
                raise ConfigError(
                    f"effort must be one of {sorted(_EFFORT_LEVELS)}, got {resolved_effort!r}.\n"
                    "fix: omit -P effort= to use the provider default"
                )
        if wire not in _WIRE_FORMATS:
            raise ValueError(f"wire must be one of {sorted(_WIRE_FORMATS)}, got {wire!r}")

        environ = dict(os.environ) if env is None else env
        provider = resolve_provider(
            model=model or environ.get(ENV_MODEL),
            base_url=base_url,
            api_key_env=api_key_env,
            env=environ,
        )
        if wire == "responses":
            self._client: ChatClient | ResponsesClient = ResponsesClient(
                provider,
                transport=transport,
                capture=self._capture,
                max_output_tokens=max_output_tokens,
                max_cost_usd=max_cost_usd,
            )
        else:
            self._client = ChatClient(provider, transport=transport, capture=self._capture)
        self._servers = (
            CapxServerClients(
                sam3_url=sam3_url,
                graspnet_url=graspnet_url,
                pyroki_url=pyroki_url,
                request_timeout_s=request_timeout_s,
                transport=service_transport if service_transport is not None else transport,
            )
            if assistance == "default"
            else None
        )
        self._sam3_client = (
            Sam3Client(
                sam3_url,
                request_timeout_s=request_timeout_s,
                transport=service_transport if service_transport is not None else transport,
            )
            if assistance == "sam3_ik"
            else None
        )
        self._pyroki_client = (
            PyrokiClient(
                pyroki_url,
                request_timeout_s=request_timeout_s,
                transport=service_transport if service_transport is not None else transport,
            )
            if assistance in ("ik", "sam3_ik") and ik_solver is None
            else None
        )
        self._assistance = assistance
        self._ik_solver = ik_solver
        self._camera_config = camera
        self._depth_key = depth_key
        self._intrinsics_key = intrinsics_key
        self._extrinsics_key = extrinsics_key
        self._max_llm_calls = max_llm_calls
        self._max_code_failures = max_code_failures
        self._temperature = temperature
        self._effort = resolved_effort
        self._max_speed_frac = max_speed_frac
        self._gripper_open_is_high = gripper_open_is_high
        self._transcript_echo = transcript_echo
        self._prior_learnings_text = prior_learnings_text
        self.config = CapxPolicyConfig(
            temperature=temperature,
            model=provider.model,
            base_url=provider.base_url,
            api_key_env=api_key_env,
            wire=wire,
            effort=resolved_effort,
            assistance=assistance,
            enabled_helpers=enabled_helpers,
            ik_backend="none" if assistance == "none" else "local" if ik_solver else "pyroki",
            sam3_url=sam3_url,
            graspnet_url=graspnet_url,
            pyroki_url=pyroki_url,
            camera=camera,
            depth_key=depth_key,
            intrinsics_key=intrinsics_key,
            extrinsics_key=extrinsics_key,
            max_llm_calls=max_llm_calls,
            max_code_failures=max_code_failures,
            max_speed_frac=max_speed_frac,
            max_queued_steps=max_queued_steps,
            max_output_tokens=max_output_tokens,
            max_cost_usd=max_cost_usd,
            wire_capture=wire_capture,
            request_timeout_s=request_timeout_s,
            gripper_open_is_high=gripper_open_is_high,
            transcript_echo=transcript_echo,
            prior_learnings=prior_learnings_path,
            prior_learnings_sha256=prior_learnings_sha256,
        )
        self.info = PolicyInfo(name="capx", action_space=Box(shape=(1,)))
        self._motion: MotionQueue | None = None
        self._sandbox: CodeSandbox | None = None
        self._state_key: str | None = None
        self._state_labels: tuple[str, tuple[str, ...]] | None = None
        self._embodiment_name = "(unbound)"
        self._embodiment_docs: str | None = None
        self._action_summary = "unbound; call bind() before act()"
        self._messages: list[dict[str, Any]] = []
        self._delta_cursor = 0
        self._calls_used = 0
        self._consecutive_failures = 0
        self._reference_images_sent = False
        self._closed = False
        atexit.register(self.close)

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt an embodiment that satisfies the plan-0021 single-arm profile."""
        space = embodiment_info.action_space
        prefix = "plan 0021 CaP-X v1 profile requires"
        if len(space.shape) != 1 or space.dim < 2:
            raise ValueError(f"{prefix} a 1-D Box action space with at least 2 dimensions")
        semantics = space.semantics
        if semantics is None or semantics.control_mode != "joint_pos":
            mode = None if semantics is None else semantics.control_mode
            raise ValueError(f"{prefix} control_mode='joint_pos', got {mode!r}")
        if semantics.gripper == "none":
            raise ValueError(f"{prefix} ActionSemantics.gripper to be continuous or binary")
        labels = semantics.dim_labels
        if labels is None:
            gripper_index = space.dim - 1
        else:
            matches = [index for index, label in enumerate(labels) if label == "gripper"]
            if len(matches) != 1:
                raise ValueError(
                    f"{prefix} exactly one dim_labels entry named 'gripper'; found {matches}"
                )
            gripper_index = matches[0]
        if embodiment_info.control_hz is None:
            raise ValueError(f"{prefix} a declared finite control_hz > 0")

        state_spec = embodiment_info.observation_space.state
        if state_spec is None:
            raise ValueError(f"{prefix} an observation StateSpec for full joint state")
        matching = [field.key for field in state_spec.fields if field.shape == (space.dim,)]
        if len(matching) != 1:
            raise ValueError(
                f"{prefix} exactly one state field with shape ({space.dim},); "
                f"found {matching or 'none'}"
            )
        state_key = matching[0]

        camera_names = embodiment_info.observation_space.camera_names
        if self._camera_config is None:
            if len(camera_names) != 1:
                raise ValueError(
                    f"{prefix} one camera when camera=None; declared {sorted(camera_names)}"
                )
            camera = next(iter(camera_names))
        else:
            camera = self._camera_config
            if camera not in camera_names:
                raise ValueError(
                    f"{prefix} the configured camera to be declared; {camera!r} is absent, "
                    "available: "
                    f"{sorted(camera_names)}"
                )

        try:
            motion = MotionQueue(
                space,
                control_hz=embodiment_info.control_hz,
                max_speed_frac=self._max_speed_frac,
                gripper_index=gripper_index,
                gripper_open_is_high=self._gripper_open_is_high,
                max_queued_steps=self._max_queued_steps,
            )
        except ValueError as exc:
            raise ValueError(f"{prefix} valid bounded motion arithmetic: {exc}") from exc
        self._motion = motion
        self._sandbox = CodeSandbox(
            servers=self._servers,
            motion=motion,
            camera=camera,
            state_key=state_key,
            depth_key=self._depth_key,
            intrinsics_key=self._intrinsics_key,
            extrinsics_key=self._extrinsics_key,
            assistance=self._assistance,
            ik_solver=self._ik_solver,
            sam3_client=self._sam3_client,
            pyroki_client=self._pyroki_client,
        )
        self._state_key = state_key
        self._state_labels = (state_key, labels) if labels is not None else None
        self._embodiment_name = embodiment_info.name
        self._embodiment_docs = getattr(embodiment_info, "docs", None)
        arm_labels = [
            labels[index] if labels is not None else str(index)
            for index in range(space.dim)
            if index != gripper_index
        ]
        polarity = (
            "high=open, low=closed" if self._gripper_open_is_high else "low=open, high=closed"
        )
        self._action_summary = (
            f"joint_pos with {motion.arm_dof} arm joints {arm_labels}, gripper dim "
            f"{gripper_index} ({polarity}), control_hz={embodiment_info.control_hz:g}"
        )
        self.info = PolicyInfo(
            name="capx",
            action_space=space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        """Start a trial with fresh code, transcript, failure, and IK state."""
        system = _SYSTEM_TEMPLATE.format(
            budget=self._max_llm_calls,
            name=self._embodiment_name,
            action_summary=self._action_summary,
            helper_docs=(
                _HELPER_DOCS
                if self._assistance == "default"
                else (
                    _MOTION_HELPER_DOCS
                    + (_IK_HELPER_DOCS if self._assistance in ("ik", "sam3_ik") else "")
                    + (_SAM3_HELPER_DOCS if self._assistance == "sam3_ik" else "")
                )
            ).format(
                depth_key=self._depth_key,
                intrinsics_key=self._intrinsics_key,
                extrinsics_key=self._extrinsics_key,
            ),
        )
        if self._embodiment_docs is not None and self._embodiment_docs.strip():
            system += "\n\nEmbodiment notes:\n" + self._embodiment_docs.strip()
        if self._max_queued_steps is not None:
            system += (
                f"\n\nEach generated code block may queue at most {self._max_queued_steps} "
                "control steps. "
                "Include gripper ramps and holds in this budget. Prefer one short physical "
                "phase, then inspect the next observation. A code exception discards all "
                "motions from that block; variables can persist. Revise recoverable errors "
                "using the feedback. Do not infer that a queued target has been reached."
            )
        if self._prior_learnings_text is not None:
            system = (
                system
                + "\n\nNotes from a previous attempt at tasks like this one. They may "
                + "be wrong or stale; the current observation always wins:\n"
                + self._prior_learnings_text
            )
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {scene.instruction}"},
        ]
        self._delta_cursor = 0
        self._calls_used = 0
        self._usage_totals.clear()
        if isinstance(self._client, ResponsesClient):
            self._client.usage_ledger.reset()
        self._consecutive_failures = 0
        self._reference_images_sent = False
        if self._sandbox is not None:
            self._sandbox.reset()
        if self._servers is not None:
            self._servers.pyroki.reset()
        if self._pyroki_client is not None:
            self._pyroki_client.reset()
        self._echo(f"[capx] goal: {scene.instruction}")

    def transcript(self) -> list[dict[str, Any]] | None:
        """Return an image-free deep copy of the current trial conversation."""
        return _sanitize(self._messages) if self._messages else None

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        """Capture attempts using the same credential-free sink as the native agent."""
        if self._capture is not None:
            self._capture.begin_trial(log_dir, run_id, f"{_safe(scene_id)}-e{epoch}")

    def on_trial_end(self, record: Any, log_dir: str, run_id: str) -> None:
        """Persist usage and a portable transcript even when evaluation fails."""
        if self._capture is not None:
            capture_path = self._capture.end_trial()
            if capture_path is not None:
                record.metadata["wire_capture"] = capture_path
        record.metadata["llm_usage"] = {
            "llm_calls": self._calls_used,
            **self._usage_totals,
            **(
                self._client.usage_ledger.report()
                if isinstance(self._client, ResponsesClient)
                else {}
            ),
        }
        messages = self.transcript()
        if messages:
            relative = f"transcripts/{run_id}/{_safe(record.scene_id)}-e{record.epoch}.jsonl"
            path = Path(log_dir) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(message) + "\n" for message in messages))
            record.metadata["transcript"] = relative

    def transcript_delta(self) -> list[dict[str, Any]] | None:
        """Return only sanitized messages appended since the previous live-stream read."""
        new = self._messages[self._delta_cursor :]
        self._delta_cursor = len(self._messages)
        return _sanitize(new) if new else None

    def act(self, observation: Observation) -> ActionChunk:
        """Run codegen turns until code queues motion or the model requests a stop."""
        motion = self._motion
        sandbox = self._sandbox
        state_key = self._state_key
        if motion is None or sandbox is None or state_key is None:
            raise RuntimeError(
                "CapxPolicy.act() before bind(); run it through eval() or call "
                "policy.bind(embodiment.info) first"
            )
        sandbox.set_observation(observation)
        state = observation.state[state_key]
        self._messages.append(
            {
                "role": "user",
                "content": _observation_content(
                    observation,
                    self._state_labels,
                    include_reference=not self._reference_images_sent,
                ),
            }
        )
        if _REFERENCE_IMAGES_KEY in observation.extra:
            self._reference_images_sent = True
        llm_latency = 0.0
        while True:
            if self._calls_used >= self._max_llm_calls:
                self._echo("[capx] -- LLM call budget exhausted; forcing GIVE_UP")
                return motion.hold_chunk(
                    state,
                    stop_reason="GIVE_UP",
                    inference_latency_s=llm_latency,
                )
            started = time.monotonic()
            message = self._client.complete(
                self._messages,
                [],
                temperature=self._temperature,
                reasoning_effort=self._effort,
            )
            llm_latency += time.monotonic() - started
            self._calls_used += 1
            for key, value in (message.usage or {}).items():
                self._usage_totals[key] = self._usage_totals.get(key, 0) + value
            self._messages.append(message.raw())
            reply = (message.content or "").strip()
            if reply:
                self._echo(f"[capx] << {reply}")
            control = _CONTROL_WORD.fullmatch(reply)
            if control is not None:
                return motion.hold_chunk(
                    state,
                    stop_reason=control.group(1).upper(),
                    inference_latency_s=llm_latency,
                )

            code = _extract_code(message)
            result = sandbox.execute(code)
            self._messages.append({"role": "user", "content": _execution_report(code, result)})
            self._echo_execution(result)
            if result.raised:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

            if motion.has_actions():
                return motion.take_chunk(inference_latency_s=llm_latency, code=code)
            if result.raised and self._consecutive_failures >= self._max_code_failures:
                raise RuntimeError(
                    "model code failed in "
                    f"{self._consecutive_failures} consecutive turns; see transcript stderr"
                )

    def close(self) -> None:
        """Idempotently close LLM and shared CaP-X HTTP connection pools."""
        if self._closed:
            return
        self._closed = True
        self._client.close()
        if self._servers is not None:
            self._servers.close()
        if self._sam3_client is not None:
            self._sam3_client.close()
        if self._pyroki_client is not None:
            self._pyroki_client.close()
        with contextlib.suppress(Exception):  # defensive during interpreter teardown
            atexit.unregister(self.close)

    def _echo(self, text: str) -> None:
        if self._transcript_echo:
            print(text, file=sys.stderr, flush=True)

    def _echo_execution(self, result: ExecutionResult) -> None:
        if not self._transcript_echo:
            return
        if result.stdout:
            self._echo(f"[capx] stdout:\n{result.stdout.rstrip()}")
        if result.stderr:
            self._echo(f"[capx] stderr:\n{result.stderr.rstrip()}")


def _extract_code(message: AssistantMessage) -> str:
    """Normalize raw, fenced, prose-wrapped, or ``REGENERATE``-prefixed code.

    A reply that is exactly one fenced block (or a bare snippet) is used as
    is; otherwise the first fenced block wins, so surrounding prose does not
    burn a failure turn on a ``SyntaxError``.
    """
    code = (message.content or "").strip()
    first, separator, remainder = code.partition("\n")
    if first.strip() == "REGENERATE" and separator:
        code = remainder.strip()
    fenced = _FENCED_CODE.fullmatch(code)
    if fenced is not None:
        return fenced.group(1)
    embedded = _FENCED_ANYWHERE.search(code)
    return embedded.group(1) if embedded is not None else code


def _execution_report(code: str, result: ExecutionResult) -> str:
    """Build a bounded tail-first CaP-X execution feedback message."""
    report = (
        "Executed code:\n"
        f"```python\n{code}\n```\n"
        "stdout:\n"
        f"```text\n{result.stdout}\n```\n"
        "stderr:\n"
        f"```text\n{result.stderr}\n```\n"
        "Respond with FINISH, GIVE_UP, or REGENERATE followed by Python."
    )
    if len(report) <= _EXECUTION_REPORT_CHAR_LIMIT:
        return report
    tail_size = _EXECUTION_REPORT_CHAR_LIMIT - len(_REPORT_TRUNCATION_MARKER)
    return _REPORT_TRUNCATION_MARKER + report[-tail_size:]


def _state_lines(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None,
) -> list[str]:
    """Render observation state with action labels when the bound vector aligns."""
    lines: list[str] = []
    for key, value in observation.state.items():
        array = np.asarray(value, dtype=np.float64)
        rounded = np.round(array, 4).tolist()
        if state_labels is not None and key == state_labels[0]:
            labels = state_labels[1]
            if array.shape == (len(labels),):
                labeled = " ".join(
                    f"{label}={item}" for label, item in zip(labels, rounded, strict=True)
                )
                lines.append(f"state[{key}]: {labeled}")
                continue
        lines.append(f"state[{key}]: {rounded}")
    return lines


def _observation_content(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None,
    *,
    include_reference: bool = True,
) -> list[dict[str, Any]]:
    """Encode fresh state and cameras, with goal images once when provided."""
    parts: list[dict[str, Any]] = []
    if include_reference and _REFERENCE_IMAGES_KEY in observation.extra:
        references = observation.extra[_REFERENCE_IMAGES_KEY]
        if not isinstance(references, Mapping) or not references:
            raise ValueError("observation.extra['reference_images'] must be a non-empty mapping")
        parts.append(
            {
                "type": "text",
                "text": (
                    "Reference goal images. These show the desired object arrangement; "
                    "the robot arm pose does not need to match."
                ),
            }
        )
        for name, image in references.items():
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError(
                    "observation.extra['reference_images'] values must be HxWx3 uint8 arrays"
                )
            parts.append({"type": "text", "text": f"reference camera {name!r}:"})
            parts.append({"type": "image_url", "image_url": {"url": png_data_url(array)}})

    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.extend(_state_lines(observation, state_labels))
    parts.append({"type": "text", "text": "\n".join(lines)})
    for name, image in observation.images.items():
        parts.append({"type": "text", "text": f"camera {name!r}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
    return parts


def capx_policy(**kwargs: Any) -> CapxPolicy:
    """Build the registry policy while forwarding CLI ``-P`` keyword arguments."""
    return CapxPolicy(**kwargs)
