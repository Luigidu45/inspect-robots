# OCBench Hanoi reset experiment

This experiment evaluates an Inspect Robots agent that restores a solved
three-disk OCBench Hanoi scene to its original stack on the middle peg. It is
agentic closed-loop control, not an online-RL trainer and not a dataset
generator.

## Runtime flow

```text
LeRobot demonstration already on disk
  -> reset OCBench at the matching initial scene
  -> privately replay the demonstration to its solved final state
  -> agent receives a complete textual goal, current RGB, and joint positions
  -> guarded 30 Hz joint actions restore the middle-peg stack
  -> Inspect EvalLog stores frames, actions, transcript, and verdict
```

The replay only prepares the starting state: its actions, object poses,
rewards, and success signals are never shown to the policy. The agent does not
update weights during a rollout. `--prior-learnings`, when supplied, is static
text from an earlier development run, not online learning.

## Layout

```text
capx_hanoi/
  config/                         # versioned runtime manifests
    protocol.json                 # portable experiment contract
    split_manifest.json           # train/validation/test episode IDs
    episode_provenance.json       # reconstructed reset seed per episode
    evaluation_scenes.json        # fresh scenes for the non-reset baseline
  run_reset.py                    # primary reset entry point
  reset_data.py                   # read source actions from Parquet
  check_environment.py            # local preflight; no LLM request
  validate_tracking.py            # deterministic 30 Hz tracking check
  validate.py                     # optional dataset audit; writes elsewhere
  validate_assistance.py          # optional CaP-X assistance check
  validate_movements.py           # optional scripted CaP-X motion check
  tests/                          # runner tests
```

Audit results and run results are generated artifacts. Write them under
`logs/` or `/tmp`, not in this source directory.

## Inputs and provenance

The source dataset is
[`NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2`](https://huggingface.co/datasets/NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2),
revision `c5f40db6f68a71147bef8224273dfeeb9d06bd81`. The reset runner needs
only `meta/**` and `data/**`; it renders current observations in MuJoCo and
does not need the MP4 videos. The default `--goal-mode text` does not expose a
goal image. `--goal-mode reference` remains available as a controlled ablation.

[`config/protocol.json`](config/protocol.json) records the upstream OCBench
repository, reviewed generator commit and script hash, dependency versions,
control semantics, and evaluation contract. It contains no machine-local
paths. `config/episode_provenance.json` is reconstructed metadata used to reset
the matching scene; it is not a data generator and it does not expose layout
or object poses to the agent.

The environment is `hanoi-cpu-triple-task2-v0`, with 30 Hz control, 600 Hz
physics, three 224x224 RGB cameras, six arm joint positions, and normalized
gripper opening. This RGB-plus-proprioception protocol matches the dataset; it
is not the standard pixel-only OCBench protocol.

## Setup

From the `inspect-robots` repository root, create the Python 3.12 workspace environment
with `uv`. The `ocbench` extra belongs to the OCBench plugin and pins the
reviewed upstream commit plus the simulator/Parquet versions in the protocol.
`uv.lock` is the complete reproducible resolution; there is deliberately no
`requirements-reset.txt` or hand-installed editable dependency list.

```bash
uv sync --all-packages --extra ocbench --python 3.12
```

After cloning a revision that already includes `uv.lock`, use the locked form:

```bash
uv sync --locked --all-packages --extra ocbench
```

This produces the repository-standard `.venv/`. It includes the native
`agent` and OCBench plugins as editable workspace packages. CaP-X is present
only as another plugin; it is not imported when `--policy agent` is selected.
Headless EGL still requires a working host GPU/EGL driver, which Python package
management cannot supply.

Download the numeric dataset snapshot once. It is deliberately outside the
environment and outside Git: a virtual environment can be recreated and the
dataset is an immutable external input. `.data/` is the repository convention;
another absolute path is equally valid if passed through `--dataset-root`.

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2",
    repo_type="dataset",
    revision="c5f40db6f68a71147bef8224273dfeeb9d06bd81",
    allow_patterns=["meta/**", "data/**"],
    local_dir=".data/ocbench-hanoi-triple-task2",
)
PY
```

Confirm a fresh environment before a rollout. The native Inspect Robots doctor
reports the same `uv sync` remediation if OCBench or PyArrow is absent:

```bash
.venv/bin/inspect-robots doctor --embodiment ocbench
```

## Native-agent baseline

First verify that the installed native-agent path can prepare a real reset
scene; it does not make an LLM request. Exit code `2` means only that the API
key is not configured yet.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  .venv/bin/python experiments/capx_hanoi/check_environment.py \
  --dataset-root .data/ocbench-hanoi-triple-task2 \
  --output logs/audit/hanoi-native-preflight.json
```

Then check deterministic control before spending on an LLM:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  .venv/bin/python experiments/capx_hanoi/validate_tracking.py \
  --dataset-root .data/ocbench-hanoi-triple-task2 \
  --output logs/audit/hanoi-tracking.json
```

Set the key in the shell, never in a tracked file:

```bash
export OPENAI_API_KEY='...'
```

Run one development scene with the native LLM agent and no CaP-X. The default
64-call horizon is a termination bound, not a monetary spend limit:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  .venv/bin/python experiments/capx_hanoi/run_reset.py \
  --dataset-root .data/ocbench-hanoi-triple-task2 \
  --policy agent \
  --model gpt-6-astra \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --wire responses \
  --goal-mode text \
  --effort medium \
  --images always \
  --live-transcript \
  --max-llm-calls 64 \
  --split train \
  --num-scenes 1 \
  --log-dir logs/ocbench-hanoi-reset-agent-text-r0
```

Inspect Robots improves the next attempt through an auditable retry, not by
changing model weights. Summarize the generated EvalLog, inspect the Markdown,
then pass it to a new run:

```bash
.venv/bin/inspect-robots summarize \
  logs/ocbench-hanoi-reset-agent-text-r0/ocbench-hanoi-triple-task2-text-reset-30hz_*.json \
  --model gpt-6-astra \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  -o logs/ocbench-hanoi-reset-agent-text-r0/learnings.md
```

Use `--prior-learnings logs/ocbench-hanoi-reset-agent-text-r0/learnings.md`
on the next `train` run.
Once selected, freeze that file and use it unchanged on `validation`; reserve
`test` for the final configuration.

`--policy capx` remains an explicit comparison baseline. Its `none`, `ik`, and
`sam3_ik` profiles are optional; SAM3 is not needed for the native agent.

### Robot FK: optional spatial reference

Add `--robot-fk` to `run_reset.py` for any of `agent`, `capx-none`, or `capx-ik`.
The default remains RGB plus joints. With FK enabled, each current observation
also contains `state['pinch_pos']` (world XYZ in meters) and
`state['pinch_quat_wxyz']` (a unit WXYZ quaternion). These values are computed
from measured arm joints in a separate robot-only MuJoCo workspace, using the
same Robotiq pinch frame as local IK. FK does not expose disk or peg poses and
does not move the robot or change the IK solver workspace.

FK depends on the reviewed OCBench runtime's internal robot model and tool
transform. Re-run the physical FK/IK frame test after an upstream upgrade.

For CaP-X, the arrays are available in `obs['state']`. The native agent receives
them as state text. Embodiment notes describe their frame and purpose. The
`none` profile still has no `solve_ik` helper. FK gives a spatial reference; it
does not guarantee that a requested pose is reachable, that a grasp succeeds,
or that the interpolated trajectory avoids contact.

The EvalLog task metadata and run summary record `robot_fk`. FK runs use a
separate protocol identifier, such as `ocbench-hanoi-text-reset-30hz-fk-v1`.
Keep the flag identical across all three modes in a comparison and use separate
log directories for the original and FK variants.

Check the robot FK/IK frame round trip without an LLM or dataset replay:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  .venv/bin/python -m pytest -q plugins/inspect-robots-ocbench/tests/test_fk_sim.py
```

### IK execution feedback

CaP-X adds at most one IK diagnostic line per code block: the rejection count
and the first original error, even when Python catches that error. The detail
is bounded to 400 characters. It appears in execution feedback and, with live
transcripts enabled, the terminal. The diagnostic does not change the solver
or supply a target pose or Hanoi transfer strategy.

### Live visualization with Rerun

Install the optional viewer once. This does not change the standard runtime
when `--rerun` is omitted:

```bash
uv sync --locked --all-packages --extra ocbench --extra rerun
```

Add `--rerun` to a reset command to open a local live viewer. It shows the
three cameras, measured joints, commanded targets, reward, and agent
transcript, while also writing a replayable `.rrd` file into `--log-dir`.

```bash
  --rerun \
  --rerun-port 9876
```

The viewer needs a local graphical desktop. It is independent of
`MUJOCO_GL=egl`, which controls headless MuJoCo rendering.

## Outputs and evaluation discipline

Each run writes an Inspect EvalLog plus `summary-*.json` under `--log-dir`.
They contain the agent frames, actions, transcript, usage, robot diagnostics,
and task verdict. Keep these generated files outside the source directory.

Use `validation` to iterate and reserve `test` for a frozen configuration. A
future online-RL workflow must separately collect train rollouts, calculate
reward, update a trainable policy/checkpoint, and evaluate that frozen version.
The current agentic runner intentionally performs none of those updates.

## Optional dataset audit

`validate.py` audits a downloaded dataset; it does not generate one. It checks
all numeric records, action conversion, reset-seed candidates, and selected
replays; `--probe-videos` additionally inspects remote videos. Its output must
go outside this directory. This optional audit reads the pinned generator
source, so acquire that separate, read-only checkout only when auditing:

```bash
git clone https://github.com/seohongpark/ocbench.git ../ocbench
git -C ../ocbench checkout 95e3d4e16756f864b58e32a6a661286288ed3be8
```

Then run:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  .venv/bin/python experiments/capx_hanoi/validate.py \
  --dataset-root .data/ocbench-hanoi-triple-task2 \
  --ocbench-root ../ocbench \
  --output-dir logs/audit/ocbench-hanoi \
  --probe-videos
```
