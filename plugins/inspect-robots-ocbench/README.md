# OCBench Hanoi adapter

This plugin registers the `ocbench` embodiment and the `ocbench-hanoi` and
`ocbench-hanoi-reset` tasks. It
implements the frozen Hanoi experiment in `experiments/capx_hanoi/config/protocol.json`:
CPU MuJoCo, non-lite `hanoi-triple-task2`, 30 Hz, 20 physics substeps, three
224 x 224 RGB cameras and seven measured joint positions. It is a custom
RGB + proprioception experiment. Official OCBench visual tasks use RGB alone.

Install the plugin's locked `ocbench` extra in the repository workspace:

```bash
uv sync --all-packages --extra ocbench --python 3.12
```

It resolves the reviewed upstream OCBench commit, MuJoCo stack and PyArrow from
`uv.lock`; it is not a manually maintained requirements file. OCBench is
imported only on the first reset, keeping plugin discovery and unit tests
independent of MuJoCo. Use `MUJOCO_GL=egl` for headless rendering.

Actions are absolute UR5e positions (six radians, then normalized gripper
opening: 0 closed, 1 open). The adapter bounds targets using actuator ranges,
converts them to normalized deltas relative to measured state, and clips each
delta to OCBench's native limits. CaP-X's `move_to_configuration` queues
simultaneous arm and fractional gripper movements. Its interpolation changes
the timing of a recorded trajectory; replay success must be measured separately.

The task loads a supplied JSON scene manifest. Its reset seeds override the
framework trial seed and the adapter checks each initial world hash. A mismatch
fails the trial. All scheduled resets remain in the evaluation denominator.
Only RGB and `joint_pos` reach the policy. Object positions, peg geometry,
rewards and success flags are reserved for the simulator and scorer.

Success requires OCBench success, healthy physics and no task failure. A
policy's `FINISH` request does not imply success. This package does not include
an LLM or a training pipeline. The `capx` policy connects to a separately
configured model endpoint.

The reset task reconstructs the final state of a recorded successful episode
by replaying its absolute joint targets with source-task success termination
disabled. Replay is setup and does not count against the reset agent's step
budget. The policy starts from that solved outer-peg stack and receives current
RGB, joint state, and a textual definition of the canonical middle-peg goal.
Goal images are disabled by default and can be enabled explicitly for a
reference-conditioned ablation. Demonstration actions and privileged object
geometry remain private in both modes.

Reset success uses OCBench's layout-aware tower predicate for the original
middle peg and requires the gripper to be released for five consecutive control
steps. The source Hanoi success flag is logged separately and cannot end or
pass the reset task. The arm may finish in any safe pose. This makes the target
an object arrangement rather than an exact simulator-state or pixel match.

The standard task requires a bitwise initial-world hash match. The reset task
records that comparison but treats it as advisory because full-precision MuJoCo
and IK state can vary across compatible runtimes while exported float32 state
and randomized peg dynamics remain identical. A reset scene is admitted only
when replaying every recorded target stays healthy and reaches OCBench's source
task success. The actual and expected hashes remain in reset provenance.
