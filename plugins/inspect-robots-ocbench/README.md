# OCBench Hanoi adapter

This plugin registers the `ocbench` embodiment and `ocbench-hanoi` task. It
implements the frozen Hanoi experiment in `experiments/capx_hanoi/protocol.json`:
CPU MuJoCo, non-lite `hanoi-triple-task2`, 30 Hz, 20 physics substeps, three
224 x 224 RGB cameras and seven measured joint positions. It is a custom
RGB + proprioception experiment. Official OCBench visual tasks use RGB alone.

Install the reviewed OCBench checkout in the same Python environment as this
plugin. OCBench is imported on the first reset, keeping plugin discovery and
unit tests independent of MuJoCo. The checkout commit and simulation versions
are recorded in the experiment protocol; installing an arbitrary OCBench release
does not establish reproducibility. Use `MUJOCO_GL=egl` for headless rendering.

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
