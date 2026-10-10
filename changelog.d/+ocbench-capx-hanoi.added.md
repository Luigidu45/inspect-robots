Added an OCBench Hanoi plugin for the dataset-matched RGB and proprioception
experiment, with manifest seed/hash validation and bounded absolute joint
control. CaP-X now supports synchronized full configurations, including
fractional gripper targets. Experiment scripts replay recorded code through
CaP-X and evaluate separately configured vision models. A reference-image reset
task now reconstructs demonstration final states, scores restoration to the
original stack, and reports reset success and agent-step efficiency.
