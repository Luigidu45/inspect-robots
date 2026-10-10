"""Evaluate RGB and proprioceptive Hanoi policies without importing the simulator eagerly."""

from inspect_robots_ocbench.embodiment import OCBenchEmbodiment, ocbench_embodiment
from inspect_robots_ocbench.task import hanoi_task

__all__ = ["OCBenchEmbodiment", "hanoi_task", "ocbench_embodiment"]
