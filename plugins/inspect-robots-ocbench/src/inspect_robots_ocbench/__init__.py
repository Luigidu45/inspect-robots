"""Evaluate RGB and proprioceptive Hanoi policies without importing the simulator eagerly."""

from inspect_robots_ocbench.embodiment import (
    OCBenchEmbodiment,
    OCBenchHanoiResetEmbodiment,
    ocbench_embodiment,
)
from inspect_robots_ocbench.task import RESET_INSTRUCTION, hanoi_reset_task, hanoi_task

__all__ = [
    "RESET_INSTRUCTION",
    "OCBenchEmbodiment",
    "OCBenchHanoiResetEmbodiment",
    "hanoi_reset_task",
    "hanoi_task",
    "ocbench_embodiment",
]
