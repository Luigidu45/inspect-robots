"""Select advertised helpers independently of the code-generation model."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

IKSolver = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]

MOTION_HELPERS = ("move_to_joints", "move_to_configuration", "open_gripper", "close_gripper")
ASSISTANCE_HELPERS = {
    "none": MOTION_HELPERS,
    "ik": (*MOTION_HELPERS, "solve_ik"),
    "sam3_ik": (*MOTION_HELPERS, "segment", "solve_ik"),
    # Preserve the existing plugin API outside the staged Hanoi experiment.
    "default": (*MOTION_HELPERS, "segment", "plan_grasp", "solve_ik"),
}


def assistance_helpers(assistance: str) -> tuple[str, ...]:
    """Return supported helper names, rejecting unavailable assistance levels."""
    if assistance not in ASSISTANCE_HELPERS:
        raise ValueError(
            f"assistance must be one of {tuple(ASSISTANCE_HELPERS)}, got {assistance!r}"
        )
    return ASSISTANCE_HELPERS[assistance]
