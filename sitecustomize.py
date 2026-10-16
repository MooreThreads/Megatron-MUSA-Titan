"""Load RL MUSA hooks when a process opts in through the environment.

Python imports the first ``sitecustomize`` found on ``sys.path``.  Putting this
file first via ``PYTHONPATH=$MUSA_TITAN_PATH:...`` covers the Ray job driver,
Ray workers, and SGLang/Megatron child processes without a Miles or slime
sitecustomize.  CUDA/CPU processes leave the RL startup markers unset and are
therefore unchanged.
"""

import os
import sys


_TRUTHY_STARTUP_VALUES = {"1", "true", "yes"}
_SLIME_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"slime"}
_MILES_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"miles"}

startup_command = " ".join(getattr(sys, "orig_argv", sys.argv))
miles_requested = os.environ.get("MILES_MUSA_PATCH_AT_STARTUP", "").strip().lower() in _MILES_STARTUP_VALUES
slime_requested = os.environ.get("SLIME_MUSA_PATCH_AT_STARTUP", "").strip().lower() in _SLIME_STARTUP_VALUES
if (
    (miles_requested or slime_requested)
    and os.environ.get("MUSA_TITAN_RL_BOOTSTRAPPING") != "1"
    and os.environ.get("MILES_MUSA_PATCH_BOOTSTRAPPING") != "1"
    and "multiprocessing.resource_tracker" not in startup_command
):
    os.environ["MUSA_TITAN_RL_BOOTSTRAPPING"] = "1"
    if miles_requested:
        os.environ["MILES_MUSA_PATCH_BOOTSTRAPPING"] = "1"
    try:
        import musa_titan

        musa_titan.apply_rl_patch()
    finally:
        os.environ.pop("MUSA_TITAN_RL_BOOTSTRAPPING", None)
        os.environ.pop("MILES_MUSA_PATCH_BOOTSTRAPPING", None)
