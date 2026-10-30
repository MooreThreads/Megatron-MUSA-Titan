"""Select pretraining and RL startup patches with environment variables.

PRETRAIN_MUSA_PATCH_AT_STARTUP=1 enables shared Megatron and pretraining patches.
SLIME_MUSA_PATCH_AT_STARTUP=1 enables shared Megatron patches and Slime integration.
MILES_MUSA_PATCH_AT_STARTUP=1 enables shared Megatron patches and Miles integration.
VERL_MUSA_PATCH_AT_STARTUP=1 installs delayed VERL/MUSA import hooks.

Python imports the first ``sitecustomize`` found on ``sys.path``.  Putting this
file first via ``PYTHONPATH=$MUSA_TITAN_PATH:...`` covers the Ray job driver,
Ray workers, and SGLang/Megatron child processes without a Miles, slime, or
VERL sitecustomize.  CUDA/CPU processes leave the RL startup markers unset and
are therefore unchanged.
"""

import importlib.util
import os
import sys
from pathlib import Path


_TRUTHY_STARTUP_VALUES = {"1", "true", "yes"}
_SLIME_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"slime"}
_MILES_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"miles"}
_VERL_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"verl"}

startup_command = " ".join(getattr(sys, "orig_argv", sys.argv))


pretrain_requested = (
    os.environ.get("PRETRAIN_MUSA_PATCH_AT_STARTUP", "").strip().lower()
    in _TRUTHY_STARTUP_VALUES
)
if (
    pretrain_requested
    and os.environ.get("MUSA_TITAN_PRETRAIN_BOOTSTRAPPING") != "1"
):
    os.environ["MUSA_TITAN_PRETRAIN_BOOTSTRAPPING"] = "1"
    try:
        import musa_titan

        musa_titan.apply_pretrain_patch()
    except Exception as error:
        # Python's site module otherwise logs ordinary exceptions and continues.
        raise SystemExit("MUSA pretraining patch initialization failed") from error
    finally:
        os.environ.pop("MUSA_TITAN_PRETRAIN_BOOTSTRAPPING", None)

miles_requested = os.environ.get("MILES_MUSA_PATCH_AT_STARTUP", "").strip().lower() in _MILES_STARTUP_VALUES
slime_requested = os.environ.get("SLIME_MUSA_PATCH_AT_STARTUP", "").strip().lower() in _SLIME_STARTUP_VALUES
verl_requested = os.environ.get("VERL_MUSA_PATCH_AT_STARTUP", "").strip().lower() in _VERL_STARTUP_VALUES
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
elif verl_requested and "multiprocessing.resource_tracker" not in startup_command:
    # Load the VERL finders without importing musa_titan.  That package pulls
    # in torch_musa, which must not run in Ray helpers that only inherit the
    # environment marker.
    hook_path = Path(__file__).resolve().parent / "musa_titan" / "rl" / "verl_musa" / "import_hooks.py"
    spec = importlib.util.spec_from_file_location("_musa_titan_verl_import_hooks", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_startup_hooks()
