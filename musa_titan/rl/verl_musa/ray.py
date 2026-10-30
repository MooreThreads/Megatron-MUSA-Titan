"""Ray visible-device compatibility for MUSA on older VERL checkouts."""

import os
from functools import wraps

from musa_titan.rl.common import _VERL_AUTHORED
from musa_titan.utils.patching import git_commit, replace_attr


_INSTALLED = False


def install() -> None:
    """Teach VERL's legacy Ray helper about the MUSA no-set variable."""
    global _INSTALLED
    if _INSTALLED:
        return

    from verl.utils import ray_utils

    original = ray_utils.ray_noset_visible_devices
    if getattr(original, "_verl_musa_patch", False):
        _INSTALLED = True
        return

    @wraps(original)
    def musa_ray_noset_visible_devices(env_vars=os.environ):
        return bool(
            env_vars.get("RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES")
        ) or original(env_vars)

    musa_ray_noset_visible_devices._verl_musa_patch = True
    replace_attr(
        ray_utils,
        "ray_noset_visible_devices",
        musa_ray_noset_visible_devices,
        reason="Recognize Ray's MUSA no-set flag when VERL assigns worker-local devices.",
        commits={"verl": git_commit(ray_utils.__file__)},
        authored=_VERL_AUTHORED,
        upstream_path="verl.utils.ray_utils.ray_noset_visible_devices",
        current_path="musa_titan/rl/verl_musa/ray.py",
        commit_branch_verified=True,
    )
    _INSTALLED = True
