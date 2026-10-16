# Miles MUSA integration

This package contains the MUSA-specific Miles training, rollout, and
weight-update integration.  Miles official code is left unchanged.

## Miles version requirement

This adapter currently supports only the official Miles `v0.1.0` codebase.
The Miles checkout used as `working_dir` must correspond to that release
(including the official `v0.1.0` source layout and APIs).

The adapter is not a general compatibility layer for Miles `main`, newer
releases, or unrelated development snapshots.  Other versions may change
module paths, argument names, SGLang integration points, or weight-update
interfaces and are unsupported until the adapter is revalidated.  Do not mix
the `v0.1.0` launcher with a different Miles checkout and assume that the
MUSA patch will remain compatible.

No MUSA-specific changes need to be applied to the Miles source itself; use
the matching official `v0.1.0` checkout together with this package and the
startup environment described below.

## Load path

Put `MUSA_TITAN_PATH` first on `PYTHONPATH` and set:

- `MUSA_VISIBLE_DEVICES`
- `MUSA_TITAN_PATH` (the legacy `MUSA_PATCH_PATH` name is also accepted)
- `MILES_MUSA_PATCH_AT_STARTUP=1`

Python then imports this checkout's `sitecustomize.py` before Miles or
SGLang, and `sitecustomize` imports `musa_titan` and calls
`apply_rl_patch()`.  `musa_titan` applies the CUDA compatibility aliases
and, because the startup marker is set, loads
`musa_titan.rl.miles_musa.patch_after_import_miles()`.

Do not add a Miles-repo `sitecustomize.py` or `miles/musa_patch_setup.py`.
Ray `worker_process_setup_hook` is not a substitute: job submission only
accepts a module name, the hook runs after worker Python has started, and
it does not cover the Ray job driver or SGLang `multiprocessing` children.

`setup()` remains available for non-Python launchers that already have
`MUSA_TITAN_PATH` on `sys.path` and want an explicit call.  Official Miles
entrypoints do not call it.

## Module layout

The adapter is split by runtime responsibility.  `__init__.py` is kept as
the compatibility facade: it assembles the bootstrap sequence, stores the
patch state, and re-exports the existing public functions and weight-update
entry points.  Miles callers can continue to use imports such as:

```python
from musa_titan.rl.miles_musa import (
    patch_after_import_miles,
    weight_update_backend,
)
```

The implementation modules are:

| Module | Responsibility |
| --- | --- |
| `config.py` | MUSA detection, explicit setup path ordering, MCCL backend selection, `UPDATE_MODE`, and update-weight rendezvous port calculation. |
| `weight_update.py` | Miles Megatron distributed weight-update groups, MCCL process-group creation, tensor transfer modes, rollout-engine splitting, update locks, and broadcast updater patches. |
| `runtime_patches.py` | SGLang optional weight-update endpoints, HF argument compatibility, and SGLang intra-group address handling. Runtime environment variables and Ray device assignment come from the launcher's `runtime_env.yaml` and Ray configuration. |
| `compat.py` | Source-level compatibility rewrites for the supported Miles/Megatron API combinations. |
| `import_hooks.py` | Lazy Miles import finder, post-load patch registry, source-rewrite dispatch, and patching of modules that were imported before the finder was installed. |
| `__init__.py` | Public API, setup state, `setup()`, and the ordered `patch_after_import_miles()` bootstrap assembly. |

The split is organizational only.  It does not move MUSA behavior into the
official Miles checkout and does not change the supported launcher or
environment variables.

## Initialization sequence

With `MILES_MUSA_PATCH_AT_STARTUP=1`, the effective sequence is:

```text
sitecustomize.py
  -> musa_titan.apply_rl_patch()
  -> base Megatron/Torch compatibility
  -> import musa_titan.rl.miles_musa
  -> patch_after_import_miles()
  -> SGLang compatibility and real MUSA capability helper
  -> install Miles import finder
  -> patch already-imported Miles modules
  -> restore the base Torch aliases
```

The initializer keeps `_PATCHING`, `_PATCHED`, `_SETUP_COMPLETE`, and
`_IMPORT_HOOK_INSTALLED` so re-entrant imports remain guarded.  The concrete
patch functions are registered in `import_hooks.py`, while their state is
still observable through the original `musa_titan.rl.miles_musa` module path
for existing tests and callers.

## Compatibility boundary

The refactor preserves:

- the existing public imports and default arguments;
- the `cpu:gloo,musa:mccl` weight-update backend;
- `broadcast`, `gloo-broadcast`, `sync-broadcast`, and `p2p-broadcast`;
- Ray/SGLang child-process startup through the launcher's runtime environment;
- Miles module post-load patching and supported legacy API fallbacks;
- the opt-in startup behavior when the marker is absent or present.

The refactor does not claim support for arbitrary Miles versions.  The
modules continue to target the official Miles `v0.1.0` source layout and
must be revalidated if Miles changes its import paths, actor APIs, SGLang
integration, or weight-update interfaces.
