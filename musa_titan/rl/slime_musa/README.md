# slime MUSA Integration

This package contains the MUSA-specific slime weight-update integration.

Shared Megatron patches should be added to `musa_titan/megatron`, while code
that only connects slime components or supports the RL workflow belongs here.
Runnable launch scripts and configurations belong in
`examples/rl/slime_musa`.

Set `SLIME_MUSA_PATCH_AT_STARTUP=1` in slime training processes (`true`,
`yes`, and `slime` are also accepted), then call
`musa_titan.apply_rl_patch()` (or rely on this repository's `sitecustomize.py`)
to install the hooks.

`patch_after_import_slime()` imports Slime's weight-update module and installs
the MUSA hooks. Importing `musa_titan.rl.slime_musa` itself does not import
Slime, which avoids the circular import created when Slime's MUSA accelerator
calls `apply_rl_patch()` while that module is still loading.
