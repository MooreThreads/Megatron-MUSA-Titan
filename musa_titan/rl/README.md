# Reinforcement Learning

Reinforcement-learning integrations and optimizations live here.

Currently supported frameworks:

- [slime_musa](slime_musa/README.md)
- [miles_musa](miles_musa/README.md)

RL training reuses the shared Megatron patches from `musa_titan/megatron`.
SGLang is treated as an external inference dependency; this repository does not
currently maintain SGLang patches.

## Load path

Add this repository to `PYTHONPATH`, then apply the RL entry point before
importing Megatron, slime, or miles training modules:

```python
from musa_titan import apply_rl_patch

apply_rl_patch()
```

`apply_rl_patch()` always installs the shared Megatron patches. Framework-specific
hooks remain opt-in:

- `SLIME_MUSA_PATCH_AT_STARTUP=1` loads `musa_titan.rl.slime_musa`
  (`true`, `yes`, and `slime` are also accepted)
- `MILES_MUSA_PATCH_AT_STARTUP=1` loads `musa_titan.rl.miles_musa`
  (`true`, `yes`, and `miles` are also accepted)

Putting this repository first on `PYTHONPATH` also loads the repo-root
`sitecustomize.py`, which calls `apply_rl_patch()` when one of those markers is
set. That covers Ray workers and SGLang/Megatron child processes without adding
a slime or miles `sitecustomize.py`.
