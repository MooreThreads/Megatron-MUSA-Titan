# Miles MUSA Examples

Miles launch scripts and MUSA configurations live here.

Python patches and reusable integration code belong in
`musa_titan/rl/miles_musa`.

Enable the Miles hooks with `MILES_MUSA_PATCH_AT_STARTUP=1` (`true`, `yes`,
and `miles` are also accepted) and this repository on `PYTHONPATH`.
`sitecustomize.py` then calls `musa_titan.apply_rl_patch()` before Miles
training modules are imported.

Before launching, copy `set_env.sh.example` to the local, ignored
`set_env.sh` and set the paths for the model, checkpoint, dataset, Miles,
Megatron, and MUSA Titan checkouts. The launchers expand `runtime_env.yaml`
from that file before submitting the Ray job; the template itself contains no
machine-specific paths. Set `RAY_HEAD_IP` in `set_env.sh`, or create the
ignored `hostfile` with one Ray node per line, before launching.
