# RMSNorm tests

Run from the repository root in the existing PyTorch/torch_musa/Triton environment.
Select an available MUSA device with `MUSA_VISIBLE_DEVICES`.

## Operator and integration checks

```bash
PYTHONPATH=.:/path/to/Megatron-LM MUSA_VISIBLE_DEVICES=0 python3 -m pytest \
  musa_titan/rl_align_kernels/tests/ops/rmsnorm -q -p no:cacheprovider
```

- `test_rmsnorm_kernels.py`: BF16/FP16/FP32 forward checks, byte invariance,
  independent FP64 gradient references, native backward, empty/strided inputs,
  graph replay and optimizer updates. This file can also run on its own without
  adding Megatron to `PYTHONPATH`.
- `test_megatron_rmsnorm_integration.py`: actual TE/Megatron normalization paths,
  residual fusion, input/parameter gradients, parameter identity, recomputation,
  graph boundaries and fallback behavior. Requires the compatible Megatron and
  MT-TransformerEngine versions described in the integration module. It uses one
  process with model-parallel size one; it is not a full-model training test.

## SGLang source comparison

```bash
PYTHONPATH=. MUSA_VISIBLE_DEVICES=0 python3 \
  musa_titan/rl_align_kernels/tests/ops/rmsnorm/compare_rmsnorm_sources.py \
  --sglang-source /path/to/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
```

The comparison extracts the RMSNorm definitions from the supplied source without
importing the serving stack. It checks original-dtype bytes and reports source
hashes, comparison counts and `max_abs_diff`. Optional `--rmsnorm-reference` and
`--fused-reference` arguments accept independent reference source files; the
fused reference must expose the four-output API. This does not compare fused add
against the native in-place `sgl_kernel` implementation.

See the [operator contract](../../../kernels/rmsnorm/README.md) for supported
inputs and the scope of backward reuse.
