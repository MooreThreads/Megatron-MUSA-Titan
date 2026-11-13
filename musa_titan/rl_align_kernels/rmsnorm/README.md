# Batch-invariant RMSNorm operators

```python
import torch

from musa_titan.rl_align_kernels.rmsnorm import (
    rms_norm_batch_invariant,
    fused_add_rmsnorm_batch_invariant,
)

with torch.no_grad():
    normalized = rms_norm_batch_invariant(x, weight, eps=1e-6)
    normalized, residual_sum, rstd, residual_sum_fp32 = fused_add_rmsnorm_batch_invariant(
        x, residual, weight, eps=1e-6,
    )
```

Both implementations use Triton, one program per row, 1024-column chunks and
four warps. Batch size does not select a different reduction algorithm.
The fused implementation uses one kernel and disables floating-point fusion
to preserve the specified arithmetic.

## Layout

```text
rmsnorm/
├── __init__.py
├── kernels/
│   ├── __init__.py
│   ├── rmsnorm.py
│   └── fused_add_rmsnorm.py
├── tests/
│   └── .gitkeep
├── benchmarks/
│   └── README.md
└── README.md
```

`kernels/` contains the implementations. `tests/` reserves the test location;
test scripts are not distributed in this change. `benchmarks/` contains the
[recorded operator measurements](benchmarks/README.md).

## Contract

- Inputs have shape `(..., H)` with `H > 0`; weight has shape `(H,)`.
- Input, weight and optional residual share a GPU device and dtype:
  BF16, FP16 or FP32. Noncontiguous tensors are copied to contiguous storage.
- Outputs use new allocations. No input or residual is modified.
- Empty leading dimensions are supported.
- Epsilon must be finite and positive.
- Ordinary RMSNorm supports first-order MUSA gradients for input and weight.
  It returns only normalized output; forward rstd is cached internally only
  when gradients are needed. Do not modify saved input or weight before backward.
- Fused add RMSNorm supports first-order MUSA gradients for input, residual
  and weight. `normalized` and `residual_sum` are differentiable; `rstd` and
  `residual_sum_fp32` are non-differentiable caches. Do not mutate these caches
  or weight before backward. Higher-order gradients are unsupported.
- Importing `musa_titan.rl_align_kernels.rmsnorm` exposes the operator APIs. Calling
  `apply_pretrain_patch()` or `apply_rl_patch()` installs the Megatron/TE
  integration described below.
- The package requires the existing Titan PyTorch/torch_musa/Triton environment.
  CUDA forward is accepted by the tensor contract but is not validated by the
  MUSA tests; CUDA autograd is rejected.

Ordinary RMSNorm follows SGLang's **batch-invariant Triton** path:

```text
s = FP32(input)
rstd = 1 / sqrt(sum(s*s) / H + eps)
normalized = cast_input_dtype((s * rstd) * FP32(weight))
```

Fused add uses `s = FP32(input) + FP32(residual)` **before rounding** for both
the statistics and normalized output. It additionally returns
`residual_sum = cast_input_dtype(s)` and the actual FP32 `rstd`, shaped as
`input.shape[:-1]`. The fourth output, `residual_sum_fp32`, stores the unrounded
`s` with `input.shape` in an independent allocation. It adds four bytes per
input element to forward output storage. It is not equivalent to first rounding `input + residual`
to BF16/FP16 and then calling ordinary RMSNorm.

This numerical contract does not claim bitwise equality with SGLang's native
in-place `sgl_kernel.fused_add_rmsnorm`,
which has a different reduction implementation and API.

## Megatron integration

Call the training entry point **before constructing the model**:

```python
from musa_titan import apply_rl_patch  # or apply_pretrain_patch
apply_rl_patch()
```

The shared Megatron installer applies `musa_titan.megatron.rmsnorm.apply_patch()`
once. All replacements use the patch audit helper and report the Megatron
checkout revision and installed Transformer Engine package version.

- **Standalone norms:** TE `RMSNorm.forward` and the RMSNorm branch of Titan's
  Megatron `FusedLayerNorm` use `rms_norm_batch_invariant`, including Q/K and
  final norms built through `TENorm`. Parameters, names and sequence-parallel
  attributes are preserved. Standalone zero-centered gamma is supported.
- **Norm fused with Linear/MLP:** TE's `rmsnorm_fwd` entry point and its explicit
  extension/operation-fuser aliases use the Triton forward for
  non-quantized, non-zero-centered MUSA RMSNorm. The kernel writes directly
  into TE's output buffer and returns the actual rstd. TE retains GEMM,
  communications and its existing native backward.
- **Residual-add + norm:** newly constructed decoder `TransformerLayer`s with
  a standalone pre-MLP RMSNorm fuse self-attention's residual add with that
  norm. The fused kernel returns both the normalized MLP input and residual;
  the following norm call consumes the normalized result once. The original
  layer schedule, MLP and parameter ownership are preserved. Tensor-local
  handoff avoids a module-level cache across microbatches or recomputation.

Residual fusion requires identity cross-attention placeholders, no bias or
dropout, matching activation dtypes and no selective pre-MLP norm recomputation,
FP8, activation offload, FSDP, attention-only graph scope or norm input/backward
hooks. Other paths keep
their original residual add and use ordinary RMSNorm where supported.
Full-layer checkpoint recomputation is supported. Inter-layer/final MLP
residual adds remain on Megatron's original schedule; this patch does not
reorganize a TransformerBlock into an SGLang decoder schedule.

LayerNorm, unsupported devices/shapes, quantized normalization and zero-centered
normalization inside TE fused Linear/MLP retain their original implementations.
The integration has bounded model-parallel-size-one correctness coverage;
the [kernel benchmarks](benchmarks/README.md) are not full-model or multi-rank
performance results.

## Native backward

Ordinary RMSNorm saves input, weight and the actual Triton forward rstd, then
calls `torch.ops.aten._fused_rmsnorm_backward` with the original input/weight
dtype, as in Megatron/MT-TE. It does not recompute forward statistics.
Noncontiguous operands are flattened and made contiguous for the native call;
input gradients are reshaped back to the original input dimensions. Empty
batches return empty input gradients and zero weight gradients.

```python
# x or weight must have requires_grad=True.
rms_norm_batch_invariant(x, weight).float().square().mean().backward()
```

For fused add RMSNorm, a `torch.autograd.Function` saves the FP32
sum, the actual forward rstd and weight. Backward calls
`torch.ops.aten._fused_rmsnorm_backward` with FP32 operands. This is the native
operator called by MT-TransformerEngine's `rmsnorm_bwd` for Megatron, whose
MUSA backend invokes muDNN `RMSNorm::RunBwd`. No TE import or replacement of
upstream framework code is needed.

The adapter supplies the cached FP32 sum, rather than the rounded residual
output, and uses the cached rstd without rerunning forward. It adds the gradient
from `residual_sum` in FP32, then converts the result to input dtype for both
input branches. Weight gradients are converted back to weight dtype. Empty
batches and residual-only losses bypass the native normalization backward.

```python
# x, residual or weight must have requires_grad=True.
y, added, rstd, sum_fp32 = fused_add_rmsnorm_batch_invariant(x, residual, weight)
(y.float().square().mean() + added.float().square().mean()).backward()
```

Forward remains batch invariant. Native backward reductions are not claimed
to be batch invariant or bitwise identical to SGLang backward. These operator
properties do not establish end-to-end RL alignment.
