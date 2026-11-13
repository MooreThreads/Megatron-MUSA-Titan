# Batch-invariant RMSNorm operators

```python
import torch

from musa_titan.kernels import (
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
- Importing `musa_titan.kernels` exposes the operator APIs. Calling
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
the kernel benchmarks below are not full-model or multi-rank performance results.

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

## Benchmark

The comparison uses the previously extracted batch-invariant forward reference,
not the installed library's automatic forward dispatcher. Ordinary reference
RMSNorm directly launches the extracted `LayerNormGlobalKernelVlen`. The
add+RMSNorm reference uses the standalone extracted `bias_dropout_add_fused_train`
(no bias, dropout=0), then that same fixed-dispatch RMSNorm.

The library exports `launch_megatron(input, weight, output, rstd, rows, stream)`.
The recorded benchmark checked the BF16/H=2048/eps=1e-6/DIRECT/block/vlen
contract, retained source/library hashes and compiler flags, and recorded actual
kernel grid/block parameters from profiler traces. It used the extracted
reference source and library without falling back to installed-library forward.

**Forward dispatch (all four shapes):**

- Extracted RMSNorm: `DIRECT`, `BLOCK_X=1024`, `BLOCK_Y=1`, `vlen=8`,
  `grid=(M,1,1)`, `block=(1024,1,1)`, dynamic shared memory=0. BF16 input,
  output and weight; FP32 statistics. Built with `-O3 -fno-fast-math
  -ffp-contract=off`, `mp_31`.
- Titan ordinary and fused: `BLOCK_SIZE=1024`, `num_warps=4`,
  `grid=(M,1,1)`, observed `block=(128,1,1)`; fused disables FP fusion.
- The extracted residual add uses BF16 elementwise addition. Its actual
  grid/block and the reference backward dispatches are listed below.

For combined forward+backward, the reference uses native MUSA
`aten::_fused_rmsnorm_backward` with the **extracted forward's saved rstd**.
The available standalone extraction is forward-only; this is not a claim that
backward kernels were separately extracted or that backward is batch invariant.
Actual reference add/backward dispatches (dimensions are x/y/z):

```text
M=16: add: block=1024x1x1, grid=4x1x1; dX: block=64x1x1, grid=16x1x1; dWeight: block=64x16x1, grid=32x1x1
M=128: add: block=1024x1x1, grid=32x1x1; dX: block=64x1x1, grid=128x1x1; dWeight: block=16x16x1, grid=16x16x1; dWeight_merge: block=128x1x1, grid=16x1x1
M=4096: add: block=1024x1x1, grid=1024x1x1; dX: block=64x1x1, grid=4096x1x1; dWeight: block=16x16x1, grid=16x16x1; dWeight_merge: block=128x1x1, grid=16x1x1
M=131072: add: block=1024x1x1, grid=32768x1x1; dX: block=16x32x1, grid=4096x1x1; dWeight: block=16x32x1, grid=16x16x1; dWeight_merge: block=128x1x1, grid=16x1x1
```

These are single-operator timings, not full-model forward. The combined path
includes the complete forward plus input/weight gradients and, for fused add,
the residual-output gradient branch. Same inputs, weights and upstream gradients
are used for both implementations. Forward runs with gradients enabled.

Timing: one S5000, BF16, H=2048, eps=1e-6, seed=2231; five warmup calls and
three rounds of 50 calls. Each round sums all GPU kernel durations and divides
by 50; tables show the median round mean in microseconds. CPU launch gaps,
compilation and input preparation are excluded. `Ref / Titan > 1` means Titan
is faster **than this fixed-dispatch extraction**, not the installed dispatcher.

`max_abs_diff` compares every element of the corresponding original-dtype
results after conversion to FP32. It is a cross-implementation comparison, not
the sampled CPU-reference check. Differences are measured outside timing.
Upstream gradients are unscaled random BF16 tensors; dWeight is batch-summed.
The extracted add rounds its sum to BF16 before RMSNorm, while Titan normalizes
the unrounded FP32 sum. Batch invariance of each fixed forward implementation
does not imply byte equality between implementations.

### RMSNorm forward

| Rows | Titan (us) | Extracted reference (us) | Ref / Titan | max_abs_diff y |
| --- | ---: | ---: | ---: | ---: |
| 16 | 6.03 | 5.89 | 0.98x | 0 |
| 128 | 6.19 | 13.32 | 2.15x | 0.00390625 |
| 4096 | 29.25 | 305.68 | 10.45x | 0.015625 |
| 32 x 4096 | 803.36 | 6296.98 | 7.84x | 0.03125 |

### RMSNorm forward + backward

| Rows | Titan (us) | Extracted reference (us) | Ref / Titan | max_abs_diff y | max_abs_diff dX | max_abs_diff dWeight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 22.19 | 22.86 | 1.03x | 0 | 0 | 0 |
| 128 | 35.71 | 42.86 | 1.20x | 0.00390625 | 0.015625 | 0 |
| 4096 | 111.19 | 318.94 | 2.87x | 0.015625 | 0.015625 | 0 |
| 32 x 4096 | 3251.93 | 8725.22 | 2.68x | 0.03125 | 0.03125 | 0 |

### Fused add RMSNorm forward

| Rows | Titan (us) | Extracted reference (us) | Ref / Titan | max_abs_diff y | max_abs_diff residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 6.08 | 10.82 | 1.78x | 0.03125 | 0 |
| 128 | 6.68 | 17.64 | 2.64x | 0.03125 | 0 |
| 4096 | 86.38 | 328.90 | 3.81x | 0.03125 | 0 |
| 32 x 4096 | 2879.63 | 7387.48 | 2.57x | 0.03125 | 0 |

### Fused add RMSNorm forward + backward

| Rows | Titan (us) | Extracted reference (us) | Ref / Titan | max_abs_diff y | max_abs_diff residual | max_abs_diff dX | max_abs_diff dResidual | max_abs_diff dWeight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 41.63 | 32.19 | 0.77x | 0.03125 | 0 | 0.03125 | 0.03125 | 0.0625 |
| 128 | 59.62 | 52.62 | 0.88x | 0.03125 | 0 | 0.03125 | 0.03125 | 0.25 |
| 4096 | 412.03 | 468.77 | 1.14x | 0.03125 | 0 | 0.03125 | 0.03125 | 1 |
| 32 x 4096 | 14257.65 | 10959.57 | 0.77x | 0.03125 | 0 | 0.03125 | 0.03125 | 8 |

The four measured shapes are 16, 128, 4096 and 32 x 4096 (=131072) rows. Ordinary
RMSNorm matches exactly at M=16 in this run; larger shapes expose output/input-
gradient differences. Fused residual outputs match at all four shapes, while
normalized outputs and gradients differ. Raw traces, full-tensor differences,
per-round timing and dispatch records are retained by the benchmark.
