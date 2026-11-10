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
- Importing this module does not install Megatron, TE or SGLang replacements.
  In particular, it does not change training's existing RMSNorm backward.
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
to be batch invariant or bitwise identical to SGLang backward. This does not
install a Megatron layer adapter or establish end-to-end RL alignment.

## Upstream references

The ordinary RMSNorm implementation is adapted from
[vLLM batch_invariant.py](https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649).
It uses the same algorithm as SGLang's `sglang.srt.batch_invariant_ops`.
Titan adds input validation, empty-batch handling, explicit four-warp launch
configuration and an optional rstd store for native backward.

## Validation

From the repository root on a MUSA host:

```bash
PYTHONPATH=. python3 -m pytest test/test_rmsnorm_kernels.py -q
```

To check the migration against explicit source checkouts:

```bash
PYTHONPATH=. python3 test/compare_rmsnorm_sources.py \
  --sglang-source /path/to/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
```

This comparison records source hashes and checks original-dtype bytes for
ordinary outputs against SGLang. Optional `--rmsnorm-reference` and
`--fused-reference` arguments accept additional standalone source files.
It also checks that ordinary RMSNorm's autograd-enabled forward matches its
inference output byte-for-byte.
SGLang's RMSNorm definitions are loaded directly
from the supplied source, without importing its unrelated serving stack.

Tests cover FP64-reference correctness, original-dtype byte invariance under
batch/position/layout changes, tail widths, non-unit weights, cancellation,
input immutability, independent output storage, strided weights, empty batches,
graph replay with changed inputs. Ordinary backward checks include independent
FP64 autograd references, input-only/weight-only/joint gradients, strided and
empty inputs, byte-exact comparison with the native backward given the same
saved rstd, saved-tensor mutation detection and two effective optimizer updates.
Fused backward checks include independent FP64 autograd references,
all three input gradients, normalized/residual/combined losses, strided and
aliased inputs, empty batches, native use of the actual forward caches,
cache mutation detection, and two effective optimizer updates.

## Benchmark

```bash
PYTHONPATH=.:/path/to/Megatron-LM python3 test/benchmark_rmsnorm_kernels.py \
  --output logs/rmsnorm-native-benchmark
```

The default matrix uses BF16, H=2048 and rows 16, 128, 4096 and 131072
(32 x 4096). Both operators have forward and combined forward+backward
measurements. Forward runs with gradients enabled, including backward-cache
writes. The combined path includes the entire forward and gradient calculation;
input preparation and warmup/compilation are outside timing.

The baseline uses MT-TE's native MUSA RMSNorm ATen operators. For residual-add,
it calls the imported Megatron `bias_dropout_add_fused_train` with no bias and
zero dropout, followed by native RMSNorm. Megatron rounds its added residual
to BF16 before normalization; Titan preserves the FP32 sum until normalization.
These paths intentionally have different rounding semantics.

Reported microseconds are the median of three round means, each summing
GPU profiler kernel durations over 50 calls and dividing by 50. This includes
all GPU kernels in the selected path, but excludes CPU launch gaps and does
not represent end-to-end training latency. Traces, source hashes and per-round
samples are saved in the selected output directory.

Measured on one MTT S5000 with PyTorch 2.7.1, torch_musa 2.7.1+5f0ecd1 and
Triton 3.2.0. Ratios below 1 mean Titan is slower. These are operator-level
GPU execution times, not a full-model forward or training-step measurement.

### RMSNorm forward

| Rows (H=2048) | Titan (us) | Native Megatron (us) | Megatron / Titan |
| --- | ---: | ---: | ---: |
| 16 | 6.11 | 8.72 | 1.43x |
| 128 | 6.25 | 8.68 | 1.39x |
| 4096 | 29.05 | 29.65 | 1.02x |
| 32 x 4096 = 131072 | 804.50 | 919.23 | 1.14x |

### RMSNorm forward + backward

| Rows (H=2048) | Titan (us) | Native Megatron (us) | Megatron / Titan |
| --- | ---: | ---: | ---: |
| 16 | 22.02 | 25.18 | 1.14x |
| 128 | 35.66 | 38.67 | 1.08x |
| 4096 | 110.72 | 109.25 | 0.99x |
| 32 x 4096 = 131072 | 3252.83 | 3349.66 | 1.03x |

### Fused add RMSNorm forward

| Rows (H=2048) | Titan (us) | Native Megatron (us) | Megatron / Titan |
| --- | ---: | ---: | ---: |
| 16 | 6.11 | 11.02 | 1.80x |
| 128 | 6.69 | 12.14 | 1.81x |
| 4096 | 84.73 | 68.90 | 0.81x |
| 32 x 4096 = 131072 | 2891.34 | 2075.93 | 0.72x |

### Fused add RMSNorm forward + backward

| Rows (H=2048) | Titan (us) | Native Megatron (us) | Megatron / Titan |
| --- | ---: | ---: | ---: |
| 16 | 41.70 | 31.85 | 0.76x |
| 128 | 60.13 | 46.45 | 0.77x |
| 4096 | 422.44 | 198.95 | 0.47x |
| 32 x 4096 = 131072 | 14255.88 | 5652.24 | 0.40x |

Ordinary forward is faster at all four tested sizes; combined ordinary
forward+backward is close to parity at 4096 rows. Fused forward is faster at
16/128 rows and slower at the larger sizes. Fused forward+backward is slower
at all four sizes; the measurement includes the FP32 cache, casts and FP32
native backward required by its unrounded-sum contract.
