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
The fused implementation uses one kernel and disables floating-point fusion,
preserving the operator-lab implementation's arithmetic.

## Contract

- Inputs have shape `(..., H)` with `H > 0`; weight has shape `(H,)`.
- Input, weight and optional residual share a GPU device and dtype:
  BF16, FP16 or FP32. Noncontiguous tensors are copied to contiguous storage.
- Outputs use new allocations. No input or residual is modified.
- Empty leading dimensions are supported.
- Epsilon must be finite and positive.
- Ordinary RMSNorm remains forward-only and rejects gradient-enabled inputs
  unless called under `torch.no_grad()` or inference mode.
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

The fused operator preserves operator-lab's numerical contract. This does not
claim bitwise equality with SGLang's native in-place `sgl_kernel.fused_add_rmsnorm`,
which has a different reduction implementation and API.

## Fused backward

When any input requires gradients, a `torch.autograd.Function` saves the FP32
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

## Source provenance

Ported from `operator-lab/rmsnorm` at checkout
`d25080e4631494bf7a2abea458b73f33a4fec3ba`, including the current uncommitted
fused-operator changes adding the FP32 residual cache. The source file hashes
identify the exact inputs:

| Source | SHA256 |
| --- | --- |
| `rmsnorm_batch_invariant.py` | `08de537893d8cce96b84488f2454b9f23b4f920d7a6005b647386d940730ccb6` |
| `fused_add_rmsnorm_batch_invariant.py` | `32c2a5d0a6e2a814a347f92971aad4a8103535664b0152b5d36eba8c98b64ba8` |

The ordinary RMSNorm source records its adaptation from
[vLLM batch_invariant.py](https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649).
It uses the same algorithm as SGLang's `sglang.srt.batch_invariant_ops`.
The port adds imports, input validation, empty-batch handling and explicit
four-warp launch configuration to the extracted ordinary RMSNorm source.

## Validation

From the repository root on a MUSA host:

```bash
PYTHONPATH=. python3 -m pytest test/test_rmsnorm_kernels.py -q
PYTHONPATH=. python3 test/benchmark_rmsnorm_kernels.py
```

To check the migration against explicit source checkouts:

```bash
PYTHONPATH=. python3 test/compare_rmsnorm_sources.py \
  --operator-lab /path/to/operator-lab/rmsnorm \
  --sglang-source /path/to/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
```

This comparison records source hashes and checks original-dtype bytes for
ordinary outputs against both source implementations and all four fused
outputs against operator-lab. SGLang's RMSNorm definitions are loaded directly
from the supplied source, without importing its unrelated serving stack.

Tests cover FP64-reference correctness, original-dtype byte invariance under
batch/position/layout changes, tail widths, non-unit weights, cancellation,
input immutability, independent output storage, strided weights, empty batches,
graph replay with changed inputs and explicit rejection of ordinary RMSNorm
autograd. Fused backward checks include independent FP64 autograd references,
all three input gradients, normalized/residual/combined losses, strided and
aliased inputs, empty batches, native use of the actual forward caches,
cache mutation detection, and two effective optimizer updates.

The benchmark reports forward graph-replay event intervals against FP32
PyTorch reference expressions on the same device. These include replay
overhead and are not isolated kernel timings. The forward benchmark includes
the fourth FP32 cache output. It does not measure backward,
end-to-end training or the native SGLang fused kernel.
