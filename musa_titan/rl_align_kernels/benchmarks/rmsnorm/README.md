# RMSNorm benchmarks

[Operator APIs and numerical contracts](../../kernels/rmsnorm/README.md).

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
