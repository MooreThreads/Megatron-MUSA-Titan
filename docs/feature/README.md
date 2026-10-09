# Feature usage and environment variables

This guide covers the Pretraining and RL features listed in the project README.
Set environment variables in the Bash script that starts **each training worker**,
before Python imports the training libraries. An export on the launcher node is
not automatically forwarded through SSH or Ray. Existing example scripts may
also overwrite exported values; edit the worker-side script when necessary.

## Basic setup

```bash
export ACCELERATOR_BACKEND="musa"
# Run from this repository's root; adjust the Megatron checkout path as needed.
export PYTHONPATH="$PWD:$PWD/../Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `ACCELERATOR_BACKEND` | `musa` in the example Python entry points | Selects the MUSA initialization path. It does not activate adaptations without the Python call below. |
| `PYTHONPATH` | Environment-dependent | Makes this repository and the selected Megatron checkout importable. |

At the beginning of the pretraining entry point, before importing Megatron:

```python
import os

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    from musa_titan import apply_pretrain_patch
    apply_pretrain_patch()
```

## Pretraining

### Blockwise FP8 training

The [Kimi-K2 1T MoE recipe](../examples/pretrain/kimi-k2/kimi-k2/pretrain_kimi_1000B.sh)
accepts this shell switch:

The recipe adds these Megatron arguments:

```text
--transformer-impl transformer_engine
--fp8-format e4m3
--fp8-param-gather
--fp8-recipe mxfp8
```

The MT-TransformerEngine maps `MXFP8BlockScaling` to its
`MTFP8BlockScaling` implementation. 

### DeepEP over ACE

```bash
export USE_DEEPEP_ACE=1
```

Add to the Megatron arguments:

```text
--moe-token-dispatcher-type flex
--moe-enable-deepep
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `USE_DEEPEP_ACE` | `0` | Loads the ACE-specific DeepEP dispatcher adaptations. |

This requires an ACE-capable DeepEP build exposing `get_ace_combine_buffer`


### EP communication overlap over DeepEP/ACE

Keep the DeepEP/ACE configuration above and add:

```text
--overlap-moe-expert-parallel-comm
```

There is no separate Titan environment variable for EP overlap. The flag enables
batch-level EP communication overlap during the VPP. `USE_DEEPEP_ACE=1`
alone does **not** enable ep overlap.

`--delay-wgrad-compute` is an optional companion flag, not a prerequisite. It
requires EP overlap and a supported non-legacy grouped-GEMM implementation.

### Activation offloading with VPP and 1F1B

Enable offloading through customized Megatron arguments, not an environment switch:

```text
--offload-moe-fc1-input
--offload-moe-fused-swiglu-input
```

These offload routed-expert FC1 inputs and fused SwiGLU inputs, respectively.
The current Titan configuration requires **PP size >= 3**.


### RDMA-backend PP communication overlap

We use Mooncake P2P RDMA to overlap pipeline-parallel (PP) communication with
computation under virtual pipeline parallelism (VPP). Activations and gradients
are transferred between pipeline stages through the RDMA backend, allowing
communication to overlap with the execution of other virtual stages.

Enable the Mooncake backend with the following environment variables in each
training worker's launch script. Configure a valid VPP layout and enable the
corresponding P2P overlap options in your Megatron training configuration;
selecting the RDMA backend alone does not enable scheduling overlap.

```bash
export PP_USE_MOONCAKE=1
export MOONCAKE_P2P_SLOT_SIZE_BYTES=134217728
export MC_WORKERS_PER_CTX=2
export MC_SLICE_SIZE=65536
```

This requires a Mooncake-enabled Megatron integration and working RDMA
connectivity between the training nodes. The 128 MiB slot size above is an
example; adjust it to accommodate the largest activation or gradient transfer
in your pipeline configuration.

### Trace collection for [Distributed Holistic Trace Analysis](https://github.com/yongpan0709/DistributedHolisticTraceAnalysis)

We provide two complementary profiling and trace analysis tools:

1. **Single-GPU PyTorch call-stack analysis.** Reconstructs and analyzes PyTorch
   operator call stacks from a GPU's trace, helping connect execution costs to
   the corresponding Python code. It can be used with any program written in
   PyTorch, not just Megatron, and is not tied to a specific accelerator backend.
   Set `with_stack=True` in `torch.profiler.profile(...)` when collecting traces
   to record the call-stack information required by the analysis.

2. **Large-scale distributed Megatron trace analysis.** Analyzes traces across
   training ranks to identify expensive operators, communication bottlenecks,
   pipeline bubbles, and straggler nodes, helping prioritize performance
   optimizations at cluster scale. It can also be used with Megatron training
   on other accelerator backends
