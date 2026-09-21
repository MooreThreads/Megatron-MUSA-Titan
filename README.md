## About

Megatron-MUSA-Titan focuses on pretraining and reinforcement learning infrastructure for **1T+ parameter models on clusters of 10,000+ Moore Threads GPUs**. Built around [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), it brings MUSA-specific optimizations to pretraining and reinforcement learning while staying close to the upstream ecosystem.

## Highlights

- **Trillion-parameter pretraining at scale.**. Achieved high Model FLOPs Utilization (MFU) and stable training during Kimi 1000B pretraining on a 10,000+ Moore Threads GPUs.
- **One training foundation for pretraining and RL.** Supports Megatron pretraining and reinforcement learning with [slime](https://github.com/THUDM/slime) and [miles](https://github.com/radixark/miles), using Megatron for training and [SGLang](https://github.com/sgl-project/sglang) for rollout generation.
- **Work with upstream repositories.** With [torch_musa](https://github.com/MooreThreads/torch_musa)'s PyTorch integration and [MUSA C](https://docs.mthreads.com/musa-sdk/version-5.2.0/introduction)'s compatibility with CUDA C, developers can use official Megatron-LM, slime, and miles repositories on Moore Threads GPUs. With a compatible MUSA environment, users can simply clone the official Megatron-LM, slime, and miles repositories and run training on Moore Threads GPUs **without modifying the upstream code**.


## What belongs here

Our focus is training infrastructure on Moore Threads GPUs, rather than model implementations or training algorithms. We concentrate on efficient computation, communication, memory management, and distributed training across 10,000+ Moore Threads GPUs.

We follow an upstream-first approach: general training algorithms or bug fix should be contributed to the relevant upstream projects. This repository retains two kinds of changes:

- **MUSA-specific workarounds** for GPU or software-stack issues that cannot yet be addressed upstream.
- **Large-scale training optimizations** tailored to Moore Threads GPUs and cluster architecture, shared between pretraining and RL wherever possible.

For RL, SGLang serves as the rollout engine. Standalone inference optimization is outside this repository's scope.

## Repository layout

```text
musa_titan/
├── megatron/       # Shared Megatron adaptations and optimizations
├── pretrain/       # Pretraining-specific functionality
└── rl/
    └── slime_musa/ # Framework-specific RL integration

examples/
├── pretrain/       # Model-specific pretraining recipes
└── rl/
    └── slime/ # Directory reserved for slime launch recipes
```

RL reuses the shared Megatron functionality rather than duplicating it. The current RL directories provide the integration structure; framework-specific launch recipes are not yet included in this checkout.

## Getting started

### 1. Prepare the environment

Install a compatible Moore Threads driver, MUSA toolkit, PyTorch with [torch_musa](https://github.com/MooreThreads/torch_musa), and [MT-TransformerEngine](https://github.com/MooreThreads/MT-TransformerEngine). Install any additional dependencies required by your selected recipe.

Use the official [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repository. For RL, also prepare the official slime or miles repository and a MUSA-compatible SGLang environment.

The pretraining examples generally expect Megatron-LM alongside this checkout:

```text
workspace/
├── Megatron-LM/
└── megatron-musa-titan/ # This repository;
```

Make both repositories available on `PYTHONPATH`. From this repository's root:

```bash
export PYTHONPATH="$PWD:$PWD/../Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

Our goal is to work with upstream main branches, not require long-lived framework forks. Upstream APIs and dependencies evolve, so validate framework revisions together with your MUSA environment before a large-scale run.

### 2. Enable the training adaptations

Add the following environment variable to your Bash launch script before starting training:

```bash
export ACCELERATOR_BACKEND="musa"
```

Then add the following code at the beginning of your Python training entry point, before importing Megatron modules:

```python
import os

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    from musa_titan import apply_pretrain_patch
    apply_pretrain_patch()
```

### 3. Configure and launch a recipe

Start with the [pretraining examples](examples/pretrain), including [Kimi-K2](examples/pretrain/kimi-k2), [DeepSeek-V3](examples/pretrain/deepseek-v3), and [Llama3](examples/pretrain/llama3).

Before launching, review the selected scripts and configure dataset and tokenizer paths, checkpoints, node counts, parallelism, and cluster networking. For SSH-based multi-node launchers, prepare a `hostfile` in the model's example directory with one node address per line, and ensure consistent environments and paths across nodes.

Run the launcher from its model directory unless the recipe states otherwise. For example, after configuring the DeepSeek-V3 recipe:

```bash
cd examples/pretrain/deepseek-v3
bash run_deepseekv3.sh
```

Recipes are starting points for cluster-specific configuration, not one-command reproductions of the reported 10,000-GPU benchmark. See [RL examples](examples/rl) for the current RL recipe structure.

## Contributing

Welcome any form of contribution of code, model implementation and document!

For MUSA-specific function replacements, see the [maintenance guide](musa_titan/utils/PATCHING.md). General-purpose fixes should be proposed upstream whenever practical.

## Acknowledgement

Built on the work of the Megatron-LM, torch_musa, TransformerEngine, slime, miles. Initial development also leveraged [FlagScale](https://github.com/FlagOpen/FlagScale).
