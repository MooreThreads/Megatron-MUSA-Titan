# VERL + MUSA 使用指南

当前适配分为两层：

1. `verl-hardware-plugin`：通过 VERL 插件接口注册 MUSA 平台和训练引擎。
2. `musa_titan.rl.verl_musa`：在不修改 VERL 源码的情况下，为 VERL 安装必要的运行时 patch。

```text
VERL
  ├─ verl-hardware-plugin
  │    ├─ PlatformMUSA：设备、MCCL、MUSA_VISIBLE_DEVICES、Ray 资源
  │    └─ EngineRegistry：MUSA FSDP/Megatron engine
  │
  └─ megatron-musa-titan
       ├─ sitecustomize.py：启动时安装延迟 import hook
       ├─ musa_titan/megatron：Megatron/Torch-MUSA 公共适配
       └─ musa_titan/rl/verl_musa：VERL/SGLang/TE 运行时 patch
```

## 1. VERL 如何通过插件适配 MUSA

### 1.1 插件加载

`verl-hardware-plugin` 通过 Python entry point 向 VERL 暴露插件：

```toml
[project.entry-points."verl.plugins"]
hardware = "verl_hardware_plugin"
```

本目录的 `runtime_env.yaml` 同时显式设置：

```yaml
VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
VERL_PLATFORM: "musa"
```

`VERL_PLATFORM=musa` 让 VERL 选择 MUSA platform，避免在多种加速器并存的环境中误判设备。

### 1.2 PlatformMUSA

MUSA platform 向 VERL 提供统一的硬件抽象：

| 能力 | MUSA 实现 |
| --- | --- |
| device name | `musa` |
| vendor | `moore_threads` |
| PyTorch device module | `torch.musa` |
| 通信后端 | `mccl` |
| 设备可见性变量 | `MUSA_VISIBLE_DEVICES` |
| Ray 调度资源 | `GPU` |
| Ray no-set 变量 | `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES` |
| IPC | 支持 |

由于当前 Ray 使用 `GPU` 资源调度 MUSA 设备，但不应用 CUDA 规则重写 `MUSA_VISIBLE_DEVICES`，启动环境中设置了：

```yaml
RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES: "1"
```

### 1.3 MUSA engine 注册

插件为 `device=musa, vendor=moore_threads` 注册了：

- FSDP/FSDP2 language-model 和 value-model engine。
- Megatron language-model 和 value-model engine。

这些 engine 使 VERL 可以通过正常配置选择 MUSA 后端，例如：

```bash
model_engine=megatron
trainer.device=musa
+actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
```

插件层负责“让 VERL 认识 MUSA”；具体的 Torch-MUSA、Megatron、SGLang 兼容修复由 Titan 的运行时 patch 层完成。

## 2. VERL patch 如何在 MUSA Titan 中生效

### 2.1 启动入口

Ray Job 的 `runtime_env.yaml` 设置：

```yaml
PYTHONPATH: "${RUNTIME_PYTHONPATH}"
VERL_MUSA_PATCH_AT_STARTUP: "1"
```

`RUNTIME_PYTHONPATH` 的第一项是 `MUSA_TITAN_PATH`。因此 Ray driver、worker 和继承该环境的子进程启动 Python 时，会自动执行 Titan 根目录的 `sitecustomize.py`。

`VERL_MUSA_PATCH_AT_STARTUP=1` 表示“启用 VERL MUSA 自动 patch”。`sitecustomize.py` 不会立即导入整套 Torch-MUSA/VERL/Megatron，而是先安装两个 `sys.meta_path` finder：

- `_VerlBootstrapFinder`：监听根模块 `verl` 和 `megatron`。
- `_MusaModelRunnerFinder`：监听 `sglang.srt.model_executor.model_runner`。

运行时时序：

```text
Python 启动
  → sitecustomize.py
  → 检测 VERL_MUSA_PATCH_AT_STARTUP=1
  → 安装 import finder，暂不加载重型 patch
  → import verl / import megatron
  → 执行 musa_titan.apply_rl_patch()
  → 安装 Megatron 公共 patch 和 VERL-specific patch
```

SGLang scheduler 可能不导入 VERL，所以 `model_runner` 使用独立的 post-import hook：先让 SGLang 正常定义 `LocalSerializedTensor`，再恢复原生 MUSA capability 并安装 MTP IPC tensor cache。

`multiprocessing.resource_tracker` 会被排除，避免纯资源跟踪进程加载 Torch-MUSA 和 Megatron。


## 3. 配置与启动


### 3.1 配置 `set_env.sh`

在 `examples/rl/verl/` 下创建本机专用的 `set_env.sh`

```bash
#!/usr/bin/env bash

export MUSA_TITAN_PATH="/path/to/megatron-musa-titan"
export MEGATRON_PATH="/path/to/Megatron-LM"
# We can install a version of VERL later than 0.9.0 and pin it explicitly.
export VERL_PATH="/path/to/verl"
# Megatron Bridge can be skipped if it is already installed in the environment.
export MEGATRON_BRIDGE_PATH="/path/to/megatron-bridge/src"


export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MCCL_LIB="${MCCL_LIB:-/usr/local/musa/lib/libmccl.so}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export MUSA_LD_LIBRARY_PATH="${MUSA_LD_LIBRARY_PATH:-/usr/lib:/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/lib:/usr/local/mtshmem/lib:/usr/local/openmpi/lib}"
export RUNTIME_PYTHONPATH="${MUSA_TITAN_PATH}:${VERL_PATH}:${MEGATRON_PATH}:${MEGATRON_BRIDGE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export RUNTIME_LD_LIBRARY_PATH="${MUSA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

`RUNTIME_PYTHONPATH` 必须以 `MUSA_TITAN_PATH` 开头，否则 Python 可能命中其他环境中的 `sitecustomize.py`，Titan hook 将不会自动安装。

### 3.2 配置节点

在 `examples/rl/verl/hostfile` 中每行写一个节点 IP，第一行是 Ray head：

```text
10.0.0.1
10.0.0.2
10.0.0.3
10.0.0.4
```

确保各节点：

- 能使用 SSH 互访。
- 能访问相同的模型、数据集和共享路径。
- Python/MUSA/MCCL/VERL/SGLang 环境一致。

### 3.3 配置并提交训练

当前 4 节点 Qwen3-30B GRPO 脚本是：

```text
run_qwen3-30b_grpo_node4_32k.sh
```

模型、数据、checkpoint 和自定义 reward 文件必须使用所有节点可见的路径。以下路径均为脱敏示例：

```bash
cd megatron-musa-titan/examples/rl/verl
HF_MODEL_PATH=/shared/models/Qwen3-30B-A3B \
TRAIN_FILES=/shared/data/dapo_train_16k.parquet \
VAL_FILES=/shared/data/dapo_val_1k.parquet \
CHECKPOINT_DIR=/shared/checkpoints/qwen3-30b-grpo \
REWARD_FUNCTION_PATH=/path/to/reward.py \
bash run_qwen3-30b_grpo_node4_32k.sh \
    trainer.total_epochs=2 \
    trainer.save_freq=100
```

`reward.py` 需要提供 `compute_score` 函数。不需要自定义奖励时省略 `REWARD_FUNCTION_PATH`；不需要覆盖 Hydra 参数时，删除脚本名后的参数即可。也可在命令前增加 `JOB_ID=qwen3_30b_musa_test_001` 指定 Ray Job ID，未设置时脚本会自动生成。

脚本会在文件存在时加载本机 `set_env.sh`，渲染 `runtime_env.yaml`，从 hostfile 第一行取得 Ray head IP，再通过 `ray job submit` 提交 `verl.trainer.main_ppo`。Ray Job 和 worker 会获得 `VERL_MUSA_PATCH_AT_STARTUP=1`，在导入 VERL/SGLang 时安装对应 patch。

脚本内的主要训练配置为：4 节点、每节点 8 卡，actor TP/PP/EP/ETP 为 `4/1/8/1`，rollout TP/DP/EP 为 `4/1/1`，最大 response 长度为 `32768`，每个 prompt 生成 8 个 rollout 样本。

### 3.4 其他示例

- `run_qwen3-8b_grpo_node2_32k.sh`
- `run_qwen3-8b_grpo_node2_32k_separate_async_musa.sh`
- `run_qwen3-30b_grpo_node4_32k.sh`
- `run_ds_v2-lite_chat_grpo_node2.sh`
- `run_qwen3_5_35b_megatron_ep8pp2cp4.sh`
- `run_qwen3_5_122b_megatron_h20.sh`

示例只是参考，可以根据使用习惯使用自己的训练组织方式。
