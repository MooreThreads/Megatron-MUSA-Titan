#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"

if [[ -f "${SCRIPT_DIR}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/set_env.sh"
fi
: "${MUSA_TITAN_PATH:?Set MUSA_TITAN_PATH or create examples/rl/verl/set_env.sh}"
: "${VERL_PATH:?Set VERL_PATH or create examples/rl/verl/set_env.sh}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH or create examples/rl/verl/set_env.sh}"
: "${MEGATRON_BRIDGE_PATH:?Set MEGATRON_BRIDGE_PATH or create examples/rl/verl/set_env.sh}"
: "${RUNTIME_PYTHONPATH:?Set RUNTIME_PYTHONPATH or create examples/rl/verl/set_env.sh}"
: "${RUNTIME_LD_LIBRARY_PATH:?Set RUNTIME_LD_LIBRARY_PATH or create examples/rl/verl/set_env.sh}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_logs/run_qwen3_5_122b_megatron_h20/qwen3_5_122b_megatron_sglang-geo}"
mkdir -p "${TENSORBOARD_DIR}"
########################### Quick Config ###########################

# ---- user-adjustable ----
# This launcher is intentionally MUSA-only.
TP=${TP:-1}
PP=${PP:-4}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}
GEN_EP=${GEN_EP:-1}
# SGLang speculative/MTP is intentionally disabled for this baseline.  Keep
# all speculative fields absent instead of passing speculative_algorithm=NONE.
n_devices_per_node=${NDEVICES_PER_NODE:-8}
nnodes=${NNODES:-4}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# H20-aligned batch layout. These defaults can be overridden from the shell.
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
PAD_BSHD_TO_MINIBATCH_MAX=${PAD_BSHD_TO_MINIBATCH_MAX:-False}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS:-10}
DECODER_FIRST_PIPELINE_NUM_LAYERS=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-10}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-False}
ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-1024}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-48}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-True}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1}

rollout_name="sglang"
project_name='verl_grpo_qwen3_5_122b_geo3k'
exp_name='qwen3_5_122b_megatron'
adv_estimator=grpo

: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to a shared model directory}"
# Data lives on a shared filesystem visible to every Ray worker. Do not use a
# relative path here: Ray Jobs changes the worker cwd to its uploaded package.
TEXT_ONLY=${TEXT_ONLY:-0}
: "${TRAIN_FILES:?Set TRAIN_FILES to a shared training dataset path}"
: "${VAL_FILES:?Set VAL_FILES to a shared validation dataset path}"
TRAIN_PATH_IMAGE="${TRAIN_FILES}"
TEST_PATH_IMAGE="${VAL_FILES}"

if [[ "${TEXT_ONLY}" == "1" ]]; then
    TRAIN_PATH=${TRAIN_PATH_TEXT_ONLY}
    TEST_PATH=${TEST_PATH_TEXT_ONLY}
else
    TRAIN_PATH=${TRAIN_PATH_IMAGE}
    TEST_PATH=${TEST_PATH_IMAGE}
fi

train_path=${train_path:-${TRAIN_PATH:-$HOME/data/geo3k/train.parquet}}
test_path=${test_path:-${TEST_PATH:-$HOME/data/geo3k/test.parquet}}
if [[ "${train_path}" != /* || "${test_path}" != /* ]]; then
    echo "TRAIN_PATH and TEST_PATH must be absolute paths visible on every Ray node" >&2
    exit 1
fi
if [[ "${TEXT_ONLY}" == "1" ]]; then
    if [[ ! -f "${train_path}" || ! -f "${test_path}" ]]; then
        echo "TEXT_ONLY=1 but text-only parquet is missing:" >&2
        echo "  ${train_path}" >&2
        echo "  ${test_path}" >&2
        exit 1
    fi
fi
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=128
    data.max_prompt_length=3240
    data.max_response_length=4096
    data.truncation=right
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
    data.shuffle=False
    data.seed=42
)

if [[ "${TEXT_ONLY}" == "1" ]]; then
    DATA+=(
        data.image_key=__no_image__
        data.video_key=__no_video__
        data.audio_key=__no_audio__
    )
fi

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=${ENTROPY_FROM_LOGITS_WITH_CHUNKING}
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=${ENTROPY_FROM_LOGITS_CHUNK_SIZE}
    actor_rollout_ref.actor.megatron.use_remove_padding=${USE_REMOVE_PADDING}
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=${PAD_BSHD_TO_MINIBATCH_MAX}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}
    +actor_rollout_ref.actor.megatron.override_transformer_config.decoder_first_pipeline_num_layers=${DECODER_FIRST_PIPELINE_NUM_LAYERS}
    +actor_rollout_ref.actor.megatron.override_transformer_config.decoder_last_pipeline_num_layers=${DECODER_LAST_PIPELINE_NUM_LAYERS}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.000
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD}
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0

)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    # VERL rollout worker topology: one 4-GPU TP engine; SGLang DP topology is set below.
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=6
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=1.0
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    #+actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy="extra_buffer"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_prefix_mm_cache=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=16384
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=192
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device="musa"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=192
    +actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_backend="flashinfer"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tokenizer_backend="fastokens"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend="flashinfer"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    # SGLang DP-attention topology: TP=4, attention DP=4, MoE EP=4.
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=True
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.dp_size=4
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_dense_tp_size=1
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_lm_head=True
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.ep_size=4
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=NEXTN
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_steps=3
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_eagle_topk=1
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=4
    #+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=True
    #+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_draft_weights_cpu_backup=True
    #+actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=393216

    +actor_rollout_ref.rollout.server.timeout=1800
    +actor_rollout_ref.rollout.server.max_attempts=3
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
)

REF=(
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=${ENTROPY_FROM_LOGITS_WITH_CHUNKING}
    actor_rollout_ref.ref.entropy_from_logits_chunk_size=${ENTROPY_FROM_LOGITS_CHUNK_SIZE}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

# Packed THD context parallelism requires per-token loss. Disable gradient
# averaging in the DDP collective so CP loss scaling is applied exactly once.
# Keep the CP=1 path unchanged.
if (( CP > 1 )); then
    ACTOR+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
        +actor_rollout_ref.actor.megatron.override_ddp_config.average_in_collective=False
    )
    REF+=(
        +actor_rollout_ref.ref.megatron.override_ddp_config.average_in_collective=False
    )
fi

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.device=musa
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${nnodes}
    trainer.save_freq=200
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.test_freq=10
    trainer.total_epochs=2
)

EXTRA=(
    model_engine=megatron
)

ACTOR+=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
)

########################### Launch ###########################

if [[ ! -f "${RUNTIME_ENV}" ]]; then
    echo "runtime env file not found: ${RUNTIME_ENV}" >&2
    exit 1
fi
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "hostfile not found: ${HOSTFILE}; run setup_ray.sh or set HOSTFILE" >&2
    exit 1
fi

RENDERED_RUNTIME_ENV="${TMPDIR:-/tmp}/verl_runtime_env_${$}.yaml"
trap 'rm -f "${RENDERED_RUNTIME_ENV}"' EXIT
python3 - "${RUNTIME_ENV}" "${RENDERED_RUNTIME_ENV}" <<'PY'
import os
import re
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
names = sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", template)))
missing = [name for name in names if name not in os.environ]
if missing:
    raise SystemExit("Missing runtime environment variables: " + ", ".join(missing))
Path(sys.argv[2]).write_text(os.path.expandvars(template))
PY

HEAD_IP="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")"
if [[ -z "${HEAD_IP}" ]]; then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

JOB_ID="${JOB_ID:-qwen3_5_122b_h20_musa_$(date +%Y%m%d_%H%M%S)}"
cd "${VERL_PATH}"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
PYTHON_ENTRY=(python3 -u -m verl.trainer.main_ppo)
RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" ray job submit \
    --submission-id "${JOB_ID}" \
    --runtime-env="${RENDERED_RUNTIME_ENV}" \
    -- "${PYTHON_ENTRY[@]}" \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
