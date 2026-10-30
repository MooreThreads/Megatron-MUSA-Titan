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
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_logs/verl_grpo_qwen3_5_3m5b_geo3k/qwen3_5_35b_megatron}"
mkdir -p "${TENSORBOARD_DIR}"
########################### Quick Config ###########################

# ---- user-adjustable ----
# This launcher is intentionally MUSA-only.
TP=${TP:-1}
PP=${PP:-2}
CP=${CP:-4}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-4}
GEN_EP=${GEN_EP:-1}
DISABLE_CUSTOM_AR=${DISABLE_CUSTOM_AR:-False}
n_devices_per_node=${NDEVICES_PER_NODE:-8}
nnodes=${NNODES:-2}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
OVERLAP_CPU_OPTIMIZER=${OVERLAP_CPU_OPTIMIZER:-False}
# Keep MCore's backward normalization tied to the global number of routed
# tokens.  The legacy loss path reports a similar scalar pg_loss, but scales
# the first backward gradient by roughly 1e3 for this packed MoE workload.
CALCULATE_PER_TOKEN_LOSS=${CALCULATE_PER_TOKEN_LOSS:-True}
GEN_EP=${GEN_EP:-1}

# Default tuned for the current 40-layer Qwen3.5-35B / PP=2 layout.
DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS:-18}

rollout_name="sglang"
project_name='verl_grpo_qwen3_5_35b_geo3k'
exp_name='qwen3_5_35b_megatron'
adv_estimator=grpo

: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to a shared model directory}"
# Data lives on a shared filesystem visible to every Ray worker. Do not use a
# relative path here: Ray Jobs changes the worker cwd to its uploaded package.
TEXT_ONLY=${TEXT_ONLY:-0}
: "${TRAIN_PATH_IMAGE:?Set TRAIN_PATH_IMAGE to a shared training dataset path}"
: "${TEST_PATH_IMAGE:?Set TEST_PATH_IMAGE to a shared validation dataset path}"

TRAIN_PATH_TEXT_ONLY=${TRAIN_PATH_TEXT_ONLY:-}
TEST_PATH_TEXT_ONLY=${TEST_PATH_TEXT_ONLY:-}


USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

if [[ "${TEXT_ONLY}" == "1" ]]; then
    if [[ -z "${TRAIN_PATH_TEXT_ONLY}" || -z "${TEST_PATH_TEXT_ONLY}" ]]; then
        echo "TEXT_ONLY=1 requires TRAIN_PATH_TEXT_ONLY and TEST_PATH_TEXT_ONLY (absolute paths visible on every Ray node)." >&2
        exit 1
    fi
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
    data.train_batch_size=32
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.truncation='error'
    data.filter_overlong_prompts=True
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
    actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_REMOVE_PADDING
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=$USE_REMOVE_PADDING
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.use_torch_compile=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.cp_comm_type=p2p
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=40
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.000
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=${OVERLAP_CPU_OPTIMIZER}
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0

)

if (( PP > 1 )); then
    ACTOR+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.decoder_last_pipeline_num_layers=${DECODER_LAST_PIPELINE_NUM_LAYERS}
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    # VERL rollout worker topology: one 4-GPU TP engine; SGLang DP topology is set below.
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_REMOVE_PADDING
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=0.8
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=1.0
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=${DISABLE_CUSTOM_AR}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=8192
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=128
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device="musa"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=128
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
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=NEXTN
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_steps=3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_eagle_topk=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=4
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_draft_weights_cpu_backup=True
    #+actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=393216

    +actor_rollout_ref.rollout.server.timeout=1800
    +actor_rollout_ref.rollout.server.max_attempts=3
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_REMOVE_PADDING
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

# Pass the requested loss-normalization mode explicitly. The current Qwen3-VL
# Megatron-Bridge implementation requires this to be True when CP>1, so use
# CP=1 when CALCULATE_PER_TOKEN_LOSS=False.
#
# Keep gradient reduction in fp32 and disable fused reduce/divide while
# validating the non-per-token MUSA path. This also matches the safer DDP
# behavior used by VERL's legacy mbridge construction path.
ACTOR+=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=${CALCULATE_PER_TOKEN_LOSS}
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=False
    +actor_rollout_ref.actor.megatron.override_ddp_config.average_in_collective=False
    +actor_rollout_ref.actor.megatron.override_ddp_config.gradient_reduce_div_fusion=False
)

# In per-token mode MCore performs the final normalization with the accumulated
# token count, so DDP must not average gradients inside the collective.
if [[ "${CALCULATE_PER_TOKEN_LOSS,,}" == "true" ]]; then
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
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${nnodes}
    trainer.save_freq=200
    trainer.val_before_train=False
    trainer.test_freq=50
    trainer.total_epochs=15
)

EXTRA=(
    model_engine=megatron
)

ACTOR+=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
)

########################### Launch ###########################

if [[ "${DRY_RUN:-0}" == "cfg" ]]; then
    cd "${VERL_PATH}"
    exec python3 -m verl.trainer.main_ppo --cfg job --resolve \
        "${DATA[@]}" "${ALGORITHM[@]}" "${MODEL[@]}" "${ROLLOUT[@]}" \
        "${ACTOR[@]}" "${REF[@]}" "${TRAINER[@]}" "${EXTRA[@]}" "$@"
fi

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

JOB_ID="${JOB_ID:-qwen3_5_35b_musa_$(date +%Y%m%d_%H%M%S)}"
cd "${VERL_PATH}"

PYTHON_ENTRY=(python3 -u -m verl.trainer.main_ppo)


RAY_ADDRESS="http://${HEAD_IP}:8265" ray job submit \
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
