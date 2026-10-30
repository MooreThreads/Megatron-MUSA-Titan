#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"
if [[ -f "${SCRIPT_DIR}/set_env.sh" ]]; then
    source "${SCRIPT_DIR}/set_env.sh"
fi
: "${MUSA_TITAN_PATH:?Set MUSA_TITAN_PATH or create examples/rl/verl/set_env.sh}"
: "${VERL_PATH:?Set VERL_PATH or create examples/rl/verl/set_env.sh}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH or create examples/rl/verl/set_env.sh}"
: "${MEGATRON_BRIDGE_PATH:?Set MEGATRON_BRIDGE_PATH or create examples/rl/verl/set_env.sh}"
: "${RUNTIME_PYTHONPATH:?Set RUNTIME_PYTHONPATH or create examples/rl/verl/set_env.sh}"
: "${RUNTIME_LD_LIBRARY_PATH:?Set RUNTIME_LD_LIBRARY_PATH or create examples/rl/verl/set_env.sh}"
export VERL_MUSA_ATTENTION_DEBUG="${VERL_MUSA_ATTENTION_DEBUG:-1}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_logs/}"
mkdir -p "${TENSORBOARD_DIR}"

: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to a shared model directory}"
: "${TRAIN_FILES:?Set TRAIN_FILES to a shared training dataset path}"
: "${VAL_FILES:?Set VAL_FILES to a shared validation dataset path}"


MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"

for f in "${RUNTIME_ENV}" "${HOSTFILE}" "${TRAIN_FILES}" "${HF_MODEL_PATH}/config.json"; do
    if [[ ! -f "${f}" ]]; then
        echo "required file not found: ${f}" >&2
        exit 1
    fi
done
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
JOB_ID="${JOB_ID:-dsv2_lite_chat_grpo_node2_$(date +%Y%m%d_%H%M%S)}"

DATA=(
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE:-64}"
  data.shuffle=True
  data.seed=42
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.prompt_key=prompt
  data.truncation=error
)

ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=False
  algorithm.use_kl_in_reward=False
)

REWARD=(
  reward.reward_manager.name=naive
)
if [[ -n "${REWARD_FUNCTION_PATH:-}" ]]; then
  if [[ ! -f "${REWARD_FUNCTION_PATH}" ]]; then
    echo "reward function file not found: ${REWARD_FUNCTION_PATH}" >&2
    exit 1
  fi
  REWARD+=(
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}"
    reward.custom_reward_function.name=compute_score
  )
fi

MODEL=(
  actor_rollout_ref.model.path="${HF_MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_activation_offload=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.optimizer=adam
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.sequence_parallel=False
  actor_rollout_ref.actor.megatron.use_remove_padding=True
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
  actor_rollout_ref.actor.megatron.param_offload=True
  ++actor_rollout_ref.actor.megatron.override_transformer_config.num_attention_heads=16
  ++actor_rollout_ref.actor.megatron.override_transformer_config.num_query_groups=16
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused
  ++actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False
  ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=27
  ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
   # actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=0.999
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=512
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_dense_tp_size=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_lm_head=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.allow_auto_truncate=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=1024
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=16384
)

REF=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=False
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.sequence_parallel=False
  actor_rollout_ref.ref.megatron.use_remove_padding=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
  actor_rollout_ref.ref.megatron.param_offload=True
  ++actor_rollout_ref.ref.megatron.override_transformer_config.num_attention_heads=16
  ++actor_rollout_ref.ref.megatron.override_transformer_config.num_query_groups=16
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=fused
  ++actor_rollout_ref.ref.megatron.override_transformer_config.masked_softmax_fusion=False
  ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_rope_fusion=True
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.balance_batch=True
  trainer.logger='["console","tensorboard"]'
  trainer.project_name=musa-dsv2-lite
  trainer.experiment_name=dsv2-lite-chat-megatron-bridge-sglang-grpo
  trainer.n_gpus_per_node=8
  trainer.nnodes="${NNODES:-1}"
  trainer.val_before_train=False
  trainer.save_freq=100
  trainer.test_freq=100
  trainer.total_epochs=1
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.default_local_dir="${CHECKPOINT_DIR:-/tmp/verl_dsv2_lite_chat}"
  trainer.resume_mode=disable
  model_engine=megatron
  trainer.device=musa
)

ARGS=(
  "${DATA[@]}"
  "${ALGORITHM[@]}"
  "${REWARD[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${REF[@]}"
  "${TRAINER[@]}"
)

cd "${VERL_PATH}"
RAY_JOB_SUBMIT_ARGS=()
if [[ "${RAY_JOB_WAIT:-1}" == "0" ]]; then
  RAY_JOB_SUBMIT_ARGS+=(--no-wait)
fi
RAY_ADDRESS="http://${HEAD_IP}:8265" ray job submit "${RAY_JOB_SUBMIT_ARGS[@]}" --submission-id "${JOB_ID}" \
  --runtime-env="${RENDERED_RUNTIME_ENV}" -- python3 -u -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
