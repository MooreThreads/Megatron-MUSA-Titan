#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_qwen3-8b_grpo_node2_32k.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "base launcher not found: ${BASE_SCRIPT}" >&2
    exit 1
fi

CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-mooncake}"
TRAIN_NNODES="${TRAIN_NNODES:-1}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
JOB_ID="${JOB_ID:-qwen3_8b_separate_async_musa_$(date +%Y%m%d_%H%M%S)}"
export JOB_ID


PROFILER=()

source "${BASE_SCRIPT}" \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    actor_rollout_ref.hybrid_engine=False \
    trainer.nnodes="${TRAIN_NNODES}" \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CHECKPOINT_BACKEND}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake.device=musa \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    trainer.v1.separate_async.parameter_sync_step=4 \
    trainer.v1.separate_async.num_warmup_batches=1 \
    "$@"
