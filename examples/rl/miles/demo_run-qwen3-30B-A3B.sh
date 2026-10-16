#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -f "${SCRIPT_DIR}/set_env.sh" ]]; then
   # shellcheck disable=SC1091
   source "${SCRIPT_DIR}/set_env.sh"
fi

: "${MILES_PATH:?Set MILES_PATH or create examples/rl/miles/set_env.sh}"
: "${MUSA_TITAN_PATH:?Set MUSA_TITAN_PATH or create examples/rl/miles/set_env.sh}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH or create examples/rl/miles/set_env.sh}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT or create examples/rl/miles/set_env.sh}"
: "${REF_LOAD:?Set REF_LOAD or create examples/rl/miles/set_env.sh}"
: "${SAVE_PATH:?Set SAVE_PATH or create examples/rl/miles/set_env.sh}"
: "${DATASET_PATH:?Set DATASET_PATH or create examples/rl/miles/set_env.sh}"
: "${MODEL_ARGS_SCRIPT:?Set MODEL_ARGS_SCRIPT or create examples/rl/miles/set_env.sh}"
: "${RUNTIME_PYTHONPATH:?Set RUNTIME_PYTHONPATH or create examples/rl/miles/set_env.sh}"
: "${RUNTIME_LD_LIBRARY_PATH:?Set RUNTIME_LD_LIBRARY_PATH or create examples/rl/miles/set_env.sh}"

export PYTHONUNBUFFERED=1

MODEL_ARGS_LINE="$(python3 "$MODEL_ARGS_SCRIPT" qwen3-30B-A3B)" || exit 1
read -r -a MODEL_ARGS <<< "$MODEL_ARGS_LINE"

MAX_CONCURRENT_ROLLOUTS=32

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$REF_LOAD"
   --load "$REF_LOAD"
   --save "$SAVE_PATH"
   --save-interval 500
   #--megatron-to-hf-mode bridge
)

train_files=$DATASET_PATH/amt_math17k_d_qweninstruct.parquet

ROLLOUT_ARGS=(
   --prompt-data $train_files
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --rm-type zero2one
   --start-rollout-id 0
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 1024
   --rollout-max-response-len 128
   #--global-batch-size 32
   --num-steps-per-rollout 1
   --balance-data

   --rollout-temperature 1
   --rollout-top-p 1 # Omitting this uses torch.multinomial, which has precision issues.
   #--rollout-top-k 100
)

test_files=$DATASET_PATH/aime_2025.parquet
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data $test_files
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 4096
   --eval-temperature 0.6
   --eval-top-p 0.95
   --log-passrate
   --seed 0
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 48

   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 5120 # $((8192 * 4 + 1024))
   --log-probs-chunk-size 4096
   --calculate-per-token-loss
   --use-tis

   --moe-token-dispatcher-type alltoall
   --moe-router-dtype fp32

)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --observe-training-entropy
   --eps-clip 0.0
   --eps-clip-high 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
)

SGLANG_ARGS=(
   --sglang-device musa
   --sglang-mem-fraction-static 0.7
   --rollout-num-gpus-per-engine 4
   # --sglang-tp-size 8
   --sglang-ep-size 4
   # --sglang-dp-size 1
   --sglang-dp-size 4
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head

   --sglang-attention-backend fa3
   #--sglang-moe-runner-backend deep_gemm
   --sglang-cuda-graph-max-bs $MAX_CONCURRENT_ROLLOUTS
   # MUSA graph capture for eight batch sizes is prohibitively slow during startup.
   # --sglang-disable-cuda-graph
   # --sglang-pp-size 8
   #--partial-rollout
   # --disable-overlap-schedule
   # --sglang-skip-tokenizer-init
   --no-sglang-record-nolora-graph
   --sglang-disable-overlap-schedule
   # --sglang-disable-custom-all-reduce
   --sglang-disable-piecewise-cuda-graph

   --sglang-server-concurrency $MAX_CONCURRENT_ROLLOUTS
   --sglang-max-running-requests $MAX_CONCURRENT_ROLLOUTS
   --skip-eval-before-train # Skip evaluation at step 0.
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

export no_proxy="127.0.0.1"

runtime_env_template="${SCRIPT_DIR}/runtime_env.yaml"
runtime_env="$(mktemp "${TMPDIR:-/tmp}/miles-runtime-env.XXXXXX.yaml")"
trap 'rm -f "$runtime_env"' EXIT
python3 - "$runtime_env_template" "$runtime_env" <<'PY'
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
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
if [[ -n "${RAY_HEAD_IP:-}" ]]; then
   HEAD_IP="$RAY_HEAD_IP"
elif [[ -r "$HOSTFILE" ]]; then
   HEAD_IP="$(awk 'NF {print $1; exit}' "$HOSTFILE")"
else
   echo "Set RAY_HEAD_IP or provide $HOSTFILE" >&2
   exit 1
fi
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8972}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/miles_demo_qwen3-30B-A3B.log"
JOB_ID="miles_demo_qwen3-30B-A3B_$(date +%Y%m%d-%H%M%S)"

RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" ray job submit \
   --runtime-env="$runtime_env" \
   --submission-id "$JOB_ID" \
   --no-wait \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

# The Ray CLI may output ANSI color codes; strip them before writing a plain log file
# to avoid characters such as ^[[36m.
RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" ray job logs "$JOB_ID" -f 2>&1 \
   | sed -u -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g' > "$LOG_FILE"
