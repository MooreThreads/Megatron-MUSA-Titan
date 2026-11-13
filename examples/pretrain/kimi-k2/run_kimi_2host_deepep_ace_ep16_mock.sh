#!/bin/bash

# List two nodes in the hostfile, one per line, with node 0 first.
set -euo pipefail

CURRENT_TIME=$(date "+%Y-%m-%d_%H%M%S")

TP_SIZE=1
PP_SIZE=1
EP_SIZE=16
WORLD_SIZE=16
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=16
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * DP_SIZE))

WORK_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_HOME=$(cd "$WORK_HOME/../../.." && pwd)
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_ep${EP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
HOSTFILE=${1:-"$WORK_HOME/hostfile"}
SCRIPT_FILE="$WORK_HOME/kimi-k2/pretrain_kimi_2host_deepep_ace_ep16_mock.sh"
RDZV_ID=$CURRENT_TIME
TRAIN_ITERS=${TRAIN_ITERS:-3}
MASTER_PORT=${MASTER_PORT:-12356}

mapfile -t hostlist < <(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE")
if [[ ${#hostlist[@]} -ne 2 || ${hostlist[0]} == "${hostlist[1]}" ]]; then
    echo "Expected two distinct node addresses in $HOSTFILE" >&2
    exit 2
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$WORK_HOME"}
mkdir -p "$OUTPUT_DIR/output/$RDZV_ID"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
LOG_FILE="$OUTPUT_DIR/output/$RDZV_ID/$EXPNAME.log"
echo "Launcher logs: $LOG_FILE.*"

pids=()
for NODE_RANK in "${!hostlist[@]}"; do
    host=${hostlist[$NODE_RANK]}
    cmd=(
        bash -l "$SCRIPT_FILE" "$PATCH_HOME" "$EXPNAME"
        "$TP_SIZE" "$PP_SIZE" "$EP_SIZE" "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
        "$NODE_RANK" "${hostlist[0]}" "$MASTER_PORT" "$TRAIN_ITERS"
        "$OUTPUT_DIR" "$RDZV_ID" "${MEGATRON_PATH:-}"
    )
    printf -v cmd_ssh '%q ' "${cmd[@]}"
    ssh -n -o BatchMode=yes "$host" "$cmd_ssh" > "$LOG_FILE.$NODE_RANK.$host" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
