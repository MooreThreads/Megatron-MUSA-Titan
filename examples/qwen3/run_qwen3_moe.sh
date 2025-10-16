#!/bin/bash

CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
EP_SIZE=${EP_SIZE:-8}
WORLD_SIZE=${WORLD_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-32}
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))

echo -e "\033[32mDP_SIZE: $DP_SIZE, PP_SIZE: $PP_SIZE, EP_SIZE: $EP_SIZE, \
WORLD_SIZE: $WORLD_SIZE, MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE, \
NUM_MICROBATCHES: $NUM_MICROBATCHES, GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE\033[0m"

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/dataset/llama2_dataset/llama_00_text_document"}
  HOSTFILE=./hostfile
  LOG_FILE=$WORK_HOME/output/$CURRENT_TIME/$EXPNAME.log
  TOKENIZED_MODEL=${TOKENIZED_MODEL:-"${TOKENIZER_PATH}"} # ?
  SCRIPT_FILE=./Qwen3-30B-A3B/run_pretrain_qwen3_musa.sh
  RDZV_ID=$CURRENT_TIME
  MASTER_PORT=${MASTER_PORT:-12345}
set +u



COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

hostarray=($hostlist)
MASTER_ADDR=${hostarray[0]}
export MASTER_ADDR

export GLOO_SOCKET_IFNAME=$(ip route get 8.8.8.8 | grep -oP 'dev \K\S+' 2>/dev/null || echo "eth0")

# Check if hostlist is empty
if [ -z "$hostlist" ]; then
  echo "Error: hostlist is empty. Please add IP addresses to the hostfile."
  exit 1
fi

RUN_LOCAL=${RUN_LOCAL:-0}

COUNT=0
for host in ${hostlist[@]}; do
  echo -e "Main log file: \033[34m$LOG_FILE.$COUNT.$host\033[0m"
  echo -e "Distributed log_dir: \033[34m$WORK_HOME/output_log/$RDZV_ID/$EXPNAME\033[0m"

  cmd="bash -c 'cd $WORK_HOME; \
     export MASTER_ADDR=$MASTER_ADDR; \
     export MASTER_PORT=$MASTER_PORT; \
     export WORLD_SIZE=$WORLD_SIZE; \
     export RANK=$COUNT; \
     export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME; \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $EP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID $MASTER_PORT"

  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  # cmd_ssh=$cmd" '"

  echo $cmd_ssh

  if [[ "$RUN_LOCAL" -ne 0 ]]; then
    eval $cmd_ssh
  else
    ssh -f -n $host $cmd_ssh
  fi

  # echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done
