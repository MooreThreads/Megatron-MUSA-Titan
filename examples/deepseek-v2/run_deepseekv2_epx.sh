#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJ_DIR="$(dirname "$SCRIPT_DIR")"
DURATION_BETWEEN_TESTS=0.3
MASTER_ADDR="10.116.36.208"
HOST_ADDR="10.116.36.208"
# CCP ADDR and PORT
EPX_CCP_ADDR="$MASTER_ADDR"
export EPX_CCP_ADDR
EPX_CCP_PORT="9009"
export EPX_CCP_PORT
# STORE ADDR and PORT
EPX_STORE_ADDR="$MASTER_ADDR"
export EPX_STORE_ADDR
EPX_STORE_PORT="45678"
export EPX_STORE_PORT
# EPX_LCP_ADDR and PORT
export EPX_LCP_ADDR="$HOST_ADDR"
# get epx session
# EPX_SESSION="$(uuidgen)"
EPX_SESSION="a0515990-ffbf-11ef-8a45-6fde377b8f7a"
export EPX_SESSION
# Enable EPX in Megatron-LM
export USE_GLOO_BACKEND=0
export USE_EPX=1
export USE_MCCL_BACKEND=1 # epx environment
export EPX_GROUP_RANK=8
export GPUS_PER_NODE=8
# MCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# 10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:FATAL
export MEGATRON_LOGGING_LEVEL=30

export EPX_PATH=/home/dist/epx
EPX_STORE_PATH="${EPX_PATH}/epx-py/examples/epx_store.py"
export PYTHONPATH=${EPX_PATH}/epx-py/python:$PYTHONPATH
export EPX_LCP_BIN="$EPX_PATH/target/debug/epx-lcp"
export DATA_PATH="/home/dist/llama2_dataset/llama_00_text_document"
export TOKENIZED_MODEL="/home/dist/llama2_dataset/tokenizer.model"

if [ "$MASTER_ADDR" = "$HOST_ADDR" ]; then
    python $EPX_STORE_PATH --addr "$HOST_ADDR" &
    STORE_PID=$!
fi

"$SCRIPT_DIR"/run_deepseekv2.sh &
R0_PID=$!

wait $R0_PID

if [ "$MASTER_ADDR" = "$HOST_ADDR" ]; then
    kill -9 $STORE_PID
    pkill -f $EPX_STORE_PATH
    pkill -f $EPX_STORE_PATH
fi
