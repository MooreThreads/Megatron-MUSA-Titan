#!/bin/bash
set -eo pipefail

PATCH_HOME=$1
EXPNAME=$2
TP_SIZE=$3
PP_SIZE=$4
EP_SIZE=$5
MICRO_BATCH_SIZE=$6
GLOBAL_BATCH_SIZE=$7
NODE_RANK=$8
MASTER_ADDR=$9
MASTER_PORT=${10}
TRAIN_ITERS=${11}
OUTPUT_DIR=${12}
RDZV_ID=${13}
MEGATRON_CHECKOUT=${14}

export PATCH_HOME
export MEGATRON_PATH=${MEGATRON_CHECKOUT:-${MEGATRON_PATH:-"$PATCH_HOME/../Megatron-LM"}}
export PYTHONPATH="$PATCH_HOME:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"

# DeepEP selects a local HCA by PCI topology when no override is set.
export MCCL_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME:-bond0}

export PRETRAIN_MUSA_PATCH_AT_STARTUP=1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_KERNEL_TIMEOUT=${MUSA_KERNEL_TIMEOUT:-3200000}
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20480000}
export PYTORCH_MUSA_ALLOC_CONF=${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}
export TORCH_MCCL_AVOID_RECORD_STREAMS=1

export USE_MUSA_MOE=1
export USE_DEEPEP_ACE=1
export USE_CE_ALLTOALL=0

GPUS_PER_NODE=8
NUM_NODES=2

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --log_dir "$OUTPUT_DIR/output_log/$RDZV_ID/$EXPNAME/$NODE_RANK"
    --redirects 3
)

MODEL_ARGS=(
    --num-layers 2
    --hidden-size 7168
    --num-attention-heads 64
    --seq-length 4096
    --max-position-embeddings 4096
    --norm-epsilon 1e-6
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --vocab-size 163840
    --padded-vocab-size 163968
    --ffn-hidden-size 18432
    --position-embedding-type rope
    --no-position-embedding
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --rope-type yarn
)

TRAINING_ARGS=(
    --seed 42
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --init-method-std 0.006
    --use-mcore-models
    --no-bias-dropout-fusion
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
    --distributed-backend nccl
    --distributed-timeout-minutes 8
    --multi-latent-attention
    --qk-layernorm
    --enable-experimental
    --recompute-granularity selective
    --recompute-modules mla_up_proj layernorm moe
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

LEARNING_RATE_ARGS=(
    --lr 2.2e-4
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --min-lr 1.5e-6
    --initial-loss-scale 65536
    --min-loss-scale 1.0
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
)

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --eval-interval "$TRAIN_ITERS"
    --eval-iters 0
)

MOE_LAYER_FREQ="[0,1]"
MOE_ARGS=(
    --num-experts 384
    --expert-model-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-grouped-gemm
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

cmd=(
    torchrun "${DISTRIBUTED_ARGS[@]}" "$MEGATRON_PATH/pretrain_gpt.py"
    "${MODEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
)
printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
