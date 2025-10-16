#!/bin/bash

set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=$5
  TP_SIZE=$6
  PP_SIZE=$7
  EP_SIZE=$8
  MICRO_BATCH_SIZE=$9
  GLOBAL_BATCH_SIZE=${10}
  TOKENIZED_MODEL=${11}
  RDZV_ID=${12}
  MASTER_PORT=${13}
set +u
# export ENABLE_PROFILER=1
# export PROFILER_FREQ=6
# export PROFILER_WARMUP_STEPS=3
# export PROFILER_ACTIVE_STEPS=3
# export PROFILER_REPEAT_NUM=1
# export MUSA_LAUNCH_BLOCKING=1
# export PROFILER_PROFILE_MEMORY=1

export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-'0,1,2,3,4,5,6,7'}
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-160}
# export MOE_ROUTER_GROUP_TOPK=${MOE_ROUTER_GROUP_TOPK:-3}

# export MUSA_EXECUTION_TIMEOUT=20000000 # Recommended for use with zero-bubble
export ENABLE_ZERO_BUBBLE=0 # if set 1, Enable zero_bubble

# VMM
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"
export TORCH_MCCL_AVOID_RECORD_STREAMS=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export MUSA_BLOCK_ARBITRATION_MODE=2
export CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG=${CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG:-0}

# export USE_RECOMPUTE_VARIANCE=1
export ENABLE_D2H_IN_PERMUTATION=0
export NO_LOSS_REDUCE=0
export USE_MUSA_MOE=1

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
DATA_PATH=$DATA_DIR

DATA_CACHE_PATH=$WORK_HOME/data_cache/$EXPNAME
mkdir -p $DATA_CACHE_PATH


LOG_PATH=$WORK_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$WORK_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH


export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1) # tail for cuda/ head for musa
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
export MASTER_PORT=${MASTER_PORT:-12356}

echo "Distributed log_dir: $WORK_HOME/output_log/$RDZV_ID/$EXPNAME"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --log_dir $WORK_HOME/output_log/$RDZV_ID/$EXPNAME
    --redirects ${LOG_REDIRECTS_LEVEL:-3}
)

MODEL_ARGS=(
    --num-layers 48  # 48
    --hidden-size 2048
    --num-attention-heads 128
    --seq-length 4096 
    --max-position-embeddings 40960 
    --norm-epsilon 1e-6
    --attention-dropout 0.0
    --hidden-dropout 0.0 
    --disable-bias-linear
    --vocab-size 151936
    --make-vocab-size-divisible-by 1187
    --ffn-hidden-size 6144 
    --position-embedding-type rope
    --no-position-embedding 
    --rotary-base 1000000
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --no-rope-fusion
)

# 24414062 1T
TRAINING_ARGS=(
    --seed 42
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    #--train-samples 24414062 
    --lr-warmup-iters 200
    --train-iters 300
    --init-method-std 0.02 
    --use-mcore-models 
    # --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 0
    --distributed-backend nccl
    --tp-only-amax-red
)

ATTENTION_ARGS=(
    --group-query-attention
    --num-query-groups 4 
    --num-attention-heads 32 
    --kv-channels 128 
    --qk-layernorm
    --rotary-scaling-factor 1 
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

LEARNING_RATE_ARGS=(
    --lr 1.5e-5
    --lr-decay-style cosine
    # --lr-warmup-samples ${WARMUP_SAMPLES}
    --min-lr 1.5e-6
    --initial-loss-scale 65536
    --min-loss-scale 1.0
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE
	--pipeline-model-parallel-size $PP_SIZE
)

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZED_MODEL}
    --split 1
    #--dataloader-type mtepx  #default single
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    #--save-interval 100000
    --eval-interval 1
    #--save $CHECKPOINT_PATH
    #--load $CHECKPOINT_PATH
    --eval-iters 0
    --tensorboard-dir $TB_PATH
)

NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
MOE_LAYER_FREQ="([1]*${NUM_LAYERS})*1"

MOE_ARGS=(
    --num-experts ${MOE_NUM_EXPERTS}
    --expert-model-parallel-size $EP_SIZE
    --moe-token-dispatcher-type alltoall
    --moe-router-score-function softmax
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3 
    --moe-ffn-hidden-size 768 
    --moe-layer-freq "$MOE_LAYER_FREQ"
    # --moe-grouped-gemm
    # --moe-permute-fusion
    # --moe-layer-recompute
)

# bf16
TRANSFORMER_ENGINE_ARGS=(
   #--fp8-format e4m3
   #--fp8-recipe mxfp8
   --transformer-impl transformer_engine
   # --fp8-format hybrid
   #--fp8-param-gather
)


cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_qwen3.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${ATTENTION_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]}
    "

USE_EPX=${USE_EPX:-0}

# run cmd directly
if [ $USE_EPX -eq 0 ]; then
  echo $cmd
  $cmd
  exit $?
fi

# run cmd with fault tolerance #?
source "${PATCH_HOME}/examples/deepseek-v2/deepseek-v2-lite/fault_tolerance_function.sh"
ft_training "$cmd"
