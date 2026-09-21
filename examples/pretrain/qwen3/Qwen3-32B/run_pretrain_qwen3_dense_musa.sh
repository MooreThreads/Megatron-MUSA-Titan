#!/bin/bash

set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=$5
  TP_SIZE=$6
  PP_SIZE=$7
  MICRO_BATCH_SIZE=$8
  GLOBAL_BATCH_SIZE=${9}
  TOKENIZED_MODEL=${10}
  RDZV_ID=${11}
  MASTER_PORT=${12}
set +u
# export ENABLE_PROFILER=1
# export PROFILER_FREQ=4
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

# export USE_MUSA_MOE=1

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
# export MUSA_LAUNCH_BLOCKING=1

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
# export MUSA_LAUNCH_BLOCKING=1

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
    --num-layers 64
    --hidden-size 5120 
    --num-attention-heads 64
    --group-query-attention 
    --num-query-groups 8
    --seq-length 4096 
    --max-position-embeddings 40960
    --norm-epsilon 1e-6
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --vocab-size 151936
    --ffn-hidden-size 25600 
    --position-embedding-type rope 
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --no-rope-fusion
    --qk-layernorm 
    --make-vocab-size-divisible-by 1187
    --rotary-percent 1.0
    --rotary-base 1000000
    --kv-channels 128
)

# 244140625 1T
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
    --recompute-method uniform 
    --recompute-num-layers 1
    --distributed-backend nccl 
    --tp-only-amax-red
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

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZED_MODEL} \
    --split 1
"

# DATA_ARGS=(
#     --data-path $DATA_PATH 
#     --vocab-file $VOCAB_FILE 
#     --merge-file $MERGE_FILE 
#     --split 949,50,1
# )



EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    #--save-interval 200000 
    --eval-interval 1
    #--save $CHECKPOINT_PATH 
    #--load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
)

# bf16
TRANSFORMER_ENGINE_ARGS=(
   #--fp8-format e4m3
   #--fp8-recipe mxfp8
   --transformer-impl transformer_engine
   # --fp8-format hybrid
   #--fp8-param-gather
)

# if [ -n "${WANDB_API_KEY}" ]; then
#     EVAL_AND_LOGGING_ARGS+=(
#         --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
#         --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
#     )
# fi

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_qwen3.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
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
