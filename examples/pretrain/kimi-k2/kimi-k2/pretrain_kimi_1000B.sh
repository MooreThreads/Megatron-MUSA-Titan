#!/bin/bash
set -eo pipefail
set -x
# Kimi pretraining launcher.
# Set the environment variables below according to the cluster configuration.
# To initialize an external TransformerEngine installation, source the script
# referenced by TE_INSTALL_SCRIPT before invoking this launcher.

WORK_HOME="$PWD"
PATCH_HOME="$PWD"/../../..
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${WORK_HOME}/artifacts"}
MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
TP_SIZE=${TP:-1}
PP_SIZE=${PP:-31}
EP_SIZE=${EP:-8}
FP8=${FP8:-false}
MTP=${MTP:-0}
FORCE_LB=${FORCE_LB:-false}
EXIT_INTERVAL=${EXIT_INTERVAL:-20000000000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

NUM_NODES=${WORLD_SIZE:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TOKENIZED_MODEL=${TOKENIZED_MODEL:?Set TOKENIZED_MODEL to the tokenizer model path}
CURRENT_TIME=$(date "+%Y-%m-%d_%H%M")
RDZV_ID=$CURRENT_TIME
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_mtp${MTP}_forcelb${FORCE_LB}_pertensor${FP8}_NO_LOSS_REDUCE"

export ENABLE_PROFILER=${ENABLE_PROFILER:-1}
export PROFILER_FREQ=${PROFILER_FREQ:-400}
export PROFILER_SAVE_DIR=${PROFILER_SAVE_DIR:-"${ARTIFACT_ROOT}/profiler"}
export PROFILER_WITH_STACK=${PROFILER_WITH_STACK:-1}
# export MUSA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_EXECUTION_TIMEOUT=480000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=136
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_CROSS_NIC=0
export USE_RECOMPUTE_VARIANCE=0
export ENABLE_D2H_IN_PERMUTATION=0
export NO_LOSS_REDUCE=${NO_LOSS_REDUCE:-1}
# export USE_MUSA_MOE=1
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export MCCL_LIB=${MCCL_LIB:-"${MUSA_HOME}/lib/libmccl.so"}
export MUSA_ERROR_DUMP_PATH=${MUSA_ERROR_DUMP_PATH:-"${ARTIFACT_ROOT}/error_dumps/$(date "+%Y.%m.%d-%H:%M:%S")"}
MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
export USE_DEEPEP_ACE=1
export EP_BALANCE_INFO=0
export MUSA_LOG=0x1
export USE_TRITON_WEIGHTED_SWIGLU=1

export PP_USE_MOONCAKE=1
# MOONCAKE_P2P_SLOT_SIZE_BYTES can be hidden_size * seq_length * 2 (bf16 dtype of activation) if there's no MTP configuring.
export MOONCAKE_P2P_SLOT_SIZE_BYTES=58720256
export MC_BIND_NIC=${MC_BIND_NIC:-1}
export MC_WORKERS_PER_CTX=${MC_WORKERS_PER_CTX:-2}
export MC_SLICE_SIZE=${MC_SLICE_SIZE:-65536}

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${ARTIFACT_ROOT}/checkpoints/${EXPNAME}"}
mkdir -p "$CHECKPOINT_PATH"
# DATA_PATH=$DATA_DIR


LOG_PATH=$WORK_HOME/logs/$RDZV_ID
mkdir -p $LOG_PATH
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$WORK_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

# export DUMP_MEMORY_SNAPSHOT=0
# export MEMORY_SNAPSHOT_PATH=$WORK_HOME/mem_snapshot/$RDZV_ID
# mkdir -p $MEMORY_SNAPSHOT_PATH
# export RDZV_ID=$RDZV_ID

#VMM
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"
export TORCH_MCCL_AVOID_RECORD_STREAMS=1 

export NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
export GPUS_PER_NODE=8
export NODE_RANK=$RANK
# export MUSA_LAUNCH_BLOCKING=1
# export MCCL_DEBUG=INFO
DATASET_FILE=${DATASET_FILE:?Set DATASET_FILE to the dataset manifest path}
DATA_PATH="$(grep -v '^#' "${DATASET_FILE}")"
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${ARTIFACT_ROOT}/data_cache"}

TOTAL_TOKENS=${TOTAL_TOKENS:?Set TOTAL_TOKENS for the training dataset}
SEQ_LEN=4096
SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
TRAIN_SAMPLES=$SAMPLE_SIZE
TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
WARMUP_STEPS=500
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))
WSD_DECAY_SAMPLES=$((TRAIN_SAMPLES * 35 / 100)) # 0.35

OUTPUT_DIR=${OUTPUT_DIR:-"${ARTIFACT_ROOT}/outputs"}
RUN_DIR=${OUTPUT_DIR}/kimi-pretrain/${CURRENT_TIME}_${EXPNAME}/
mkdir -p "${RUN_DIR}"
LOG_FILE_TMP="${RUN_DIR}/${EXPNAME}.RANK${NODE_RANK}.${NODE_ADDR}.log"
LOG_FILE=${LOG_FILE:-${LOG_FILE_TMP}}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
#     --log_dir $LOG_FILE_TMP #$WORK_HOME/output_log/$RDZV_ID/$EXPNAME/$NODE_RANK
#     --redirects 3
)
    
LAYERS=${LAYERS:-61}
MODEL_ARGS=(
    --num-layers $LAYERS  # 61 
    --hidden-size 7168
    --num-attention-heads 64
    --seq-length 4096 
    --max-position-embeddings 4096 
    --norm-epsilon 1e-6 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --vocab-size 163840 #163840
    --ffn-hidden-size 18432
    --position-embedding-type rope
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --rope-type yarn
)

# 24414062 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-samples $TRAIN_SAMPLES #2441406200 
    --init-method-std  0.006 
    --use-mcore-models 
    # --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion
    # --no-rope-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    # --recompute-granularity full 
    # --recompute-method block 
    # --recompute-num-layers 1 
    --distributed-backend nccl
    --multi-latent-attention
    --qk-layernorm
    --enable-experimental
    # --mlp-recompute
    # --mlp-rms-recompute
    # --recompute-variance
    # --attn-recompute
    # --mla-rms-recompute
    --recompute-granularity selective
    --recompute-modules mla_up_proj moe_act layernorm mlp
    --offload-moe-fc1-input
    --offload-moe-fused-swiglu-input
    --manual-gc
    --manual-gc-interval 30
    --exit-interval ${EXIT_INTERVAL}
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    # --rotary-base 50000
    # --beta-fast 1.0
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)

LEARNING_RATE_ARGS=(
    --lr 2e-4  
    --lr-warmup-samples ${WARMUP_SAMPLES}
    --lr-decay-style WSD
    --lr-wsd-decay-style cosine
    --lr-wsd-decay-samples ${WSD_DECAY_SAMPLES}
    --min-lr 2e-5
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE  
	--pipeline-model-parallel-size $PP_SIZE 
    --account-for-loss-in-pipeline-split
    --num-virtual-stages-per-pipeline-rank 2
    --overlap-p2p-communication-warmup-flush
)

MIXED_PRECISION_ARGS=(
    --bf16 
    --attention-softmax-in-fp32 
    --no-masked-softmax-fusion 
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type NullTokenizer #NullTokenizer
    --tokenizer-model ${TOKENIZED_MODEL}
    --data-cache-path $DATA_CACHE_PATH
    --split 100,0,0
    --distributed-timeout-minutes 10
    --num-dataset-builder-threads 16
    --num-workers 2
    # --no-mmap-bin-files
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 300
    --eval-interval 1 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH
    --ckpt-format torch
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
    --no-load-optim
)

NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 1))
MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"
MOE_ARGS=(
    --num-experts 384
    --expert-model-parallel-size $EP_SIZE
    --moe-token-dispatcher-type alltoall
    # --moe-router-num-groups 8
    # --moe-router-group-topk 4
    --moe-router-topk 8
    --moe-router-score-function sigmoid #sigmoid
    --moe-router-pre-softmax
    # --moe-z-loss-coeff 0.0001
    --moe-router-topk-scaling-factor 2.827
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-grouped-gemm
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    # --moe-router-force-load-balancing
    # --overlap-moe-expert-parallel-comm
    --moe-permute-fusion
    # --moe-token-dispatcher-type alltoall
    --moe-token-dispatcher-type flex
    --moe-enable-deepep

    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
)

if [ $FORCE_LB = true ]; then
    MOE_ARGS+=(--moe-router-force-load-balancing)
fi



TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    # --fp8-format e4m3
    # --fp8-param-gather
    # --fp8-recipe mxfp8
)

if [ $FP8 = true ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --fp8-format e4m3
        --fp8-param-gather
        --fp8-recipe mxfp8
    )
fi

MULTI_TOKEN_PREDICTION_ARGS=(
    --use-multi-token-prediction
    --mtp-coeff 1e-4
    --mtp-depth ${MTP}
)

unset MLFLOW_TRACKING_URI
unset MCCL_IB_HCA

NUMA_BIND_CORES=${NUMA_BIND_CORES:-1}
TRAIN_ENTRYPOINT=("$WORK_HOME/pretrain_kimi.py")
if [[ "$NUMA_BIND_CORES" != "0" ]]; then
    # local rank 0--3 use NUMA 0 and 4--7 use NUMA 1; each rank gets eight
    # physical CPU cores plus their SMT siblings.
    DISTRIBUTED_ARGS+=(--no-python)
    TRAIN_ENTRYPOINT=("$WORK_HOME/kimi-k2/numa_bind_by_node.sh" python "$WORK_HOME/pretrain_kimi.py")
fi

torchrun "${DISTRIBUTED_ARGS[@]}" "${TRAIN_ENTRYPOINT[@]}" \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}
set +x
