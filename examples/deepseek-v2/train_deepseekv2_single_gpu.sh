#!/bin/bash
# bash train_deepseekv2_single_gpu.sh --dataset_dir /home/llama2_dataset --data_format fp8

# 默认参数
DATA_FORMAT="fp8"  # 默认使用 fp8
DATASET_DIR=""     # 必须参数

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_format)
            DATA_FORMAT="$2"
            shift 2
            ;;
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$DATASET_DIR" ]]; then
    echo "Error: --dataset_dir is required"
    exit 1
fi

# 验证数据格式
if [[ "$DATA_FORMAT" != "fp8" && "$DATA_FORMAT" != "bf16" ]]; then
    echo "Error: --data_format must be 'fp8' or 'bf16'"
    exit 1
fi

# 生成唯一实验ID
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
mkdir -p "./output/$CURRENT_TIME"

# 训练配置
TP_SIZE=1
PP_SIZE=1
EP_SIZE=1
WORLD_SIZE=1
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=32
(( DP_SIZE = WORLD_SIZE / (TP_SIZE * PP_SIZE) ))
(( GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * NUM_MICROBATCHES * DP_SIZE ))
export GPUS_PER_NODE=1
export MOE_NUM_EXPERTS=20
export MOE_ROUTER_GROUP_TOPK=1
export MUSA_VISIBLE_DEVICES='5'

# 生成hostfile
ip a | grep -oP 'inet \K[\d.]+' | grep -v '^127\.' | head -1 > hostfile

# 设置环境变量
WORK_HOME="$PWD"
PATCH_HOME="$PWD/../.."
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_${DATA_FORMAT}"
DATA_PATH="${DATASET_DIR}/llama_00_text_document"
HOSTFILE="./hostfile"
LOG_FILE="./output/$CURRENT_TIME/$EXPNAME.log"
TOKENIZED_MODEL="${DATASET_DIR}/tokenizer.model"
SCRIPT_FILE="./deepseek-v2-lite/run_pretrain_deepseekv2_musa.sh"
RDZV_ID="$CURRENT_TIME"

# 精度相关配置
if [[ "$DATA_FORMAT" == "bf16" ]]; then
    # 移除FP8参数
    sed -i '/--fp8-format hybrid/d; /--fp8-param-gather/d' "$SCRIPT_FILE"

    # 添加no-gradient-accumulation-fusion参数
    sed -i '/no-gradient-accumulation-fusion/c\    --no-gradient-accumulation-fusion' "$SCRIPT_FILE"

    echo "Enabled BF16 mode with recompute optimizations"
fi

# 运行训练
cmd="bash -c 'cd $WORK_HOME && \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $EP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID'"

echo "=== Training Configuration ==="
echo "Dataset dir: $DATASET_DIR"
echo "Data format: $DATA_FORMAT"
echo "Hostfile: $(cat hostfile)"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Command:"
echo "$cmd"
eval "$cmd"
