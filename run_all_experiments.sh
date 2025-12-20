#!/bin/bash
# ======================================================================
# 批量运行所有数据集实验
# ======================================================================

set -e

echo "======================================================================"
echo "RA-KG-PPO 批量实验脚本"
echo "======================================================================"
echo ""

# 配置
DATASETS=("amazon-book" "last-fm" "yelp2018")
EPOCHS=30
BATCH_SIZE=256
N_STEPS=2048

# 创建结果目录
mkdir -p results
RESULT_DIR="results/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "实验配置:"
echo "  - 数据集: ${DATASETS[@]}"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - N Steps: $N_STEPS"
echo "  - 结果目录: $RESULT_DIR"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行每个数据集
for dataset in "${DATASETS[@]}"; do
    echo "======================================================================"
    echo "[$dataset] 开始实验"
    echo "======================================================================"
    echo ""

    DATASET_START=$(date +%s)

    # 1. 检查数据是否存在
    echo "[$dataset] [1/3] 检查数据..."
    if [ ! -f "data/$dataset/train.txt" ]; then
        echo "[$dataset] [ERROR] 数据文件不存在: data/$dataset/train.txt"
        echo "[$dataset] [SKIP] 跳过此数据集"
        echo ""
        continue
    fi

    # 2. 预处理数据
    echo "[$dataset] [2/3] 预处理数据..."
    python scripts/prepare_data.py --dataset "$dataset" --force_rebuild

    # 3. 训练模型
    echo "[$dataset] [3/3] 训练模型..."
    python train.py \
        --dataset "$dataset" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --n_steps "$N_STEPS" \
        --eval_interval 5

    # 4. 保存结果
    echo "[$dataset] 保存结果..."

    # 复制checkpoint
    if [ -d "checkpoints" ]; then
        cp -r checkpoints "$RESULT_DIR/${dataset}_checkpoints"
    fi

    # 复制日志
    if [ -d "log" ]; then
        cp -r log "$RESULT_DIR/${dataset}_logs"
    fi

    # 计算用时
    DATASET_END=$(date +%s)
    DATASET_TIME=$((DATASET_END - DATASET_START))
    DATASET_HOURS=$((DATASET_TIME / 3600))
    DATASET_MINS=$(((DATASET_TIME % 3600) / 60))

    echo "[$dataset] [OK] 完成! 用时: ${DATASET_HOURS}h ${DATASET_MINS}m"
    echo ""
done

# 计算总用时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINS=$(((TOTAL_TIME % 3600) / 60))

echo "======================================================================"
echo "所有实验完成!"
echo "======================================================================"
echo ""
echo "总用时: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "结果保存在: $RESULT_DIR"
echo ""
echo "下载结果到本地:"
echo "  scp -r user@server:~/$RESULT_DIR ./local_results"
echo ""
