#!/bin/bash
###########################################
# AutoDL 5090 一键启动训练
#
# 使用方法：
# bash start_training_5090.sh [mode]
#
# mode 选项：
#   quick   - 快速测试（10分钟，默认）
#   medium  - 中等训练（1-2小时）
#   full    - 完整训练（4-8小时）
#   ultra   - 超长训练（24小时+）
###########################################

MODE=${1:-quick}

echo "=========================================="
echo "AutoDL 5090 Training Launcher"
echo "Mode: $MODE"
echo "=========================================="

# 检查GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 根据模式选择参数
case $MODE in
    quick)
        echo -e "\n[Quick Mode] Fast test training"
        python train_5090_optimized.py \
            --dataset amazon-book \
            --total-timesteps 50000 \
            --batch-size 256 \
            --n-steps 2048 \
            --eval-freq 5 \
            --save-freq 10 \
            --use-amp
        ;;

    medium)
        echo -e "\n[Medium Mode] Standard training"
        python train_5090_optimized.py \
            --dataset amazon-book \
            --total-timesteps 500000 \
            --batch-size 512 \
            --n-steps 4096 \
            --n-epochs 15 \
            --eval-freq 5 \
            --save-freq 10 \
            --use-amp
        ;;

    full)
        echo -e "\n[Full Mode] Complete training"
        python train_5090_optimized.py \
            --dataset amazon-book \
            --total-timesteps 1000000 \
            --batch-size 1024 \
            --n-steps 4096 \
            --n-epochs 20 \
            --hidden-dim 256 \
            --kg-emb-dim 256 \
            --eval-freq 5 \
            --save-freq 10 \
            --use-amp
        ;;

    ultra)
        echo -e "\n[Ultra Mode] Extended training with larger model"
        python train_5090_optimized.py \
            --dataset amazon-book \
            --total-timesteps 2000000 \
            --batch-size 1024 \
            --n-steps 8192 \
            --n-epochs 25 \
            --hidden-dim 512 \
            --kg-emb-dim 512 \
            --item-emb-dim 256 \
            --num-layers 4 \
            --candidate-size 300 \
            --eval-freq 3 \
            --save-freq 5 \
            --use-amp
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: quick, medium, full, ultra"
        exit 1
        ;;
esac

echo -e "\n=========================================="
echo "Training completed!"
echo "Check results in: ./checkpoints_5090/"
echo "=========================================="
