#!/bin/bash
# 一键启动RA-KG-PPO完整训练（amazon-book数据集）

echo "======================================================================"
echo "RA-KG-PPO Training on Amazon-Book Dataset"
echo "======================================================================"
echo ""

# 检查数据是否准备好
echo "Step 1: Checking data..."
if [ ! -f "data/amazon-book/train.txt" ]; then
    echo "[ERROR] Data not found!"
    echo "Please run: python scripts/prepare_data.py --dataset amazon-book"
    exit 1
fi
echo "[OK] Data found"

# 检查嵌入是否存在
echo ""
echo "Step 2: Checking embeddings..."
if [ ! -f "data/amazon-book/item_embeddings.npy" ]; then
    echo "[WARNING] Embeddings not found, will be created during training"
else
    echo "[OK] Embeddings found"
fi

# 开始训练
echo ""
echo "Step 3: Starting training..."
echo "======================================================================"
echo ""

# 选择配置
read -p "Select training mode (1=Quick/2=Standard/3=Full): " mode

case $mode in
    1)
        echo "Running Quick Training (10K timesteps)..."
        python train.py \
            --dataset amazon-book \
            --total-timesteps 10000 \
            --n-steps 512 \
            --batch-size 64 \
            --n-epochs 4 \
            --log-interval 1
        ;;
    2)
        echo "Running Standard Training (100K timesteps)..."
        python train.py \
            --dataset amazon-book \
            --total-timesteps 100000 \
            --n-steps 2048 \
            --batch-size 128 \
            --n-epochs 10 \
            --eval-freq 10
        ;;
    3)
        echo "Running Full Training (500K timesteps with GPU)..."
        python train.py \
            --dataset amazon-book \
            --device cuda \
            --total-timesteps 500000 \
            --n-steps 4096 \
            --batch-size 256 \
            --n-epochs 20 \
            --hidden-dim 256
        ;;
    *)
        echo "[ERROR] Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
