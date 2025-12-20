#!/bin/bash
##############################################
# 本地简化实验 - 一键运行脚本
##############################################

echo "=========================================="
echo "RA-KG-PPO Local Simplified Training"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    exit 1
fi

echo "[1/4] Checking environment..."
python -c "import torch; import numpy; import pandas; print('✓ All packages installed')" || {
    echo "Error: Missing packages. Please run:"
    echo "  pip install torch numpy pandas scipy scikit-learn tqdm"
    exit 1
}

# 检查数据
echo ""
echo "[2/4] Checking data..."
if [ ! -f "data/amazon-book/train.txt" ]; then
    echo "Data not found. Preparing data..."
    python scripts/prepare_data.py --dataset amazon-book || {
        echo "Error: Data preparation failed"
        exit 1
    }
else
    echo "✓ Data ready"
fi

# 选择模式
echo ""
echo "[3/4] Select training mode:"
echo "  1) Quick test (2 min, 1K timesteps)"
echo "  2) Basic training (15 min, 10K timesteps) [Recommended]"
echo "  3) Extended training (1 hour, 50K timesteps)"
echo "  4) Custom"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Starting quick test..."
        python train_local_simplified.py --total-timesteps 1024 --eval-freq 1
        ;;
    2)
        echo "Starting basic training..."
        python train_local_simplified.py --total-timesteps 10000 --eval-freq 5
        ;;
    3)
        echo "Starting extended training..."
        python train_local_simplified.py --total-timesteps 50000 --eval-freq 5
        ;;
    4)
        read -p "Enter total timesteps: " timesteps
        echo "Starting custom training..."
        python train_local_simplified.py --total-timesteps $timesteps
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# 显示结果
echo ""
echo "[4/4] Training completed!"
echo "=========================================="
echo "Results saved to:"
echo "  - checkpoints_local/training_results.json"
echo "  - checkpoints_local/final_model.pt"
echo ""
echo "View results:"
echo "  cat checkpoints_local/training_results.json"
echo ""
echo "Next steps:"
echo "  1. Check EXPERIMENTAL_RESULTS.md for paper draft"
echo "  2. Read LOCAL_TRAINING_GUIDE.md for details"
echo "=========================================="
