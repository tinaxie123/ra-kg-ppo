#!/bin/bash
###########################################
# AutoDL 5090 环境配置脚本
#
# 使用方法：
# bash autodl_setup_5090.sh
###########################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "AutoDL 5090 Environment Setup"
echo "=========================================="

# 1. 检查CUDA和GPU
echo -e "\n[1/7] Checking GPU..."
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 2. 切换到项目目录
echo -e "\n[2/7] Setting up project directory..."
cd /root/autodl-tmp

# 如果项目不存在，提示用户上传
if [ ! -d "ra_kg_ppo" ]; then
    echo "Project directory not found!"
    echo "Please upload your project to /root/autodl-tmp/"
    echo "Or clone it: git clone <your-repo-url>"
    exit 1
fi

cd ra_kg_ppo
echo "Working directory: $(pwd)"

# 3. 升级pip并安装依赖（使用清华镜像加速）
echo -e "\n[3/7] Installing dependencies..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检查PyTorch版本是否支持CUDA 12
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

echo "Current PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"

# 如果需要，重新安装PyTorch以支持最新CUDA
if [[ "$CUDA_VERSION" < "12.1" ]]; then
    echo "Upgrading PyTorch for better 5090 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 安装项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 准备数据
echo -e "\n[4/7] Preparing data..."
if [ ! -d "data/amazon-book" ]; then
    echo "Downloading and preparing amazon-book dataset..."
    python scripts/prepare_data.py --dataset amazon-book
else
    echo "Dataset already exists, skipping download."
fi

# 5. 创建必要的目录
echo -e "\n[5/7] Creating directories..."
mkdir -p checkpoints_5090
mkdir -p logs
mkdir -p tensorboard_logs

# 6. 测试环境
echo -e "\n[6/7] Testing setup..."
python test_setup.py

# 7. 配置完成
echo -e "\n[7/7] Setup completed!"
echo "=========================================="
echo "Ready to train on 5090!"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Quick test (1-2 minutes):"
echo "   python test_training.py"
echo ""
echo "2. Optimized training for 5090:"
echo "   python train_5090_optimized.py"
echo ""
echo "3. Custom training:"
echo "   python train_5090_optimized.py --batch-size 1024 --total-timesteps 2000000"
echo ""
echo "4. Monitor GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5. Monitor training (in another terminal):"
echo "   tensorboard --logdir tensorboard_logs --host 0.0.0.0 --port 6006"
echo ""
echo "=========================================="
