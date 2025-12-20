#!/bin/bash
# ======================================================================
# 云服务器一键部署脚本
# 适用于：AutoDL / 阿里云 / 腾讯云 Ubuntu 20.04/22.04 with GPU
# ======================================================================

set -e  # Exit on error

echo "======================================================================"
echo "RA-KG-PPO 云服务器部署脚本"
echo "======================================================================"
echo ""

# 1. 检查环境
echo "[1/6] 检查系统环境..."
echo "----------------------------------------------------------------------"
python3 --version
nvidia-smi || echo "[WARNING] GPU not detected, will use CPU"
echo ""

# 2. 创建工作目录
echo "[2/6] 创建工作目录..."
echo "----------------------------------------------------------------------"
cd ~
if [ -d "ra_kg_ppo" ]; then
    echo "[INFO] 目录已存在，备份旧版本..."
    mv ra_kg_ppo ra_kg_ppo_backup_$(date +%Y%m%d_%H%M%S)
fi
mkdir -p ra_kg_ppo
cd ra_kg_ppo
echo "[OK] 工作目录: $(pwd)"
echo ""

# 3. 安装依赖
echo "[3/6] 安装Python依赖..."
echo "----------------------------------------------------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install numpy pandas tqdm gym scikit-learn scipy -q
echo "[OK] 依赖安装完成"
echo ""

# 4. 下载数据（如果需要）
echo "[4/6] 准备数据集..."
echo "----------------------------------------------------------------------"
mkdir -p data/amazon-book

if [ ! -f "data/amazon-book/train.txt" ]; then
    echo "[INFO] 请确保已上传数据文件到 data/amazon-book/"
    echo "      需要的文件: train.txt, test.txt"
    echo ""
    read -p "数据文件已准备好? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[ERROR] 请上传数据文件后重新运行"
        exit 1
    fi
fi
echo "[OK] 数据文件检查完成"
echo ""

# 5. 预处理数据
echo "[5/6] 预处理数据并生成嵌入..."
echo "----------------------------------------------------------------------"
python scripts/prepare_data.py --dataset amazon-book --force_rebuild
echo "[OK] 数据预处理完成"
echo ""

# 6. 开始训练
echo "[6/6] 开始训练..."
echo "----------------------------------------------------------------------"
echo ""
echo "推荐的训练配置："
echo "  - 快速测试: python train.py --dataset amazon-book --epochs 10"
echo "  - 完整训练: python train.py --dataset amazon-book --epochs 50"
echo ""
read -p "选择模式 [1=快速测试, 2=完整训练]: " mode

if [ "$mode" == "1" ]; then
    echo "[INFO] 启动快速测试模式 (10 epochs)..."
    python train.py --dataset amazon-book --epochs 10 --batch_size 256
elif [ "$mode" == "2" ]; then
    echo "[INFO] 启动完整训练模式 (50 epochs)..."
    python train.py --dataset amazon-book --epochs 50 --batch_size 256
else
    echo "[INFO] 使用默认配置启动..."
    python train.py --dataset amazon-book
fi

echo ""
echo "======================================================================"
echo "训练完成！"
echo "======================================================================"
echo ""
echo "结果保存在："
echo "  - 模型: checkpoints/"
echo "  - 日志: log/"
echo ""
echo "下载结果文件："
echo "  scp -r user@server:~/ra_kg_ppo/checkpoints ./results"
echo "  scp -r user@server:~/ra_kg_ppo/log ./results"
echo ""
