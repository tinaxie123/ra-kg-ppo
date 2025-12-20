#!/bin/bash
###########################################
# AutoDL 5090 完整部署脚本
# 一键上传和启动
###########################################

echo "=========================================="
echo "Preparing files for AutoDL..."
echo "=========================================="

# 要上传到AutoDL的文件列表
FILES=(
    "train_5090_optimized.py"
    "autodl_setup_5090.sh"
    "start_training_5090.sh"
    "monitor_5090.py"
    "requirements_5090.txt"
    "5090_OPTIMIZATION_GUIDE.md"
)

echo "Files to upload:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing!)"
    fi
done

echo ""
echo "=========================================="
echo "Upload Instructions:"
echo "=========================================="
echo ""
echo "1. 在 AutoDL 创建实例："
echo "   - 镜像: PyTorch 2.1+ (CUDA 12.1)"
echo "   - GPU: RTX 5090"
echo "   - 存储: 50GB+"
echo ""
echo "2. 启动实例后，上传整个项目到："
echo "   /root/autodl-tmp/ra_kg_ppo/"
echo ""
echo "3. 在 AutoDL 终端执行："
echo "   cd /root/autodl-tmp/ra_kg_ppo"
echo "   bash autodl_setup_5090.sh"
echo ""
echo "4. 开始训练："
echo "   bash start_training_5090.sh quick    # 快速测试"
echo "   bash start_training_5090.sh medium   # 标准训练"
echo "   bash start_training_5090.sh full     # 完整训练"
echo ""
echo "5. 监控GPU（另开终端）："
echo "   python monitor_5090.py --mode monitor"
echo ""
echo "=========================================="
echo ""
echo "详细文档: 5090_OPTIMIZATION_GUIDE.md"
echo "=========================================="
