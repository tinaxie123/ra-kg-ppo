@echo off
REM 一键启动RA-KG-PPO完整训练（amazon-book数据集）- Windows版本

echo ======================================================================
echo RA-KG-PPO Training on Amazon-Book Dataset
echo ======================================================================
echo.

REM 检查数据是否准备好
echo Step 1: Checking data...
if not exist "data\amazon-book\train.txt" (
    echo [ERROR] Data not found!
    echo Please run: python scripts\prepare_data.py --dataset amazon-book
    exit /b 1
)
echo [OK] Data found

REM 检查嵌入是否存在
echo.
echo Step 2: Checking embeddings...
if not exist "data\amazon-book\item_embeddings.npy" (
    echo [WARNING] Embeddings not found, will be created during training
) else (
    echo [OK] Embeddings found
)

REM 开始训练
echo.
echo Step 3: Starting training...
echo ======================================================================
echo.

REM 选择配置
echo Select training mode:
echo 1 - Quick Training (10K timesteps, ~2-3 minutes)
echo 2 - Standard Training (100K timesteps, ~20-30 minutes)
echo 3 - Full Training (500K timesteps with GPU, ~2-3 hours)
echo.
set /p mode="Enter your choice (1/2/3): "

if "%mode%"=="1" (
    echo Running Quick Training...
    python train.py ^
        --dataset amazon-book ^
        --total-timesteps 10000 ^
        --n-steps 512 ^
        --batch-size 64 ^
        --n-epochs 4 ^
        --log-interval 1
) else if "%mode%"=="2" (
    echo Running Standard Training...
    python train.py ^
        --dataset amazon-book ^
        --total-timesteps 100000 ^
        --n-steps 2048 ^
        --batch-size 128 ^
        --n-epochs 10 ^
        --eval-freq 10
) else if "%mode%"=="3" (
    echo Running Full Training with GPU...
    python train.py ^
        --dataset amazon-book ^
        --device cuda ^
        --total-timesteps 500000 ^
        --n-steps 4096 ^
        --batch-size 256 ^
        --n-epochs 20 ^
        --hidden-dim 256
) else (
    echo [ERROR] Invalid selection
    exit /b 1
)

echo.
echo ======================================================================
echo Training completed!
echo ======================================================================
pause
