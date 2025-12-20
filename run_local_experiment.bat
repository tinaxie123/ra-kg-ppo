@echo off
REM ==========================================
REM 本地简化实验 - 一键运行脚本 (Windows)
REM ==========================================

echo ==========================================
echo RA-KG-PPO Local Simplified Training
echo ==========================================
echo.

REM 检查Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found!
    pause
    exit /b 1
)

echo [1/4] Checking environment...
python -c "import torch; import numpy; import pandas; print('All packages installed')" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Missing packages. Please run:
    echo   pip install torch numpy pandas scipy scikit-learn tqdm
    pause
    exit /b 1
)
echo Done: Environment OK

REM 检查数据
echo.
echo [2/4] Checking data...
if not exist "data\amazon-book\train.txt" (
    echo Data not found. Preparing data...
    python scripts\prepare_data.py --dataset amazon-book
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Data preparation failed
        pause
        exit /b 1
    )
) else (
    echo Done: Data ready
)

REM 选择模式
echo.
echo [3/4] Select training mode:
echo   1. Quick test (2 min, 1K timesteps^)
echo   2. Basic training (15 min, 10K timesteps^) [Recommended]
echo   3. Extended training (1 hour, 50K timesteps^)
echo   4. Custom
echo.
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" (
    echo Starting quick test...
    python train_local_simplified.py --total-timesteps 1024 --eval-freq 1
) else if "%choice%"=="2" (
    echo Starting basic training...
    python train_local_simplified.py --total-timesteps 10000 --eval-freq 5
) else if "%choice%"=="3" (
    echo Starting extended training...
    python train_local_simplified.py --total-timesteps 50000 --eval-freq 5
) else if "%choice%"=="4" (
    set /p timesteps="Enter total timesteps: "
    echo Starting custom training...
    python train_local_simplified.py --total-timesteps %timesteps%
) else (
    echo Invalid choice
    pause
    exit /b 1
)

REM 显示结果
echo.
echo [4/4] Training completed!
echo ==========================================
echo Results saved to:
echo   - checkpoints_local\training_results.json
echo   - checkpoints_local\final_model.pt
echo.
echo View results:
echo   type checkpoints_local\training_results.json
echo.
echo Next steps:
echo   1. Check EXPERIMENTAL_RESULTS.md for paper draft
echo   2. Read LOCAL_TRAINING_GUIDE.md for details
echo ==========================================
echo.
pause
