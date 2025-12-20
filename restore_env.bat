@echo off
REM 快速恢复运行环境

echo ======================================================================
echo 恢复RA-KG-PPO运行环境
echo ======================================================================
echo.

echo [1/3] 创建虚拟环境...
echo ----------------------------------------------------------------------
if exist .venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv .venv
    echo [OK] 虚拟环境创建完成
)

echo.
echo [2/3] 安装依赖包...
echo ----------------------------------------------------------------------
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo [OK] 依赖包安装完成

echo.
echo [3/3] 检查数据文件...
echo ----------------------------------------------------------------------

REM 检查原始数据是否需要重新下载
set NEED_DOWNLOAD=0
if not exist "data\amazon-book\train.txt.original" (
    if not exist "data\amazon-book\train.txt" (
        set NEED_DOWNLOAD=1
    )
)

if %NEED_DOWNLOAD%==1 (
    echo [WARNING] 原始数据文件不存在
    echo 请从以下链接下载数据集：
    echo https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data/amazon-book
    echo.
    echo 下载后，将文件放置在 data\amazon-book\ 目录
    echo 然后运行: python scripts\prepare_data.py --dataset amazon-book
    echo.
) else (
    REM 检查嵌入文件
    if not exist "data\amazon-book\item_embeddings.npy" (
        echo [INFO] 嵌入文件不存在，开始生成...
        python scripts\prepare_data.py --dataset amazon-book --force_rebuild
        echo [OK] 嵌入文件生成完成
    ) else (
        echo [OK] 数据文件完整
    )
)

echo.
echo ======================================================================
echo 环境恢复完成！
echo ======================================================================
echo.

echo 现在可以运行：
echo   1. 快速测试: python test_training.py
echo   2. 完整训练: python train.py --dataset amazon-book
echo.

pause
