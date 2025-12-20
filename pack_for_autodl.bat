@echo off
REM ==========================================
REM Windows 快速打包脚本
REM 准备上传到 AutoDL
REM ==========================================

echo ==========================================
echo 准备打包项目...
echo ==========================================
echo.

REM 设置项目目录
set PROJECT_DIR=%~dp0
set PROJECT_NAME=ra_kg_ppo
set OUTPUT_FILE=%PROJECT_NAME%_for_autodl.zip

echo 项目目录: %PROJECT_DIR%
echo 输出文件: %OUTPUT_FILE%
echo.

REM 检查是否有 PowerShell
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到 PowerShell
    echo 请手动压缩项目文件夹
    pause
    exit /b 1
)

echo 正在压缩文件...
echo 排除: data/, checkpoints/, logs/, __pycache__/
echo.

REM 使用 PowerShell 压缩，排除大文件
powershell -Command "& { ^
    $source = '%PROJECT_DIR%'; ^
    $destination = '%PROJECT_DIR%%OUTPUT_FILE%'; ^
    $exclude = @('data', 'checkpoints', 'checkpoints_5090', 'logs', 'tensorboard_logs', '__pycache__', '.git', '.ipynb_checkpoints'); ^
    Get-ChildItem -Path $source -Recurse ^| ^
    Where-Object { ^
        $path = $_.FullName; ^
        -not ($exclude ^| Where-Object { $path -like \"*\$_\*\" }) ^
    } ^| ^
    Compress-Archive -DestinationPath $destination -Force ^
}"

if exist "%OUTPUT_FILE%" (
    echo.
    echo ==========================================
    echo 打包完成！
    echo ==========================================
    echo.
    echo 文件位置: %OUTPUT_FILE%
    echo.
    echo 下一步:
    echo 1. 登录 AutoDL 控制台
    echo 2. 启动/创建 5090 实例
    echo 3. 点击 "JupyterLab" 按钮
    echo 4. 上传 %OUTPUT_FILE%
    echo 5. 在 Terminal 中执行:
    echo    cd /root/autodl-tmp
    echo    unzip %OUTPUT_FILE%
    echo    cd %PROJECT_NAME%
    echo    bash autodl_setup_5090.sh
    echo.
    echo 详细说明: AUTODL_UPLOAD_GUIDE.md
    echo ==========================================
) else (
    echo.
    echo 错误: 打包失败
    echo 请手动压缩项目文件夹（右键 -^> 发送到 -^> 压缩文件）
)

echo.
pause
