@echo off
REM ======================================================================
REM Windows - 打包项目用于云服务器部署
REM ======================================================================

echo ======================================================================
echo 打包项目代码用于云服务器部署
echo ======================================================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 运行打包脚本
echo 开始打包...
echo.
python package_for_cloud.py

if errorlevel 1 (
    echo.
    echo [ERROR] 打包失败
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo 打包完成！
echo ======================================================================
echo.
echo 下一步：
echo   1. 将 ra_kg_ppo_deploy.tar.gz 上传到云服务器
echo   2. 准备数据文件（train.txt, test.txt）
echo   3. 在服务器上解压并运行 deploy_cloud.sh
echo.
echo 详细步骤请参考: DEPLOYMENT_GUIDE.md
echo.

pause
