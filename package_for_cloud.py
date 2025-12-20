#!/usr/bin/env python3
"""
打包项目代码，用于上传到云服务器
排除数据文件和不必要的文件，只打包核心代码
"""

import os
import tarfile
import shutil
from pathlib import Path
from datetime import datetime

def create_deployment_package():
    """创建部署包"""

    print("=" * 70)
    print("打包RA-KG-PPO项目用于云服务器部署")
    print("=" * 70)
    print()

    # 项目根目录
    project_root = Path(__file__).parent

    # 要包含的文件和目录
    include_patterns = [
        # 核心代码
        'algorithms/',
        'data/',
        'docs/',
        'envs/',
        'models/',
        'retrieval/',
        'scripts/',
        'utils/',

        # 主脚本
        'train.py',
        'test_training.py',

        # 配置文件
        'requirements.txt',
        '.gitignore',

        # 文档
        'README.md',
        'COMPLETE_IMPLEMENTATION.md',
        'DEPLOYMENT_GUIDE.md',

        # 部署脚本
        'deploy_cloud.sh',
        'run_training.sh',
    ]

    # 要排除的文件模式
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        '.venv',
        'venv',
        '*.npy',
        '*.pkl',
        '*.pth',
        '*.ckpt',
        'checkpoint_*.pth',
        'log/',
        'logs/',
        'results/',
        'figures/',
        '.vscode/',
        '.idea/',
        '.claude/',
        '.DS_Store',
        'Thumbs.db',
        '*.swp',
        '*.swo',
        '*.tmp',
        '*.bak',
        # 数据文件（太大，不上传）
        'data/*/train.txt',
        'data/*/test.txt',
        'data/*/kg_final.txt',
        'data/*/*.npy',
        'data/*/*.pkl',
    ]

    # 输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'ra_kg_ppo_deploy.tar.gz'

    print(f"[1/4] 准备打包文件...")
    print(f"      输出文件: {output_filename}")
    print()

    # 创建临时目录
    temp_dir = project_root / 'temp_deploy'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    deploy_dir = temp_dir / 'ra_kg_ppo'
    deploy_dir.mkdir()

    print(f"[2/4] 复制文件...")

    file_count = 0
    total_size = 0

    def should_exclude(path_str):
        """检查是否应该排除"""
        for pattern in exclude_patterns:
            if pattern in path_str or path_str.endswith(pattern.replace('*', '')):
                return True
        return False

    # 复制文件
    for pattern in include_patterns:
        source_path = project_root / pattern

        if not source_path.exists():
            print(f"      [SKIP] {pattern} (不存在)")
            continue

        if source_path.is_file():
            # 单个文件
            if not should_exclude(str(source_path)):
                dest_path = deploy_dir / pattern
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                file_count += 1
                total_size += source_path.stat().st_size
                print(f"      [OK] {pattern}")

        elif source_path.is_dir():
            # 目录
            for root, dirs, files in os.walk(source_path):
                # 过滤目录
                dirs[:] = [d for d in dirs if not should_exclude(d)]

                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(project_root)

                    if should_exclude(str(rel_path)):
                        continue

                    dest_path = deploy_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    file_count += 1
                    total_size += file_path.stat().st_size

            print(f"      [OK] {pattern}")

    print()
    print(f"[3/4] 创建压缩包...")

    # 创建tar.gz
    output_path = project_root / output_filename
    if output_path.exists():
        output_path.unlink()

    with tarfile.open(output_path, 'w:gz') as tar:
        tar.add(deploy_dir, arcname='ra_kg_ppo')

    # 清理临时目录
    shutil.rmtree(temp_dir)

    package_size = output_path.stat().st_size

    print(f"      压缩包大小: {package_size / (1024*1024):.2f} MB")
    print()

    print(f"[4/4] 完成！")
    print()
    print("=" * 70)
    print("打包完成！")
    print("=" * 70)
    print()
    print(f"文件统计:")
    print(f"  - 文件数量: {file_count}")
    print(f"  - 原始大小: {total_size / (1024*1024):.2f} MB")
    print(f"  - 压缩后: {package_size / (1024*1024):.2f} MB")
    print(f"  - 压缩率: {(1 - package_size/total_size)*100:.1f}%")
    print()
    print(f"输出文件: {output_filename}")
    print()
    print("=" * 70)
    print("上传到服务器:")
    print("=" * 70)
    print()
    print("方法1: 使用scp上传")
    print("----------------------------------------------------------------------")
    print(f"  scp -P 端口号 {output_filename} root@服务器地址:~")
    print()
    print("  示例 (AutoDL):")
    print(f"  scp -P 12345 {output_filename} root@region-1.autodl.com:~")
    print()
    print("方法2: 使用Web界面上传")
    print("----------------------------------------------------------------------")
    print("  1. 登录云服务器的JupyterLab/Web终端")
    print("  2. 直接拖拽上传文件")
    print()
    print("方法3: 使用Git")
    print("----------------------------------------------------------------------")
    print("  1. 先上传到GitHub")
    print("  2. 在服务器上: git clone https://github.com/你的用户名/ra_kg_ppo.git")
    print()
    print("=" * 70)
    print("在服务器上解压并运行:")
    print("=" * 70)
    print()
    print(f"  tar -xzf {output_filename}")
    print("  cd ra_kg_ppo")
    print("  bash deploy_cloud.sh")
    print()
    print("=" * 70)
    print("注意事项:")
    print("=" * 70)
    print()
    print("  [WARN] 数据文件未包含在压缩包中（文件太大）")
    print("         需要单独上传数据文件到服务器的 data/amazon-book/ 目录")
    print()
    print("         必需的数据文件:")
    print("           - train.txt")
    print("           - test.txt")
    print("           - kg_final.txt (可选，没有会自动生成)")
    print()
    print("  [WARN] 或者在服务器上运行 prepare_data.py 自动生成嵌入")
    print()

if __name__ == '__main__':
    try:
        create_deployment_package()
    except Exception as e:
        print(f"\n[ERROR] 打包失败: {e}")
        import traceback
        traceback.print_exc()
