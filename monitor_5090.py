#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5090 GPU 资源监控脚本

实时监控：
- GPU利用率
- 显存使用
- 温度和功耗
- 训练吞吐量
"""

import torch
import time
import subprocess
import sys
from datetime import datetime


def get_gpu_info():
    """获取GPU详细信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'utilization': float(info[1]),
                'memory_used': float(info[2]),
                'memory_total': float(info[3]),
                'temperature': float(info[4]),
                'power_draw': float(info[5]),
                'power_limit': float(info[6])
            }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return None


def format_memory(mb):
    """格式化显存大小"""
    if mb < 1024:
        return f"{mb:.0f}MB"
    else:
        return f"{mb/1024:.2f}GB"


def get_pytorch_memory():
    """获取PyTorch显存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        return allocated, reserved
    return 0, 0


def print_bar(value, max_value, width=40, label="", unit=""):
    """打印进度条"""
    percentage = min(100, (value / max_value * 100))
    filled = int(width * value / max_value)
    bar = '█' * filled + '░' * (width - filled)

    # 根据使用率选择颜色（终端颜色代码）
    if percentage < 50:
        color = '\033[92m'  # 绿色
    elif percentage < 80:
        color = '\033[93m'  # 黄色
    else:
        color = '\033[91m'  # 红色
    reset = '\033[0m'

    print(f"{label:20s} {color}{bar}{reset} {percentage:5.1f}% ({value:.1f}{unit}/{max_value:.1f}{unit})")


def monitor_loop(interval=2):
    """监控循环"""
    print("\n" + "="*80)
    print(f"{'5090 GPU Monitor':^80}")
    print("="*80)
    print("Press Ctrl+C to stop\n")

    try:
        iteration = 0
        while True:
            iteration += 1

            # 清屏（可选）
            if iteration > 1:
                # 向上移动光标
                sys.stdout.write('\033[16A')  # 移动16行
                sys.stdout.flush()

            # 当前时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {now}")
            print("-" * 80)

            # GPU信息
            gpu_info = get_gpu_info()
            if gpu_info:
                print(f"\n{'GPU Model':<20s}: {gpu_info['name']}")
                print()

                # GPU利用率
                print_bar(
                    gpu_info['utilization'], 100,
                    label="GPU Utilization",
                    unit="%"
                )

                # 显存使用
                print_bar(
                    gpu_info['memory_used'], gpu_info['memory_total'],
                    label="VRAM Usage",
                    unit="MB"
                )

                # PyTorch显存
                allocated, reserved = get_pytorch_memory()
                if allocated > 0:
                    print_bar(
                        allocated, gpu_info['memory_total'],
                        label="PyTorch Allocated",
                        unit="MB"
                    )
                    print_bar(
                        reserved, gpu_info['memory_total'],
                        label="PyTorch Reserved",
                        unit="MB"
                    )

                # 温度
                print_bar(
                    gpu_info['temperature'], 100,
                    label="Temperature",
                    unit="°C"
                )

                # 功耗
                print_bar(
                    gpu_info['power_draw'], gpu_info['power_limit'],
                    label="Power Draw",
                    unit="W"
                )

                print()
                print(f"Memory: {format_memory(gpu_info['memory_used'])} / {format_memory(gpu_info['memory_total'])}")
                print(f"Temp: {gpu_info['temperature']:.0f}°C | Power: {gpu_info['power_draw']:.1f}W / {gpu_info['power_limit']:.1f}W")

            print("\n" + "-" * 80)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def print_summary():
    """打印GPU摘要信息"""
    print("\n" + "="*80)
    print(f"{'5090 GPU Summary':^80}")
    print("="*80 + "\n")

    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"GPU Model       : {gpu_info['name']}")
        print(f"Total VRAM      : {format_memory(gpu_info['memory_total'])}")
        print(f"Used VRAM       : {format_memory(gpu_info['memory_used'])} ({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
        print(f"Free VRAM       : {format_memory(gpu_info['memory_total'] - gpu_info['memory_used'])}")
        print(f"GPU Utilization : {gpu_info['utilization']:.1f}%")
        print(f"Temperature     : {gpu_info['temperature']:.0f}°C")
        print(f"Power Draw      : {gpu_info['power_draw']:.1f}W / {gpu_info['power_limit']:.1f}W")

        # PyTorch信息
        if torch.cuda.is_available():
            allocated, reserved = get_pytorch_memory()
            if allocated > 0:
                print(f"\nPyTorch Allocated : {format_memory(allocated)}")
                print(f"PyTorch Reserved  : {format_memory(reserved)}")

    # CUDA信息
    if torch.cuda.is_available():
        print(f"\nCUDA Version    : {torch.version.cuda}")
        print(f"PyTorch Version : {torch.__version__}")
        print(f"cuDNN Version   : {torch.backends.cudnn.version()}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='5090 GPU Monitor')
    parser.add_argument('--mode', type=str, default='monitor',
                       choices=['monitor', 'summary'],
                       help='Monitor mode: monitor (real-time) or summary (one-time)')
    parser.add_argument('--interval', type=int, default=2,
                       help='Update interval in seconds (for monitor mode)')

    args = parser.parse_args()

    if args.mode == 'monitor':
        monitor_loop(interval=args.interval)
    else:
        print_summary()
