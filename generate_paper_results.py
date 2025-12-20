#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成论文实验结果的辅助脚本
用于收集和格式化实验数据
"""

import json
import numpy as np
from typing import Dict, List
import pandas as pd


# ============================================
# 实验结果模板（待填充）
# ============================================

RESULTS_TEMPLATE = {
    "amazon-book": {
        "RA-KG-PPO": {
            "Recall@10": None,
            "Recall@20": None,
            "Recall@50": None,
            "NDCG@10": None,
            "NDCG@20": None,
            "NDCG@50": None,
            "Hit@10": None,
            "Hit@20": None,
            "Hit@50": None,
            "Precision@10": None,
            "Precision@20": None,
            "Precision@50": None,
        },
        "KGAT": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "KGIN": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "SASRec": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "BERT4Rec": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "GRU4Rec": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "TPGR": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "UNICORN": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "BPR": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
        "NCF": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    },
    "last-fm": {
        # 同样的结构
    },
    "yelp2018": {
        # 同样的结构
    }
}

ABLATION_TEMPLATE = {
    "RA-KG-PPO (Full)": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    "w/o Knowledge Graph": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    "w/o Retrieval (LSH)": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    "w/o Policy Conditioning": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    "w/o PPO (use REINFORCE)": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
    "w/o GAE": {"Recall@20": None, "NDCG@20": None, "Hit@20": None, "Precision@20": None},
}


# ============================================
# LaTeX表格生成函数
# ============================================

def format_result(value, best=False, second_best=False):
    """格式化结果数值"""
    if value is None:
        return "0.XXX"

    formatted = f"{value:.3f}"

    if best:
        return f"\\textbf{{{formatted}}}"
    elif second_best:
        return f"\\underline{{{formatted}}}"
    else:
        return formatted


def generate_main_results_table(results: Dict, dataset: str = "amazon-book"):
    """生成主实验结果LaTeX表格"""

    methods = [
        "BPR", "NCF",
        "GRU4Rec", "SASRec", "BERT4Rec",
        "CKE", "KGAT", "KGIN",
        "DRR", "TPGR", "UNICORN",
        "RA-KG-PPO"
    ]

    metrics = ["Recall@20", "NDCG@20", "Hit@20", "Precision@20"]

    print(f"\n{'='*80}")
    print(f"Main Results Table - {dataset}")
    print(f"{'='*80}\n")

    # 找到最佳和次优
    for metric in metrics:
        values = []
        for method in methods:
            if method in results[dataset] and metric in results[dataset][method]:
                val = results[dataset][method][metric]
                if val is not None:
                    values.append(val)

        if values:
            best_val = max(values)
            sorted_vals = sorted(values, reverse=True)
            second_best_val = sorted_vals[1] if len(sorted_vals) > 1 else None

            print(f"{metric}:")
            for method in methods:
                if method in results[dataset] and metric in results[dataset][method]:
                    val = results[dataset][method][metric]
                    is_best = (val == best_val) if val is not None else False
                    is_second = (val == second_best_val) if val is not None and second_best_val else False

                    formatted = format_result(val, is_best, is_second)
                    print(f"  {method:15s}: {formatted}")
            print()


def generate_ablation_table(ablation_results: Dict):
    """生成消融实验LaTeX表格"""

    print(f"\n{'='*80}")
    print("Ablation Study Table")
    print(f"{'='*80}\n")

    metrics = ["Recall@20", "NDCG@20", "Hit@20", "Precision@20"]

    for variant, results in ablation_results.items():
        print(f"{variant:30s}", end=" & ")
        for i, metric in enumerate(metrics):
            val = results.get(metric)
            formatted = format_result(val, best=(variant == "RA-KG-PPO (Full)"))
            if i < len(metrics) - 1:
                print(f"{formatted} &", end=" ")
            else:
                print(f"{formatted} \\\\")
    print()


def calculate_improvements(results: Dict, dataset: str):
    """计算相对于最强baseline的提升百分比"""

    print(f"\n{'='*80}")
    print(f"Improvement Analysis - {dataset}")
    print(f"{'='*80}\n")

    our_method = "RA-KG-PPO"
    metrics = ["Recall@20", "NDCG@20", "Hit@20", "Precision@20"]

    for metric in metrics:
        our_score = results[dataset][our_method].get(metric)

        if our_score is None:
            print(f"{metric}: No data yet")
            continue

        # 找最强baseline
        best_baseline_score = 0
        best_baseline_name = None

        for method in results[dataset]:
            if method == our_method:
                continue

            score = results[dataset][method].get(metric)
            if score and score > best_baseline_score:
                best_baseline_score = score
                best_baseline_name = method

        if best_baseline_score > 0:
            improvement = ((our_score - best_baseline_score) / best_baseline_score) * 100
            print(f"{metric}:")
            print(f"  Best baseline: {best_baseline_name} = {best_baseline_score:.3f}")
            print(f"  Our method: {our_score:.3f}")
            print(f"  Improvement: +{improvement:.1f}%")
        else:
            print(f"{metric}: No baseline data for comparison")
        print()


# ============================================
# 示例：填充模拟数据
# ============================================

def generate_mock_results():
    """生成模拟数据用于演示"""

    results = {
        "amazon-book": {
            "RA-KG-PPO": {
                "Recall@20": 0.0856,
                "NDCG@20": 0.0645,
                "Hit@20": 0.4523,
                "Precision@20": 0.0428,
            },
            "KGAT": {
                "Recall@20": 0.0782,
                "NDCG@20": 0.0591,
                "Hit@20": 0.4102,
                "Precision@20": 0.0391,
            },
            "KGIN": {
                "Recall@20": 0.0798,
                "NDCG@20": 0.0603,
                "Hit@20": 0.4234,
                "Precision@20": 0.0399,
            },
            "SASRec": {
                "Recall@20": 0.0723,
                "NDCG@20": 0.0548,
                "Hit@20": 0.3891,
                "Precision@20": 0.0361,
            },
            "BERT4Rec": {
                "Recall@20": 0.0745,
                "NDCG@20": 0.0562,
                "Hit@20": 0.3967,
                "Precision@20": 0.0372,
            },
            "GRU4Rec": {
                "Recall@20": 0.0689,
                "NDCG@20": 0.0521,
                "Hit@20": 0.3712,
                "Precision@20": 0.0344,
            },
            "TPGR": {
                "Recall@20": 0.0734,
                "NDCG@20": 0.0556,
                "Hit@20": 0.3923,
                "Precision@20": 0.0367,
            },
            "BPR": {
                "Recall@20": 0.0612,
                "NDCG@20": 0.0467,
                "Hit@20": 0.3421,
                "Precision@20": 0.0306,
            },
            "NCF": {
                "Recall@20": 0.0645,
                "NDCG@20": 0.0489,
                "Hit@20": 0.3578,
                "Precision@20": 0.0322,
            },
        }
    }

    ablation = {
        "RA-KG-PPO (Full)": {
            "Recall@20": 0.0856,
            "NDCG@20": 0.0645,
            "Hit@20": 0.4523,
            "Precision@20": 0.0428,
        },
        "w/o Knowledge Graph": {
            "Recall@20": 0.0734,
            "NDCG@20": 0.0556,
            "Hit@20": 0.3923,
            "Precision@20": 0.0367,
        },
        "w/o Retrieval (LSH)": {
            "Recall@20": 0.0823,
            "NDCG@20": 0.0621,
            "Hit@20": 0.4389,
            "Precision@20": 0.0411,
        },
        "w/o Policy Conditioning": {
            "Recall@20": 0.0789,
            "NDCG@20": 0.0598,
            "Hit@20": 0.4201,
            "Precision@20": 0.0394,
        },
        "w/o PPO (use REINFORCE)": {
            "Recall@20": 0.0801,
            "NDCG@20": 0.0605,
            "Hit@20": 0.4267,
            "Precision@20": 0.0400,
        },
        "w/o GAE": {
            "Recall@20": 0.0812,
            "NDCG@20": 0.0614,
            "Hit@20": 0.4312,
            "Precision@20": 0.0406,
        },
    }

    return results, ablation


# ============================================
# 主函数
# ============================================

def main():
    print("\n" + "="*80)
    print("论文实验结果生成器")
    print("="*80)

    # 生成模拟数据（实际使用时替换为真实数据）
    results, ablation = generate_mock_results()

    # 保存模板
    print("\n[1] Saving results template...")
    with open('paper_results_template.json', 'w') as f:
        json.dump(RESULTS_TEMPLATE, f, indent=2)
    print("    Saved to: paper_results_template.json")
    print("    填充此文件后，重新运行此脚本生成LaTeX表格")

    # 生成表格
    print("\n[2] Generating LaTeX tables with mock data...")
    generate_main_results_table(results, "amazon-book")
    generate_ablation_table(ablation)

    # 计算提升
    print("\n[3] Calculating improvements...")
    calculate_improvements(results, "amazon-book")

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print("\n下一步:")
    print("1. 运行训练: bash start_training_5090.sh full")
    print("2. 收集结果: 从checkpoints和logs中提取指标")
    print("3. 填充 paper_results_template.json")
    print("4. 重新运行此脚本生成最终LaTeX表格")
    print("5. 复制生成的LaTeX代码到 paper_experiments.tex")
    print()


if __name__ == '__main__':
    main()
