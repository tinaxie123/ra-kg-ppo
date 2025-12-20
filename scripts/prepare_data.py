#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    load_kgat_data,
    create_user_sequences,
    train_transe_embeddings
)


def print_data_statistics(data: dict):
    print(f"\n【用户-物品交互】")
    print(f"  训练用户数: {len(data['train_data'])}")
    print(f"  测试用户数: {len(data['test_data'])}")
    print(f"  总用户数: {data['n_users']}")
    print(f"  总物品数: {data['n_items']}")
    train_interactions = sum(len(items) for items in data['train_data'].values())
    test_interactions = sum(len(items) for items in data['test_data'].values())

    print(f"  训练交互数: {train_interactions}")
    print(f"  测试交互数: {test_interactions}")
    print(f"  平均每用户交互数 (训练): {train_interactions / len(data['train_data']):.2f}")
    seq_lengths = [len(items) for items in data['train_data'].values()]
    print(f"\n  序列长度分布:")
    print(f"    Min: {min(seq_lengths)}")
    print(f"    Max: {max(seq_lengths)}")
    print(f"    Mean: {np.mean(seq_lengths):.2f}")
    print(f"    Median: {np.median(seq_lengths):.0f}")

    print(f"\n【知识图谱】")
    kg_data = data['kg_data']
    print(f"  实体数: {kg_data['n_entities']}")
    print(f"  关系数: {kg_data['n_relations']}")
    print(f"  三元组数: {len(kg_data['triples'])}")

    if len(kg_data['triples']) > 0:
       
        max_possible_triples = kg_data['n_entities'] * kg_data['n_relations'] * kg_data['n_entities']
        density = len(kg_data['triples']) / max_possible_triples * 100
        print(f"  KG密度: {density:.6f}%")
        from collections import defaultdict
        entity_degrees = defaultdict(int)
        for h, r, t in kg_data['triples']:
            entity_degrees[h] += 1
            entity_degrees[t] += 1

        degrees = list(entity_degrees.values())
        print(f"\n  实体度分布:")
        print(f"    Min: {min(degrees)}")
        print(f"    Max: {max(degrees)}")
        print(f"    Mean: {np.mean(degrees):.2f}")
        print(f"    Median: {np.median(degrees):.0f}")

    print(f"\n【嵌入】")
    print(f"  物品嵌入维度: {data['item_embeddings'].shape}")
    print(f"  KG嵌入维度: {data['kg_embeddings'].shape}")

    

def verify_data_integrity(data: dict):
    

    errors = []
    warnings = []
    all_items = set()
    for items in data['train_data'].values():
        all_items.update(items)
    for items in data['test_data'].values():
        all_items.update(items)

    max_item_id = max(all_items)
    if max_item_id >= data['n_items']:
        errors.append(f"物品ID超出范围: max={max_item_id}, n_items={data['n_items']}")
    else:
        print(f"  [OK] 物品ID范围正常: [0, {max_item_id}]")
    if data['item_embeddings'].shape[0] != data['n_items']:
        errors.append(
            f"物品嵌入数量不匹配: {data['item_embeddings'].shape[0]} != {data['n_items']}"
        )
    else:
        print(f"物品嵌入维度: {data['item_embeddings'].shape}")

    if data['kg_embeddings'].shape[0] != data['kg_data']['n_entities']:
        errors.append(
            f"KG嵌入数量不匹配: {data['kg_embeddings'].shape[0]} != {data['kg_data']['n_entities']}"
        )
    else:
        print(f"  [OK] KG嵌入维度: {data['kg_embeddings'].shape}")
    item_emb_mean = torch.mean(torch.abs(data['item_embeddings'])).item()
    kg_emb_mean = torch.mean(torch.abs(data['kg_embeddings'])).item()

    if item_emb_mean < 1e-5:
        warnings.append("物品嵌入可能未正确初始化(均值过小)")
    else:
        print(f"  [OK] 物品嵌入均值: {item_emb_mean:.6f}")

    if kg_emb_mean < 1e-5:
        warnings.append("KG嵌入可能未正确初始化(均值过小)")
    else:
        print(f"  [OK] KG嵌入均值: {kg_emb_mean:.6f}")
    if len(data['kg_data']['triples']) > 0:
      
        n_entities = data['kg_data']['n_entities']
        n_relations = data['kg_data']['n_relations']

        invalid_triples = 0
        for h, r, t in data['kg_data']['triples'][:1000]:  # 采样检查
            if h >= n_entities or t >= n_entities:
                invalid_triples += 1
            if r >= n_relations:
                invalid_triples += 1

        if invalid_triples > 0:
            errors.append(f"发现{invalid_triples}个无效三元组(采样1000条)")
        else:
            print(f"  [OK] 三元组格式正确")
    if errors:
        print("[ERROR] 发现错误:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("\n[WARN] 警告:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print("[OK] 所有检查通过!")

    print("="*60 + "\n")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='RA-KG-PPO 数据预处理'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='amazon-book',
        choices=['amazon-book', 'last-fm', 'yelp2018'],
        help='数据集名称'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/',
        help='数据路径'
    )
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='强制重新构建嵌入'
    )
    parser.add_argument(
        '--transe_epochs',
        type=int,
        default=50,
        help='TransE训练轮数'
    )

    args = parser.parse_args()
    if args.force_rebuild:
        dataset_path = os.path.join(args.data_path, args.dataset)
        item_emb_file = os.path.join(dataset_path, 'item_embeddings.npy')
        kg_emb_file = os.path.join(dataset_path, 'kg_embeddings.npy')

        if os.path.exists(item_emb_file):
            os.remove(item_emb_file)
            print(f"已删除: {item_emb_file}")

        if os.path.exists(kg_emb_file):
            os.remove(kg_emb_file)
            print(f"已删除: {kg_emb_file}")
    print(f"\n开始加载数据集: {args.dataset}")
    print(f"数据路径: {args.data_path}")
    print(f"TransE训练轮数: {args.transe_epochs}\n")

    try:
        data = load_kgat_data(args.dataset, args.data_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] 错误: {e}")
        sys.exit(1)
    print_data_statistics(data)
    is_valid = verify_data_integrity(data)

    if not is_valid:
        print("[ERROR] 数据验证失败!")
        sys.exit(1)

    print("\n[OK] 数据预处理完成!")
    print(f"[OK] 数据已保存到: {os.path.join(args.data_path, args.dataset)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
