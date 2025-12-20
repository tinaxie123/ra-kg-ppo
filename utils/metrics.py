"""
utils/metrics.py - 
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_recall_at_k(pred_items: List[int], gt_items: List[int], k: int) -> float:
    """
    Recall@K = |K ∩ | / ||
    
    Args:
        pred_items: 
        gt_items: 
        k: Top-K
    
    Returns:
        recall
    """
    pred_set = set(pred_items[:k])
    gt_set = set(gt_items)
    
    if len(gt_set) == 0:
        return 0.0
    
    return len(pred_set & gt_set) / len(gt_set)


def compute_ndcg_at_k(pred_items: List[int], gt_items: List[int], k: int) -> float:
    """
    NDCG@K: 
    
    Args:
        pred_items: 
        gt_items: 
        k: Top-K
    
    Returns:
        NDCG
    """
    pred_items = pred_items[:k]
    gt_set = set(gt_items)
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(pred_items):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)  # i+20
    
    # DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_items), k)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_hit_ratio_at_k(pred_items: List[int], gt_items: List[int], k: int) -> float:
    """
    Hit Ratio@K: K
    
    Args:
        pred_items: 
        gt_items: 
        k: Top-K
    
    Returns:
        1.0 ()  0.0 ()
    """
    pred_set = set(pred_items[:k])
    gt_set = set(gt_items)
    
    return 1.0 if len(pred_set & gt_set) > 0 else 0.0


def compute_precision_at_k(pred_items: List[int], gt_items: List[int], k: int) -> float:
    """
    Precision@K = |K ∩ | / K
    
    Args:
        pred_items: 
        gt_items: 
        k: Top-K
    
    Returns:
        precision
    """
    pred_set = set(pred_items[:k])
    gt_set = set(gt_items)
    
    return len(pred_set & gt_set) / k


def evaluate_ranking(pred_items: List[int], gt_items: List[int], 
                     k_list: List[int] = [10, 20, 50]) -> Dict[str, float]:
    """
    
    
    Args:
        pred_items: 
        gt_items: 
        k_list: K
    
    Returns:
        
    """
    metrics = {}
    
    for k in k_list:
        metrics[f'Recall@{k}'] = compute_recall_at_k(pred_items, gt_items, k)
        metrics[f'NDCG@{k}'] = compute_ndcg_at_k(pred_items, gt_items, k)
        metrics[f'Hit@{k}'] = compute_hit_ratio_at_k(pred_items, gt_items, k)
        metrics[f'Precision@{k}'] = compute_precision_at_k(pred_items, gt_items, k)
    
    return metrics


def evaluate_policy(
    policy,
    test_users: Dict[int, List[int]],
    item_embeddings: torch.Tensor,
    kg_embeddings: torch.Tensor,
    k_list: List[int] = [10, 20, 50],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    
    
    Args:
        policy: RA-KG-PPO
        test_users: 
        item_embeddings: 
        kg_embeddings: KG
        k_list: K
        device: 
    
    Returns:
        {
            'Recall@10': float,
            'Recall@20': float,
            'NDCG@10': float,
            'NDCG@20': float,
            ...
        }
    """
    
    policy.eval()
    
    metrics = {f'Recall@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics.update({f'Hit@{k}': [] for k in k_list})
    
    with torch.no_grad():
        for user_id, sequence in test_users.items():
            if len(sequence) < 5:
                continue
            
            # ground truth
            history = sequence[:-3]
            gt_items = sequence[-3:]
            
            if len(history) == 0:
                continue
            
            # 
            history_ids = torch.tensor(history[-50:], device=device).unsqueeze(0)
            history_embs = item_embeddings[history_ids]
            lengths = torch.tensor([len(history_ids[0])], device=device)
            
            # 
            hidden = policy.actor_critic.get_hidden_state(history_embs, lengths)
            query, cand_ids, cand_embs = policy.candidate_generator(hidden)
            
            # 
            logits = policy.actor_critic.actor.compute_action_logits(query, cand_embs)
            scores, indices = torch.sort(logits[0], descending=True)
            
            # ID
            pred_items = cand_ids[0][indices].cpu().tolist()
            
            # 
            for k in k_list:
                recall = compute_recall_at_k(pred_items, gt_items, k)
                ndcg = compute_ndcg_at_k(pred_items, gt_items, k)
                hit = compute_hit_ratio_at_k(pred_items, gt_items, k)
                
                metrics[f'Recall@{k}'].append(recall)
                metrics[f'NDCG@{k}'].append(ndcg)
                metrics[f'Hit@{k}'].append(hit)
    
    # 
    for key in metrics:
        if len(metrics[key]) > 0:
            metrics[key] = np.mean(metrics[key])
        else:
            metrics[key] = 0.0
    
    return metrics


def evaluate_baselines(
    test_users: Dict,
    item_embeddings: torch.Tensor,
    baseline_models: Dict,
    k_list: List[int] = [10, 20]
) -> pd.DataFrame:
    """
    
    
    Args:
        test_users: 
        item_embeddings: 
        baseline_models: {
            'KGAT': model,
            'SASRec': model,
            'TPGR': model,
            ...
        }
        k_list: K
    
    Returns:
         DataFrame
    """
    
    results = []
    
    for model_name, model in baseline_models.items():
        print(f"Evaluating {model_name}...")
        
        metrics = evaluate_policy(
            policy=model,
            test_users=test_users,
            item_embeddings=item_embeddings,
            kg_embeddings=None,  # 
            k_list=k_list
        )
        
        row = {'Model': model_name}
        row.update(metrics)
        results.append(row)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


if __name__ == '__main__':
    print("Testing metrics...")
    
    # 
    pred = [1, 5, 3, 8, 2, 7, 4, 9, 6]
    gt = [1, 3, 5]
    
    metrics = evaluate_ranking(pred, gt, k_list=[5, 10])
    
    print("\n:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    
    print("\n !")