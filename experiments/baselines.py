"""
Baseline Comparison for RA-KG-PPO

This script implements and evaluates baseline recommendation methods:
1. PopRec: Popularity-based recommendation
2. BPR: Bayesian Personalized Ranking
3. GRU4Rec: GRU-based sequential recommendation
4. SASRec: Self-Attentive Sequential Recommendation 
Usage:
    # Run all baselines
    python experiments/baselines.py --dataset amazon-book --method all

    # Run specific baseline
    python experiments/baselines.py --dataset amazon-book --method bpr
    python experiments/baselines.py --dataset amazon-book --method gru4rec
    python experiments/baselines.py --dataset amazon-book --method sasrec
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from tqdm import tqdm
from collections import Counter
from data.dataset import load_kgat_data
from utils.metrics import evaluate_policy


def get_args():
    parser = argparse.ArgumentParser(description='Baseline Comparison')

    parser.add_argument('--method', type=str, required=True,
                        choices=['all', 'pop', 'bpr', 'gru4rec', 'sasrec'],
                        help='Baseline method to evaluate')
    parser.add_argument('--dataset', type=str, default='amazon-book',
                        choices=['amazon-book', 'last-fm', 'yelp2018'])
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)  # For SASRec
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    # Evaluation
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='./baseline_results/')

    return parser.parse_args()


class PopRec:

    def __init__(self, train_data):
        print("Initializing PopRec (Popularity-based)...")
        self.item_counts = Counter()
        for user_seq in train_data.values():
            self.item_counts.update(user_seq)
        self.popular_items = [item for item, count in self.item_counts.most_common()]

        print(f" Found {len(self.popular_items)} items")

    def predict(self, user_history, k=20):
        history_set = set(user_history)
        recommendations = []

        for item in self.popular_items:
            if item not in history_set:
                recommendations.append(item)
                if len(recommendations) >= k:
                    break

        return recommendations

class BPRModel(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        user_emb = self.user_embedding(user_ids)
        pos_item_emb = self.item_embedding(pos_item_ids)
        neg_item_emb = self.item_embedding(neg_item_ids)

        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)

        return pos_scores, neg_scores

    def predict(self, user_id, item_ids):
        user_emb = self.user_embedding(user_id)
        item_embs = self.item_embedding(item_ids)
        scores = (user_emb * item_embs).sum(dim=1)
        return scores


class BPR:

    def __init__(self, num_users, num_items, embedding_dim, device, lr=1e-3):
        print(f"Initializing BPR (num_users={num_users}, num_items={num_items})...")
        self.model = BPRModel(num_users, num_items, embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.num_items = num_items

    def train_epoch(self, train_data, batch_size=256):

        self.model.train()
        user_ids = []
        pos_items = []
        neg_items = []

        for user_id, items in train_data.items():
            if len(items) < 2:
                continue
            for pos_item in items:
                user_ids.append(user_id)
                pos_items.append(pos_item)
                # Sample negative item
                neg_item = np.random.randint(0, self.num_items)
                while neg_item in items:
                    neg_item = np.random.randint(0, self.num_items)
                neg_items.append(neg_item)
        indices = np.random.permutation(len(user_ids))
        user_ids = [user_ids[i] for i in indices]
        pos_items = [pos_items[i] for i in indices]
        neg_items = [neg_items[i] for i in indices]
        total_loss = 0
        num_batches = 0

        for i in range(0, len(user_ids), batch_size):
            batch_users = torch.LongTensor(user_ids[i:i+batch_size]).to(self.device)
            batch_pos = torch.LongTensor(pos_items[i:i+batch_size]).to(self.device)
            batch_neg = torch.LongTensor(neg_items[i:i+batch_size]).to(self.device)

            pos_scores, neg_scores = self.model(batch_users, batch_pos, batch_neg)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def predict(self, user_id, user_history, k=20):
        self.model.eval()

        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            all_items = torch.LongTensor(list(range(self.num_items))).to(self.device)

            scores = self.model.predict(user_tensor, all_items).cpu().numpy()
            history_set = set(user_history)
            for item in history_set:
                if item < len(scores):
                    scores[item] = -np.inf
            top_k = np.argsort(scores)[-k:][::-1]

        return top_k.tolist(

class GRU4RecModel(nn.Module):


    def __init__(self, num_items, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_items)

        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, item_seq):
        """Forward pass"""
        item_emb = self.item_embedding(item_seq)
        output, _ = self.gru(item_emb)
        logits = self.fc(output[:, -1, :])  # Use last hidden state
        return logits


class GRU4Rec:
  
    def __init__(self, num_items, embedding_dim, hidden_dim, num_layers, device, lr=1e-3, dropout=0.2):
        print(f"Initializing GRU4Rec...")
        self.model = GRU4RecModel(num_items, embedding_dim, hidden_dim, num_layers, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.num_items = num_items

    def train_epoch(self, train_data, batch_size=256, max_seq_len=50):
     
        self.model.train()
        sequences = []
        targets = []

        for user_id, items in train_data.items():
            if len(items) < 2:
                continue
            for i in range(1, len(items)):
                seq = items[max(0, i-max_seq_len):i]
                # Pad if needed
                if len(seq) < max_seq_len:
                    seq = [0] * (max_seq_len - len(seq)) + seq
                sequences.append(seq)
                targets.append(items[i])
        indices = np.random.permutation(len(sequences))
        sequences = [sequences[i] for i in indices]
        targets = [targets[i] for i in indices]
        total_loss = 0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch_seq = torch.LongTensor(sequences[i:i+batch_size]).to(self.device)
            batch_target = torch.LongTensor(targets[i:i+batch_size]).to(self.device)

            logits = self.model(batch_seq)
            loss = F.cross_entropy(logits, batch_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def predict(self, user_history, k=20, max_seq_len=50):
     
        self.model.eval()

        with torch.no_grad():
          
            seq = user_history[-max_seq_len:]
            if len(seq) < max_seq_len:
                seq = [0] * (max_seq_len - len(seq)) + seq

            seq_tensor = torch.LongTensor([seq]).to(self.device)
            logits = self.model(seq_tensor)[0].cpu().numpy()
            history_set = set(user_history)
            for item in history_set:
                if item < len(logits):
                    logits[item] = -np.inf
            top_k = np.argsort(logits)[-k:][::-1]

        return top_k.tolist()


def evaluate_baseline(method, test_data, k=20):
    """Evaluate baseline method"""

    print(f"\nEvaluating {method.__class__.__name__}...")

    recalls = []
    ndcgs = []
    precisions = []

    for user_id, true_items in tqdm(test_data.items(), desc="Evaluating"):
        if len(true_items) == 0:
            continue
        if hasattr(method, 'predict'):
            if 'user_history' in method.predict.__code__.co_varnames:
               
                predictions = method.predict(user_history=true_items[:-1], k=k)
            else:
             
                predictions = method.predict(user_id=user_id, user_history=true_items[:-1], k=k)
        else:
            predictions = []
        true_set = set(true_items[-1:])  # Last item as ground truth
        pred_set = set(predictions[:k])
        recall = len(true_set & pred_set) / len(true_set) if len(true_set) > 0 else 0
        recalls.append(recall)
        precision = len(true_set & pred_set) / k if k > 0 else 0
        precisions.append(precision)
        dcg = 0
        for i, item in enumerate(predictions[:k]):
            if item in true_set:
                dcg += 1 / np.log2(i + 2)
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_set), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    results = {
        f'recall@{k}': np.mean(recalls),
        f'ndcg@{k}': np.mean(ndcgs),
        f'precision@{k}': np.mean(precisions)
    }

    return results


def run_baseline(args, method_name):
 
    data = load_kgat_data(args.dataset, args.data_path)
    train_data = data['train_data']
    test_data = data['test_data']

    num_items = data['item_embeddings'].shape[0]
    num_users = len(train_data)
    print(f"   Train users: {num_users}")
    print(f"   Test users: {len(test_data)}")
    print(f"   Items: {num_items}")
    if method_name == 'pop':
        method = PopRec(train_data)

    elif method_name == 'bpr':
        method = BPR(num_users, num_items, args.embedding_dim, args.device, args.learning_rate)
        for epoch in range(args.epochs):
            loss = method.train_epoch(train_data, args.batch_size)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    elif method_name == 'gru4rec':
        method = GRU4Rec(num_items, args.embedding_dim, args.hidden_dim,
                        args.num_layers, args.device, args.learning_rate, args.dropout)
       
        for epoch in range(args.epochs):
            loss = method.train_epoch(train_data, args.batch_size)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    else:
        raise ValueError(f"Unknown method: {method_name}")

    results = evaluate_baseline(method, test_data, k=args.k)
    print(f"Recall@{args.k}: {results[f'recall@{args.k}']:.4f}")
    print(f"NDCG@{args.k}: {results[f'ndcg@{args.k}']:.4f}")
    print(f"Precision@{args.k}: {results[f'precision@{args.k}']:.4f}")
    print("="*80)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{method_name}_{args.dataset}.json')
    with open(output_file, 'w') as f:
        json.dump({
            'method': method_name,
            'dataset': args.dataset,
            'results': results
        }, f, indent=2)

    print(f"\n Results saved to: {output_file}")

    return results


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.method == 'all':
        methods = ['pop', 'bpr', 'gru4rec']
        all_results = {}

        for method in methods:
            results = run_baseline(args, method)
            all_results[method] = results

        for method, results in all_results.items():
            print(f"\n{method.upper()}:")
            print(f"  Recall@{args.k}: {results[f'recall@{args.k}']:.4f}")
            print(f"  NDCG@{args.k}: {results[f'ndcg@{args.k}']:.4f}")
        print("="*80)

    else:
        run_baseline(args, args.method)


if __name__ == '__main__':
    main()
