#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RA-KG-PPO 

PyTorch
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import json

# 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_kgat_data, create_user_sequences
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.trainer import RAKGPPO
from utils.metrics import evaluate_ranking


def get_args():
    parser = argparse.ArgumentParser(
        description='RA-KG-PPO Training (Pure PyTorch)'
    )

    # 
    parser.add_argument('--dataset', type=str, default='amazon-book',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str, default='./data/',
                       help='Data directory')

    # 
    parser.add_argument('--item-emb-dim', type=int, default=64,
                       help='Item embedding dimension')
    parser.add_argument('--kg-emb-dim', type=int, default=128,
                       help='KG embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden state dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of GRU layers')
    parser.add_argument('--shared-encoder', action='store_true',
                       help='Share encoder between actor and critic')

    # LSH
    parser.add_argument('--num-hash-bits', type=int, default=8,
                       help='Number of LSH hash bits')
    parser.add_argument('--num-tables', type=int, default=4,
                       help='Number of LSH tables')
    parser.add_argument('--candidate-size', type=int, default=100,
                       help='Candidate set size')

    # PPO
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Max gradient norm')

    # 
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Epochs per update')
    parser.add_argument('--log-interval', type=int, default=1,
                       help='Log interval')
    parser.add_argument('--eval-freq', type=int, default=10,
                       help='Evaluation frequency')

    # 
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/',
                       help='Save directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    return parser.parse_args()


def main():
    args = get_args()

    # 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        args.save_dir,
        args.dataset,
        f'ra_kg_ppo_{timestamp}'
    )
    os.makedirs(save_path, exist_ok=True)

    # 
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*70)
    print("RA-KG-PPO Training (Pure PyTorch Implementation)")
    print("="*70 + "\n")

        print("Step 1: Loading data...")
    print("-" * 70)

    data = load_kgat_data(args.dataset, args.data_path)

    train_sequences = create_user_sequences(data['train_data'], min_seq_len=5)
    test_sequences = create_user_sequences(data['test_data'], min_seq_len=5)

    item_embeddings = data['item_embeddings'].to(args.device)
    kg_embeddings = data['kg_embeddings'].to(args.device)

    print(f"\n[OK] Data loaded")
    print(f"  Train users: {len(train_sequences)}")
    print(f"  Test users: {len(test_sequences)}")
    print(f"  Items: {data['n_items']}")
    print(f"  KG entities: {data['kg_data']['n_entities']}")

        print("\nStep 2: Creating environments...")
    print("-" * 70)

    train_env = RecommendationEnv(
        user_sequences=train_sequences,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=args.device
    )

    test_env = RecommendationEnv(
        user_sequences=test_sequences,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=args.device
    )

    print(f"[OK] Environments created")

        print("\nStep 3: Building models...")
    print("-" * 70)

    # -
    policy_net = RAPolicyValueNet(
        item_embedding_dim=args.item_emb_dim,
        hidden_dim=args.hidden_dim,
        kg_embedding_dim=args.kg_emb_dim,
        num_layers=args.num_layers,
        shared_encoder=args.shared_encoder
    ).to(args.device)

    # 
    candidate_generator = CandidateGenerator(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.kg_emb_dim,
        num_hash_bits=args.num_hash_bits,
        num_tables=args.num_tables,
        candidate_size=args.candidate_size
    )

    # LSH
    print("  Building LSH index...")
    candidate_generator.build_index(kg_embeddings)

    total_params = (
        sum(p.numel() for p in policy_net.parameters()) +
        sum(p.numel() for p in candidate_generator.parameters())
    )

    print(f"[OK] Models built")
    print(f"  Total parameters: {total_params:,}")

        print("\nStep 4: Creating trainer...")
    print("-" * 70)

    trainer = RAKGPPO(
        env=train_env,
        policy_net=policy_net,
        candidate_generator=candidate_generator,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=args.device
    )

    # 
    if args.resume:
        print(f"  Resuming from {args.resume}")
        trainer.load(args.resume)

    print(f"[OK] Trainer created")
    print(f"  Algorithm: PPO with GAE")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  GAE lambda: {args.gae_lambda}")

        print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")

    trainer.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        eval_env=test_env,
        eval_freq=args.eval_freq
    )

        print("\n" + "="*70)
    print("Saving model...")
    print("="*70)

    model_path = os.path.join(save_path, 'final_model.pth')
    trainer.save(model_path)

    print(f"[OK] Model saved to {model_path}")

        print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70 + "\n")

    trainer.policy_net.eval()

    # 
    final_reward = trainer.evaluate(test_env, n_episodes=100)

    print(f"Final evaluation reward: {final_reward:.4f}")

    # 
    stats = {
        'final_reward': final_reward,
        'total_timesteps': trainer.num_timesteps,
        'num_episodes': len(trainer.episode_rewards),
        'mean_reward': np.mean(trainer.episode_rewards[-100:]) if trainer.episode_rewards else 0,
        'mean_length': np.mean(trainer.episode_lengths[-100:]) if trainer.episode_lengths else 0
    }

    with open(os.path.join(save_path, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"\nResults saved to: {save_path}")
    print(f"  Model: {model_path}")
    print(f"  Config: {os.path.join(save_path, 'config.json')}")
    print(f"  Stats: {os.path.join(save_path, 'training_stats.json')}")


if __name__ == '__main__':
    main()
