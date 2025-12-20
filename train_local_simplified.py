#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

- CPU
- 
- 
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time

# 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_kgat_data, create_user_sequences
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.trainer import RAKGPPO
from utils.metrics import evaluate_ranking


def get_args():
    parser = argparse.ArgumentParser(description='RA-KG-PPO Local Training')

    # 
    parser.add_argument('--dataset', type=str, default='amazon-book',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str, default='./data/',
                       help='Data directory')

    # 
    parser.add_argument('--item-emb-dim', type=int, default=64,
                       help='Item embedding dimension')
    parser.add_argument('--kg-emb-dim', type=int, default=64,
                       help='KG embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden state dimension')
    parser.add_argument('--num-layers', type=int, default=1,
                       help='Number of GRU layers')

    # LSH
    parser.add_argument('--num-hash-bits', type=int, default=6,
                       help='Number of LSH hash bits')
    parser.add_argument('--num-tables', type=int, default=2,
                       help='Number of LSH tables')
    parser.add_argument('--candidate-size', type=int, default=50,
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
    parser.add_argument('--total-timesteps', type=int, default=10000,
                       help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=512,
                       help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=4,
                       help='Epochs per update')
    parser.add_argument('--log-interval', type=int, default=1,
                       help='Log interval')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='Evaluation frequency')

    # 
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_local/',
                       help='Save directory')

    return parser.parse_args()


def setup_device(device_arg):
    """"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print("\n" + "="*60)
    print("Device Configuration")
    print("="*60)
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Running on CPU (slower but compatible)")

    print("="*60 + "\n")

    return device


def quick_eval(policy_net, env, candidate_generator, device, num_episodes=50):
    """"""
    policy_net.eval()

    eval_rewards = []
    eval_lengths = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done and episode_length < 20:  # 
                # tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

                # 
                candidates = candidate_generator.get_candidates(
                    obs_tensor,
                    k=candidate_generator.candidate_size
                )

                # 
                action_logits, _ = policy_net(obs_tensor, candidates)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_idx = torch.argmax(action_probs, dim=-1).item()
                action = candidates[0][action_idx].item()

                # 
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

    policy_net.train()

    return {
        'mean_reward': np.mean(eval_rewards),
        'mean_length': np.mean(eval_lengths)
    }


def main():
    args = get_args()

    # 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 
    device = setup_device(args.device)

    # 
    os.makedirs(args.save_dir, exist_ok=True)

    # 
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("[1/6] Loading data...")
    start_time = time.time()

    try:
        data_dict = load_kgat_data(args.dataset, args.data_path)
        train_data = data_dict['train_data']
        test_data = data_dict['test_data']
        num_users = data_dict['n_users']
        num_items = data_dict['n_items']
        item_emb_matrix = data_dict['kg_embeddings'].numpy()

        # interactions DataFrame
        interactions = pd.DataFrame([
            {'user_id': uid, 'item_id': iid}
            for uid, items in train_data.items()
            for iid in items
        ])
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run: python scripts/prepare_data.py --dataset amazon-book")
        return

    print(f"Dataset loaded in {time.time() - start_time:.1f}s")
    print(f"Users: {num_users}, Items: {num_items}")

    print("\n[2/6] Creating user sequences...")
    train_seqs = create_user_sequences(interactions, max_len=10)
    # 
    train_seqs = dict(list(train_seqs.items())[:min(len(train_seqs), 5000)])
    print(f"Using {len(train_seqs)} training sequences")

    print("\n[3/6] Building model...")
    policy_net = RAPolicyValueNet(
        num_items=num_items,
        item_emb_dim=args.item_emb_dim,
        kg_emb_dim=args.kg_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        shared_encoder=False
    )

    # KG
    with torch.no_grad():
        policy_net.kg_embeddings.weight.copy_(
            torch.from_numpy(item_emb_matrix).float()
        )

    num_params = sum(p.numel() for p in policy_net.parameters())
    print(f"Model parameters: {num_params:,}")

    print("\n[4/6] Building candidate generator...")
    candidate_generator = CandidateGenerator(
        item_embeddings=item_emb_matrix,
        num_hash_bits=args.num_hash_bits,
        num_tables=args.num_tables,
        candidate_size=args.candidate_size
    )

    print("\n[5/6] Building environment...")
    env = RecommendationEnv(
        interactions=interactions,
        train_sequences=train_seqs,
        num_items=num_items
    )

    print("\n[6/6] Building trainer...")
    trainer = RAKGPPO(
        env=env,
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
        device=str(device)
    )

    # 
    print("\n" + "="*60)
    print("Starting Training (Local Mode)")
    print("="*60)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    num_updates = args.total_timesteps // args.n_steps
    training_start = time.time()

    results_log = {
        'config': vars(args),
        'training_metrics': [],
        'eval_metrics': []
    }

    for update in range(num_updates):
        update_start = time.time()
        current_timestep = (update + 1) * args.n_steps

        # 
        metrics = trainer.train_step()
        metrics['update'] = update + 1
        metrics['timestep'] = current_timestep
        metrics['time'] = time.time() - update_start
        results_log['training_metrics'].append(metrics)

        # 
        if update % args.log_interval == 0:
            print(f"\nUpdate {update+1}/{num_updates} | Timestep {current_timestep:,}/{args.total_timesteps:,}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Reward: {metrics.get('mean_reward', 0):.4f}")
            print(f"  Update Time: {metrics['time']:.2f}s")

        # 
        if (update + 1) % args.eval_freq == 0:
            print("\nEvaluating...")
            eval_start = time.time()
            eval_metrics = quick_eval(
                policy_net, env, candidate_generator, device, num_episodes=50
            )
            eval_metrics['update'] = update + 1
            eval_metrics['timestep'] = current_timestep
            eval_metrics['time'] = time.time() - eval_start
            results_log['eval_metrics'].append(eval_metrics)

            print(f"Eval - Mean Reward: {eval_metrics['mean_reward']:.4f}, "
                  f"Mean Length: {eval_metrics['mean_length']:.2f}")

    training_time = time.time() - training_start

    # 
    results_log['training_time_seconds'] = training_time
    results_path = os.path.join(args.save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    # 
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'policy_net': policy_net.state_dict(),
        'config': vars(args),
        'training_time': training_time,
        'final_metrics': results_log['training_metrics'][-1] if results_log['training_metrics'] else {}
    }, final_path)

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    print(f"Model saved to: {final_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
