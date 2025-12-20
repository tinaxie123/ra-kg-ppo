#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RA-KG-PPO


"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_kgat_data, create_user_sequences
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.trainer import RAKGPPO


def test_quick_training():
    """"""

    print("\n" + "="*70)
    print("RA-KG-PPO Quick Test")
    print("="*70 + "\n")

    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

        print("[1/6] Loading data...")

    try:
        data = load_kgat_data('amazon-book', './data/')
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        print("Please run: python scripts/prepare_data.py --dataset amazon-book")
        return False

    # 
    train_sequences = create_user_sequences(data['train_data'], min_seq_len=5)
    test_sequences = create_user_sequences(data['test_data'], min_seq_len=5)

    # 100
    train_sequences = dict(list(train_sequences.items())[:100])
    test_sequences = dict(list(test_sequences.items())[:20])

    item_embeddings = data['item_embeddings'].to(device)
    kg_embeddings = data['kg_embeddings'].to(device)

    print(f"  [OK] Train users: {len(train_sequences)}")
    print(f"  [OK] Test users: {len(test_sequences)}")
    print(f"  [OK] Items: {data['n_items']}")

        print("\n[2/6] Creating environment...")

    train_env = RecommendationEnv(
        user_sequences=train_sequences,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=device
    )

    test_env = RecommendationEnv(
        user_sequences=test_sequences,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=device
    )

    print("  [OK] Environments created")

        print("\n[3/6] Building models...")

    policy_net = RAPolicyValueNet(
        item_embedding_dim=64,
        hidden_dim=128,
        kg_embedding_dim=128,
        num_layers=2,
        shared_encoder=True
    ).to(device)

    candidate_generator = CandidateGenerator(
        hidden_dim=128,
        embedding_dim=128,
        num_hash_bits=8,
        num_tables=4,
        candidate_size=100
    )

    print("  Building LSH index...")
    candidate_generator.build_index(kg_embeddings)

    total_params = (
        sum(p.numel() for p in policy_net.parameters()) +
        sum(p.numel() for p in candidate_generator.parameters())
    )

    print(f"  [OK] Total parameters: {total_params:,}")

        print("\n[4/6] Creating trainer...")

    trainer = RAKGPPO(
        env=train_env,
        policy_net=policy_net,
        candidate_generator=candidate_generator,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_steps=512,  # 
        batch_size=64,
        n_epochs=4,  # 
        device=device
    )

    print("  [OK] Trainer created")

        print("\n[5/6] Training (quick test with 2 updates)...")
    print("-" * 70)

    try:
        trainer.learn(
            total_timesteps=1024,  # 2updates
            log_interval=1,
            eval_env=test_env,
            eval_freq=1
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n  [OK] Training completed successfully")

        print("\n[6/6] Evaluating...")

    eval_reward = trainer.evaluate(test_env, n_episodes=10)

    print(f"  [OK] Evaluation reward: {eval_reward:.4f}")

        print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"  Total timesteps: {trainer.num_timesteps}")
    print(f"  Total episodes: {len(trainer.episode_rewards)}")
    print(f"  Mean reward: {np.mean(trainer.episode_rewards):.4f}")
    print(f"  Eval reward: {eval_reward:.4f}")
    print()

    print("[OK] All components working correctly!")
    print("="*70 + "\n")

    return True


if __name__ == '__main__':
    success = test_quick_training()

    if success:
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("\n1. Full training:")
        print("   python train.py --dataset amazon-book --total-timesteps 100000")
        print("\n2. With GPU:")
        print("   python train.py --dataset amazon-book --device cuda")
        print("\n3. Custom parameters:")
        print("   python train.py --lr 5e-4 --gamma 0.95 --n-steps 4096")
        print("\n" + "="*70 + "\n")
        sys.exit(0)
    else:
        print("\n[ERROR] Test failed! Please check the error messages above.")
        sys.exit(1)
