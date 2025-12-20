"""
Ablation Study for RA-KG-PPO

This script runs ablation experiments to analyze the contribution of each component:
1. Full Model: RA-KG-PPO with all components
2. No-KG: Remove knowledge graph embeddings
3. No-LSH: Replace LSH retrieval with random sampling
4. No-PPO: Replace PPO with simple policy gradient (REINFORCE)

Usage:
    python experiments/ablation.py --dataset amazon-book --variant full
    python experiments/ablation.py --dataset amazon-book --variant no-kg
    python experiments/ablation.py --dataset amazon-book --variant no-lsh
    python experiments/ablation.py --dataset amazon-book --variant no-ppo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from data.dataset import load_kgat_data
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.trainer import PPOTrainer
from algorithms.rollout_buffer import RolloutBuffer
from utils.metrics import evaluate_policy


def get_args():
    parser = argparse.ArgumentParser(description='Ablation Study for RA-KG-PPO')

    # Experiment settings
    parser.add_argument('--variant', type=str, required=True,
                        choices=['full', 'no-kg', 'no-lsh', 'no-ppo'],
                        help='Ablation variant to run')
    parser.add_argument('--dataset', type=str, default='amazon-book',
                        choices=['amazon-book', 'last-fm', 'yelp2018'])
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model hyperparameters
    parser.add_argument('--item-emb-dim', type=int, default=128)
    parser.add_argument('--kg-emb-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)

    # Training hyperparameters
    parser.add_argument('--total-timesteps', type=int, default=100000)
    parser.add_argument('--n-steps', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)

    # Retrieval hyperparameters
    parser.add_argument('--candidate-size', type=int, default=200)
    parser.add_argument('--num-hash-bits', type=int, default=10)
    parser.add_argument('--num-tables', type=int, default=8)

    # Output
    parser.add_argument('--output-dir', type=str, default='./ablation_results/')
    parser.add_argument('--eval-freq', type=int, default=10000)

    return parser.parse_args()


class NoKGVariant:
    """Variant without knowledge graph embeddings"""

    def __init__(self, args, data):
        print(f"\n==> Initializing No-KG Variant")
        print("    Using only item embeddings (no KG information)")

        # Use only item embeddings, set KG embeddings to zeros
        self.kg_emb_dim = 0
        data['kg_embeddings'] = torch.zeros_like(data['kg_embeddings'])


class NoLSHVariant:
    """Variant with random sampling instead of LSH retrieval"""

    def __init__(self, args, candidate_gen, all_item_ids):
        print(f"\n==> Initializing No-LSH Variant")
        print("    Using random sampling instead of LSH retrieval")

        self.candidate_gen = candidate_gen
        self.all_item_ids = all_item_ids
        self.candidate_size = args.candidate_size

    def random_candidates(self, hidden):
        """Replace LSH with random sampling"""
        batch_size = hidden.shape[0]
        # Random sample candidates
        indices = np.random.choice(len(self.all_item_ids),
                                  size=(batch_size, self.candidate_size),
                                  replace=False)
        cand_ids = torch.tensor([[self.all_item_ids[i] for i in row] for row in indices])
        # Get embeddings for these candidates
        cand_embs = self.candidate_gen.item_embeddings[cand_ids]
        query = hidden
        return query, cand_ids, cand_embs


class NoPPOVariant:
    """Variant using simple REINFORCE instead of PPO"""

    def __init__(self, args):
        print(f"\n==> Initializing No-PPO Variant")
        print("    Using REINFORCE (simple policy gradient) instead of PPO")

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma

    def simple_pg_update(self, policy_net, buffer, optimizer):
        """Simple policy gradient update (REINFORCE)"""

        # Compute returns
        returns = []
        G = 0
        for r in reversed(buffer.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=buffer.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        log_probs = torch.stack(buffer.log_probs)
        policy_loss = -(log_probs * returns).mean()

        # Update
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer.step()

        return {'policy_loss': policy_loss.item()}


def run_ablation_experiment(args):
    """Run a single ablation experiment"""

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "="*80)
    print(f"ABLATION STUDY: {args.variant.upper()}")
    print("="*80)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_kgat_data(args.dataset, args.data_path)

    item_embeddings = data['item_embeddings'].to(args.device)
    kg_embeddings = data['kg_embeddings'].to(args.device)
    train_data = data['train_data']
    test_data = data['test_data']

    print(f"✓ Loaded {args.dataset}")
    print(f"  - Train users: {len(train_data)}")
    print(f"  - Test users: {len(test_data)}")
    print(f"  - Items: {item_embeddings.shape[0]}")

    # Apply variant modifications
    if args.variant == 'no-kg':
        variant = NoKGVariant(args, data)
        kg_embeddings = data['kg_embeddings'].to(args.device)

    # Create environment
    print("\n[2/5] Creating environment...")
    env = RecommendationEnv(
        user_sequences=train_data,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=args.device
    )
    print("✓ Environment created")

    # Build models
    print("\n[3/5] Building models...")
    policy_net = RAPolicyValueNet(
        item_embedding_dim=args.item_emb_dim,
        hidden_dim=args.hidden_dim,
        kg_embedding_dim=args.kg_emb_dim if args.variant != 'no-kg' else 0,
        num_layers=args.num_layers,
        shared_encoder=True
    ).to(args.device)

    candidate_gen = CandidateGenerator(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.kg_emb_dim,
        num_hash_bits=args.num_hash_bits,
        num_tables=args.num_tables,
        candidate_size=args.candidate_size
    ).to(args.device)

    # Build LSH index
    candidate_gen.build_index(kg_embeddings)
    print("✓ Models built")

    # Apply No-LSH variant if needed
    if args.variant == 'no-lsh':
        all_item_ids = list(range(item_embeddings.shape[0]))
        lsh_variant = NoLSHVariant(args, candidate_gen, all_item_ids)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(candidate_gen.parameters()),
        lr=args.learning_rate
    )

    # Trainer
    print("\n[4/5] Initializing trainer...")
    if args.variant == 'no-ppo':
        trainer_variant = NoPPOVariant(args)
        trainer = None
        print("✓ Using REINFORCE (simple PG)")
    else:
        trainer = PPOTrainer(
            policy=policy_net,
            candidate_gen=candidate_gen,
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            device=args.device
        )
        print("✓ Using PPO trainer")

    # Training
    print("\n[5/5] Training...")
    print("="*80)

    results = {
        'variant': args.variant,
        'args': vars(args),
        'training_history': [],
        'eval_history': []
    }

    buffer = RolloutBuffer(
        buffer_size=args.n_steps,
        device=args.device
    )

    timesteps = 0
    episode = 0

    pbar = tqdm(total=args.total_timesteps, desc="Training")

    while timesteps < args.total_timesteps:
        # Collect experience
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 50:
            # Get action
            with torch.no_grad():
                item_emb = torch.FloatTensor(obs['item_embeddings']).unsqueeze(0).to(args.device)
                length = torch.tensor([obs['length']]).to(args.device)

                hidden = policy_net.get_hidden_state(item_emb, length)

                if args.variant == 'no-lsh':
                    query, cand_ids, cand_embs = lsh_variant.random_candidates(hidden)
                else:
                    query, cand_ids, cand_embs = candidate_gen(hidden)

                logits = policy_net.actor.compute_action_logits(query, cand_embs)
                value = policy_net.critic(hidden)

                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

                action = cand_ids[0, action_idx].item()

            # Step
            next_obs, reward, done, info = env.step(action)

            # Store
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item() if value is not None else 0.0
            )

            episode_reward += reward
            obs = next_obs
            timesteps += 1
            steps += 1
            pbar.update(1)

            # Update policy
            if buffer.size() >= args.n_steps or done:
                if args.variant == 'no-ppo':
                    loss_info = trainer_variant.simple_pg_update(policy_net, buffer, optimizer)
                else:
                    loss_info = trainer.update(buffer)

                buffer.reset()

        episode += 1
        results['training_history'].append({
            'episode': episode,
            'timesteps': timesteps,
            'reward': episode_reward
        })

        # Evaluation
        if timesteps % args.eval_freq == 0 or timesteps >= args.total_timesteps:
            print(f"\n\nEvaluation at {timesteps} timesteps...")
            eval_metrics = evaluate_policy(
                policy_net, candidate_gen, test_data,
                item_embeddings, kg_embeddings,
                device=args.device, k=20
            )

            results['eval_history'].append({
                'timesteps': timesteps,
                **eval_metrics
            })

            print(f"Recall@20: {eval_metrics['recall@20']:.4f}")
            print(f"NDCG@20: {eval_metrics['ndcg@20']:.4f}")

    pbar.close()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.variant}_{args.dataset}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"✓ Ablation experiment completed: {args.variant}")
    print(f"✓ Results saved to: {output_file}")
    print("="*80)

    return results


def main():
    args = get_args()
    results = run_ablation_experiment(args)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    if results['eval_history']:
        final_eval = results['eval_history'][-1]
        print(f"Variant: {args.variant}")
        print(f"Recall@20: {final_eval['recall@20']:.4f}")
        print(f"NDCG@20: {final_eval['ndcg@20']:.4f}")
        print(f"Precision@20: {final_eval['precision@20']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
