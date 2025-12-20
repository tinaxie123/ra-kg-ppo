import os
import sys
import argparse
import torch
import torch.cuda.amp as amp
import numpy as np
from datetime import datetime
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_kgat_data, create_user_sequences
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.trainer import RAKGPPO
from utils.metrics import evaluate_ranking


def get_args():
    parser = argparse.ArgumentParser(
        description='RA-KG-PPO Training - 5090 Optimized'
    )
    parser.add_argument('--dataset', type=str, default='amazon-book',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str, default='./data/',
                       help='Data directory')
    parser.add_argument('--item-emb-dim', type=int, default=128,  # 128
                       help='Item embedding dimension')
    parser.add_argument('--kg-emb-dim', type=int, default=256,  # 256
                       help='KG embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,  # 256
                       help='Hidden state dimension')
    parser.add_argument('--num-layers', type=int, default=3,  # 
                       help='Number of GRU layers')
    parser.add_argument('--shared-encoder', action='store_true',
                       help='Share encoder between actor and critic')
    parser.add_argument('--num-hash-bits', type=int, default=10,  # 
                       help='Number of LSH hash bits')
    parser.add_argument('--num-tables', type=int, default=8,  # 
                       help='Number of LSH tables')
    parser.add_argument('--candidate-size', type=int, default=200,  # 
                       help='Candidate set size')
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
    parser.add_argument('--total-timesteps', type=int, default=1000000,  
                       help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=4096,  # rollout
                       help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=512,  
                       help='Batch size for SGD')
    parser.add_argument('--n-epochs', type=int, default=15,  # epoch
                       help='Epochs per update')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--amp-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='AMP dtype')
    parser.add_argument('--log-interval', type=int, default=1,
                       help='Log interval')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Checkpoint save frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_5090/',
                       help='Save directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()

def setup_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This script requires GPU.")

    device = torch.device('cuda:0')

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available Memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True  
    torch.backends.cuda.matmul.allow_tf32 = True  
    torch.backends.cudnn.allow_tf32 = True

    return device


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = setup_device()
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, 'config_5090.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")
    interactions, kg_data, item_emb_matrix = load_kgat_data(
        data_path=args.data_path,
        dataset=args.dataset
    )
    num_users = interactions['user_id'].max() + 1
    num_items = interactions['item_id'].max() + 1
    num_entities = kg_data['head_id'].max() + 1
    num_relations = kg_data['relation_id'].max() + 1

    print(f"Dataset loaded: {num_users} users, {num_items} items")
    print(f"KG: {num_entities} entities, {num_relations} relations")
    print(f"KG embeddings: {item_emb_matrix.shape}")
   
    train_seqs = create_user_sequences(interactions, max_len=20)
    print(f"Created {len(train_seqs)} training sequences")
  
    policy_net = RAPolicyValueNet(
        num_items=num_items,
        item_emb_dim=args.item_emb_dim,
        kg_emb_dim=args.kg_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        shared_encoder=args.shared_encoder
    )
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
    print(f"LSH initialized: {args.num_tables} tables, {args.num_hash_bits} bits")
 
    env = RecommendationEnv(
        interactions=interactions,
        train_sequences=train_seqs,
        num_items=num_items
    )
    print(f"Environment initialized with {len(train_seqs)} sequences")
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
    scaler = amp.GradScaler() if args.use_amp else None
    if args.use_amp:
        print(f"Mixed precision training enabled ({args.amp_dtype})")
    start_timestep = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_timestep = checkpoint['timestep']
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resumed from timestep {start_timestep}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Rollout steps: {args.n_steps}")
    print(f"Model parameters: {num_params:,}")
    print(f"Mixed precision: {args.use_amp}")

    num_updates = args.total_timesteps // args.n_steps

    for update in range(start_timestep // args.n_steps, num_updates):
        current_timestep = (update + 1) * args.n_steps
        metrics = trainer.train_step()
        if update % args.log_interval == 0:
            print(f"\nUpdate {update+1}/{num_updates} | Timestep {current_timestep:,}/{args.total_timesteps:,}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Explained Var: {metrics.get('explained_variance', 0):.4f}")
            print(f"  Mean Reward: {metrics.get('mean_reward', 0):.4f}")
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        if (update + 1) % args.eval_freq == 0:
            policy_net.eval()
            with torch.no_grad():
                eval_rewards = []
                for _ in range(100):
                    obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        candidates = candidate_generator.get_candidates(obs_tensor, k=args.candidate_size)
                        action_logits, _ = policy_net(obs_tensor, candidates)
                        action_probs = torch.softmax(action_logits, dim=-1)
                        action_idx = torch.argmax(action_probs, dim=-1).item()
                        action = candidates[0][action_idx].item()
                        obs, reward, done, _ = env.step(action)
                        episode_reward += reward
                    eval_rewards.append(episode_reward)

            mean_eval_reward = np.mean(eval_rewards)
            print(f"Evaluation - Mean Reward: {mean_eval_reward:.4f}")
            policy_net.train()
        if (update + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f'checkpoint_{current_timestep}.pt'
            )
            checkpoint = {
                'policy_net': policy_net.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'timestep': current_timestep,
                'config': vars(args)
            }
            if scaler:
                checkpoint['scaler'] = scaler.state_dict()

            torch.save(checkpoint, checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'policy_net': policy_net.state_dict(),
        'config': vars(args)
    }, final_path)
    print(f"\nFinal model saved to {final_path}")
    print("\nTraining completed!")

if __name__ == '__main__':
    main()
