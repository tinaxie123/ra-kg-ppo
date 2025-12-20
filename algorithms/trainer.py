

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from algorithms.rollout_buffer import RolloutBuffer


class RAKGPPO:
    """
    RA-KG-PPO PyTorch

    
    1.  + 
    2. 
    3. PPO
    4. 
    """

    def __init__(self,
                 env: RecommendationEnv,
                 policy_net: RAPolicyValueNet,
                 candidate_generator: CandidateGenerator,
                 # PPO
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 # 
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 # 
                 device: str = 'cpu'):
        """
        Args:
            env: 
            policy_net: -
            candidate_generator: 
            learning_rate: 
            gamma: 
            gae_lambda: GAE lambda
            clip_range: PPO
            clip_range_vf: 
            entropy_coef: 
            value_coef: 
            max_grad_norm: 
            n_steps: 
            batch_size: 
            n_epochs: epoch
            device: 
        """
        self.env = env
        self.policy_net = policy_net.to(device)
        self.candidate_generator = candidate_generator
        self.device = device

        # PPO
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # 
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # 
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.candidate_generator.parameters()),
            lr=learning_rate
        )

        # 
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            device=device
        )

        # 
        self.num_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollouts(self) -> bool:
        """
        

        Returns:
            
        """
        self.policy_net.eval()
        self.rollout_buffer.reset()

        episode_reward = 0
        episode_length = 0

        # 
        obs = self.env.reset()

        for step in range(self.n_steps):
            # 
            item_embs = torch.FloatTensor(
                obs['item_embeddings']
            ).unsqueeze(0).to(self.device)

            lengths = torch.LongTensor([obs['length']]).to(self.device)

            with torch.no_grad():
                # 
                hidden_state = self.policy_net.get_hidden_state(
                    item_embs, lengths
                )

                # 
                query_vector, candidate_ids, candidate_embeddings = \
                    self.candidate_generator(hidden_state)

                # 
                logits = self.policy_net.actor.compute_action_logits(
                    query_vector, candidate_embeddings
                )

                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

                # 
                value = self.policy_net.critic(item_embs, lengths).squeeze(-1)

            # ID
            action = candidate_ids[0, action_idx.item()].item()

            # 
            next_obs, reward, done, info = self.env.step(action)

            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1

            # buffer
            self.rollout_buffer.add(
                observation=obs,
                action=action_idx.item(),  # 
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done,
                query_vector=query_vector[0],
                candidate_ids=candidate_ids[0],
                candidate_embeddings=candidate_embeddings[0]
            )

            obs = next_obs

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                episode_reward = 0
                episode_length = 0
                obs = self.env.reset()

        # 
        with torch.no_grad():
            item_embs = torch.FloatTensor(
                obs['item_embeddings']
            ).unsqueeze(0).to(self.device)
            lengths = torch.LongTensor([obs['length']]).to(self.device)
            last_value = self.policy_net.critic(item_embs, lengths).squeeze(-1).item()

        # 
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        return True

    def train(self) -> Dict[str, float]:
        """
        epoch

        Returns:
            
        """
        self.policy_net.train()

        # 
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                # 
                item_embs = batch['item_embeddings']
                lengths = batch['lengths']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                query_vecs = batch['query_vectors']
                cand_embs = batch['candidate_embeddings']

                # 
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 
                log_probs, values, entropy = self.policy_net.evaluate_actions(
                    item_embs, query_vecs, cand_embs, actions, lengths
                )

                # PPO
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # 
                if self.clip_range_vf is not None:
                    values_pred = values
                    values_pred_clipped = torch.clamp(
                        values,
                        returns - self.clip_range_vf,
                        returns + self.clip_range_vf
                    )
                    value_loss_1 = (returns - values_pred).pow(2)
                    value_loss_2 = (returns - values_pred_clipped).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                # 
                entropy_loss = -entropy.mean()

                # 
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # 
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) +
                    list(self.candidate_generator.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # clip fraction
                with torch.no_grad():
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > self.clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions)
        }

    def learn(self,
              total_timesteps: int,
              log_interval: int = 1,
              eval_env: Optional[RecommendationEnv] = None,
              eval_freq: int = 10) -> 'RAKGPPO':
        """
        

        Args:
            total_timesteps: 
            log_interval: 
            eval_env: 
            eval_freq: 

        Returns:
            self
        """
        num_updates = total_timesteps // self.n_steps

        print(f"\n{'='*60}")
        print(f"Starting RA-KG-PPO Training")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Number of updates: {num_updates}")
        print(f"Steps per update: {self.n_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs per update: {self.n_epochs}")
        print(f"{'='*60}\n")

        for update in tqdm(range(1, num_updates + 1), desc="Training"):
            # 
            self.collect_rollouts()

            # 
            train_stats = self.train()

            # 
            if update % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0

                print(f"\nUpdate {update}/{num_updates}")
                print(f"  Timesteps: {self.num_timesteps}")
                print(f"  Episodes: {len(self.episode_rewards)}")
                print(f"  Mean reward (last 100): {mean_reward:.4f}")
                print(f"  Mean length (last 100): {mean_length:.2f}")
                print(f"  Policy loss: {train_stats['policy_loss']:.4f}")
                print(f"  Value loss: {train_stats['value_loss']:.4f}")
                print(f"  Entropy: {-train_stats['entropy_loss']:.4f}")
                print(f"  Clip fraction: {train_stats['clip_fraction']:.4f}")

            # 
            if eval_env is not None and update % eval_freq == 0:
                eval_reward = self.evaluate(eval_env, n_episodes=10)
                print(f"  Evaluation reward: {eval_reward:.4f}")

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}\n")

        return self

    def evaluate(self, env: RecommendationEnv, n_episodes: int = 10) -> float:
        """
        

        Args:
            env: 
            n_episodes: 

        Returns:
            
        """
        self.policy_net.eval()
        episode_rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                item_embs = torch.FloatTensor(
                    obs['item_embeddings']
                ).unsqueeze(0).to(self.device)
                lengths = torch.LongTensor([obs['length']]).to(self.device)

                with torch.no_grad():
                    hidden_state = self.policy_net.get_hidden_state(item_embs, lengths)
                    query_vector, candidate_ids, candidate_embeddings = \
                        self.candidate_generator(hidden_state)

                    logits = self.policy_net.actor.compute_action_logits(
                        query_vector, candidate_embeddings
                    )

                    # 
                    action_idx = torch.argmax(logits, dim=-1)
                    action = candidate_ids[0, action_idx.item()].item()

                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return np.mean(episode_rewards)

    def save(self, path: str):
        """"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'candidate_generator': self.candidate_generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.candidate_generator.load_state_dict(checkpoint['candidate_generator'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    print("RA-KG-PPO Trainer")
    print("Use train.py for training")
