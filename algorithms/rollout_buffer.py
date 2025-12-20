"""
 (Rollout Buffer)

PPO
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Generator, Tuple


class RolloutBuffer:
    """
    PPO

    
    -  (item_embeddings, kg_embeddings, lengths)
    -  (actions)
    -  (rewards)
    -  (values)
    -  (log_probs)
    -  (dones)
    -  (query_vectors, candidate_ids, candidate_embeddings)
    """

    def __init__(self, buffer_size: int, device: str = 'cpu'):
        """
        Args:
            buffer_size: 
            device: 
        """
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        # 
        self.observations = {
            'item_embeddings': [],
            'kg_embeddings': [],
            'lengths': []
        }
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # 
        self.query_vectors = []
        self.candidate_ids = []
        self.candidate_embeddings = []

        # GAE
        self.advantages = None
        self.returns = None

    def add(self,
            observation: Dict[str, np.ndarray],
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            done: bool,
            query_vector: torch.Tensor,
            candidate_ids: torch.Tensor,
            candidate_embeddings: torch.Tensor):
        """
        
        """
        if self.full:
            return

        # 
        self.observations['item_embeddings'].append(observation['item_embeddings'])
        self.observations['kg_embeddings'].append(observation['kg_embeddings'])
        self.observations['lengths'].append(observation['length'])

        # 
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        # 
        self.query_vectors.append(query_vector.cpu())
        self.candidate_ids.append(candidate_ids.cpu())
        self.candidate_embeddings.append(candidate_embeddings.cpu())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self,
                                       last_value: float,
                                       gamma: float = 0.99,
                                       gae_lambda: float = 0.95):
        """
        GAE

        Args:
            last_value: 
            gamma: 
            gae_lambda: GAE lambda
        """
        n_steps = len(self.rewards)

        # 
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae_lam = 0

        # numpy
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # GAE
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]

            # TD
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]

            # GAE
            advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        #  =  + 
        returns = advantages + values

        self.advantages = advantages
        self.returns = returns

    def get(self, batch_size: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        

        Args:
            batch_size: None

        Yields:
            batch: 
        """
        assert self.full or self.pos > 0, "Buffer is empty"

        n_steps = self.pos
        indices = np.arange(n_steps)

        if batch_size is None:
            batch_size = n_steps

        # 
        np.random.shuffle(indices)

        # tensor
        item_embs = torch.FloatTensor(
            np.array(self.observations['item_embeddings'])
        ).to(self.device)

        kg_embs = torch.FloatTensor(
            np.array(self.observations['kg_embeddings'])
        ).to(self.device)

        lengths = torch.LongTensor(
            np.array(self.observations['lengths'])
        ).to(self.device)

        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)

        # 
        query_vecs = torch.stack(self.query_vectors).to(self.device)
        cand_ids = torch.stack(self.candidate_ids).to(self.device)
        cand_embs = torch.stack(self.candidate_embeddings).to(self.device)

        # 
        start_idx = 0
        while start_idx < n_steps:
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield {
                'item_embeddings': item_embs[batch_indices],
                'kg_embeddings': kg_embs[batch_indices],
                'lengths': lengths[batch_indices],
                'actions': actions[batch_indices],
                'old_log_probs': old_log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
                'query_vectors': query_vecs[batch_indices],
                'candidate_ids': cand_ids[batch_indices],
                'candidate_embeddings': cand_embs[batch_indices]
            }

            start_idx += batch_size

    def reset(self):
        """"""
        self.observations = {
            'item_embeddings': [],
            'kg_embeddings': [],
            'lengths': []
        }
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.query_vectors = []
        self.candidate_ids = []
        self.candidate_embeddings = []
        self.advantages = None
        self.returns = None
        self.pos = 0
        self.full = False

    def __len__(self):
        return self.pos


if __name__ == '__main__':
    # 
    print("Testing RolloutBuffer...")

    buffer = RolloutBuffer(buffer_size=10)

    # 
    for i in range(10):
        obs = {
            'item_embeddings': np.random.randn(50, 64),
            'kg_embeddings': np.random.randn(50, 128),
            'length': 10 + i
        }
        action = i % 5
        reward = np.random.rand()
        value = np.random.rand()
        log_prob = -np.random.rand()
        done = (i == 9)

        query_vec = torch.randn(128)
        cand_ids = torch.randint(0, 100, (50,))
        cand_embs = torch.randn(50, 128)

        buffer.add(obs, action, reward, value, log_prob, done,
                   query_vec, cand_ids, cand_embs)

    print(f"[OK] Buffer size: {len(buffer)}")

    # 
    buffer.compute_returns_and_advantages(last_value=0.5)
    print(f"[OK] Advantages computed: {buffer.advantages.shape}")
    print(f"[OK] Returns computed: {buffer.returns.shape}")

    # 
    batch_count = 0
    for batch in buffer.get(batch_size=4):
        batch_count += 1
        print(f"[OK] Batch {batch_count}: actions shape = {batch['actions'].shape}")

    print("\n[OK] All tests passed!")
