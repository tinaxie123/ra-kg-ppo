

import gym
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional


class RecommendationEnv(gym.Env):
    
    def __init__(self, user_sequences: Dict[int, List[int]],
                 item_embeddings: torch.Tensor,
                 kg_embeddings: torch.Tensor,
                 max_seq_len: int = 50,
                 reward_type: str = 'click',
                 device: str = 'cpu'):
        super().__init__()
        
        self.user_sequences = user_sequences
        self.user_ids = list(user_sequences.keys())
        self.item_embeddings = item_embeddings.to(device)
        self.kg_embeddings = kg_embeddings.to(device)
        self.max_seq_len = max_seq_len
        self.reward_type = reward_type
        self.device = device
        
        self.num_items = item_embeddings.shape[0]
        self.embedding_dim = item_embeddings.shape[1]
        self.kg_embedding_dim = kg_embeddings.shape[1]
        
        # 
        self.current_user = None
        self.current_seq = None
        self.current_pos = 0
        self.done = False
        
        # Gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_seq_len, self.embedding_dim),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_items)
    
    def reset(self, user_id: Optional[int] = None) -> Dict:
        """
        
        
        Args:
            user_id: ID (None)
        
        Returns:
            observation: 
        """
        if user_id is None:
            user_id = np.random.choice(self.user_ids)
        
        self.current_user = user_id
        full_seq = self.user_sequences[user_id]
        
        # 
        min_start_pos = max(5, self.max_seq_len)
        if len(full_seq) <= min_start_pos + 5:
            start_pos = 0
        else:
            start_pos = np.random.randint(0, len(full_seq) - min_start_pos)
        
        self.current_seq = full_seq[start_pos:]
        self.current_pos = 0
        self.done = False
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        """
        
        
        Returns:
            obs: {
                'item_embeddings': [seq_len, embedding_dim]
                'kg_embeddings': [seq_len, kg_embedding_dim]
                'length': 
                'user_id': ID
            }
        """
        # 
        history = self.current_seq[:self.current_pos]
        
        if len(history) == 0:
            # :
            history = [0]
        
        # padding
        if len(history) > self.max_seq_len:
            history = history[-self.max_seq_len:]
        
        actual_length = len(history)
        
        # embedding
        item_ids = torch.tensor(history, device=self.device)
        item_emb = self.item_embeddings[item_ids]
        kg_emb = self.kg_embeddings[item_ids]
        
        # Padding
        if actual_length < self.max_seq_len:
            pad_len = self.max_seq_len - actual_length
            item_pad = torch.zeros(pad_len, self.embedding_dim, device=self.device)
            kg_pad = torch.zeros(pad_len, self.kg_embedding_dim, device=self.device)
            
            item_emb = torch.cat([item_emb, item_pad], dim=0)
            kg_emb = torch.cat([kg_emb, kg_pad], dim=0)
        
        return {
            'item_embeddings': item_emb.cpu().numpy(),
            'kg_embeddings': kg_emb.cpu().numpy(),
            'length': actual_length,
            'user_id': self.current_user
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        
        
        Args:
            action: ID
        
        Returns:
            observation: 
            reward: 
            done: 
            info: 
        """
        if self.done:
            raise RuntimeError("Episode,reset()")
        
        # ground truth
        if self.current_pos >= len(self.current_seq):
            self.done = True
            return self._get_observation(), 0.0, True, {}
        
        ground_truth_item = self.current_seq[self.current_pos]
        
        # 
        reward = 1.0 if action == ground_truth_item else 0.0
        
        #  (ground truth)
        self.current_pos += 1
        
        # 
        if self.current_pos >= len(self.current_seq) or self.current_pos >= 20:
            self.done = True
        
        next_obs = self._get_observation()
        
        info = {
            'ground_truth_item': ground_truth_item,
            'predicted_item': action,
            'hit': reward > 0
        }
        
        return next_obs, reward, self.done, info
    
    def render(self, mode='human'):
        """"""
        pass
    
    def seed(self, seed=None):
        """"""
        np.random.seed(seed)


class VectorizedRecEnv:
    """
    
    
    """
    
    def __init__(self, env_fns: List, **kwargs):
        """
        Args:
            env_fns: 
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, id: Optional[np.ndarray] = None):
        """"""
        if id is None:
            id = np.arange(self.num_envs)
        
        obs_list = []
        for i in id:
            obs = self.envs[i].reset()
            obs_list.append(obs)
        
        return self._stack_obs(obs_list)
    
    def step(self, actions: np.ndarray, id: Optional[np.ndarray] = None):
        """"""
        if id is None:
            id = np.arange(self.num_envs)
        
        obs_list, rew_list, done_list, info_list = [], [], [], []
        
        for i, action in zip(id, actions):
            obs, rew, done, info = self.envs[i].step(action)
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        
        return (
            self._stack_obs(obs_list),
            np.array(rew_list),
            np.array(done_list),
            np.array(info_list)
        )
    
    def _stack_obs(self, obs_list: List[Dict]) -> Dict:
        """"""
        keys = obs_list[0].keys()
        stacked = {}
        
        for key in keys:
            if key == 'user_id':
                stacked[key] = np.array([obs[key] for obs in obs_list])
            elif key == 'length':
                stacked[key] = np.array([obs[key] for obs in obs_list])
            else:
                stacked[key] = np.stack([obs[key] for obs in obs_list])
        
        return stacked
    
    def seed(self, seed: Optional[int] = None):
        """"""
        for i, env in enumerate(self.envs):
            env.seed(seed + i if seed is not None else None)
    
    def close(self):
        """"""
        for env in self.envs:
            env.close()


if __name__ == '__main__':
    print("Testing Recommendation Environment...")
    
    # 
    num_users = 10
    num_items = 100
    
    user_sequences = {
        i: list(np.random.randint(0, num_items, size=20))
        for i in range(num_users)
    }
    
    item_embeddings = torch.randn(num_items, 64)
    kg_embeddings = torch.randn(num_items, 128)
    
    # 
    env = RecommendationEnv(
        user_sequences=user_sequences,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=10,
        device='cpu'
    )
    
    # reset
    obs = env.reset()
    print(f" Observation keys: {obs.keys()}")
    print(f" Item embeddings shape: {obs['item_embeddings'].shape}")
    print(f" Length: {obs['length']}")
    
    # step
    action = np.random.randint(0, num_items)
    next_obs, reward, done, info = env.step(action)
    print(f" Reward: {reward}")
    print(f" Done: {done}")
    print(f" Info: {info}")
    
    print("\n All tests passed!")