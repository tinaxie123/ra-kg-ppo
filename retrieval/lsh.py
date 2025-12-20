
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class LSHIndex:
    
    def __init__(self, embedding_dim: int, num_hash_bits: int = 8, 
                 num_tables: int = 4, device: str = 'cpu'):
      
        self.embedding_dim = embedding_dim
        self.num_hash_bits = num_hash_bits
        self.num_tables = num_tables
        self.device = device
        self.projection_matrices = []
        for _ in range(num_tables):
            W = torch.randn(embedding_dim, num_hash_bits, device=device)
            W = W / torch.norm(W, dim=0, keepdim=True)
            self.projection_matrices.append(W)
        self.hash_tables: List[Dict[str, List[int]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]
        
        self.item_embeddings = None 
        self.num_items = 0
    
    def _compute_hash_code(self, embeddings: torch.Tensor, 
                          table_idx: int) -> List[str]:
      
        W = self.projection_matrices[table_idx]
        projections = embeddings @ W
        hash_codes = (projections > 0).int()
        hash_strings = []
        for code in hash_codes:
            hash_str = ''.join(code.cpu().numpy().astype(str))
            hash_strings.append(hash_str)
        
        return hash_strings
    
    def build_index(self, item_embeddings: torch.Tensor):
       
        self.item_embeddings = item_embeddings.to(self.device)
        self.num_items = item_embeddings.shape[0]
        
        print(f"Building LSH index for {self.num_items} items...")
        for table_idx in range(self.num_tables):
            hash_codes = self._compute_hash_code(item_embeddings, table_idx)
            
            for item_id, hash_code in enumerate(hash_codes):
                self.hash_tables[table_idx][hash_code].append(item_id)
        avg_bucket_size = np.mean([
            len(bucket) for table in self.hash_tables 
            for bucket in table.values()
        ])
        print(f"Index built. Avg bucket size: {avg_bucket_size:.2f}")
    
    def query(self, query_embeddings: torch.Tensor, 
              top_k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
       
        batch_size = query_embeddings.shape[0]
        all_candidates = []
        
        for i in range(batch_size):
            query = query_embeddings[i:i+1]
            candidates_set = set()
            
            for table_idx in range(self.num_tables):
                hash_code = self._compute_hash_code(query, table_idx)[0]
                candidates_set.update(
                    self.hash_tables[table_idx].get(hash_code, [])
                )
            
            candidates_list = list(candidates_set)
            if len(candidates_list) < top_k:
                remaining = top_k - len(candidates_list)
                all_items = set(range(self.num_items))
                extra = list(all_items - candidates_set)
                np.random.shuffle(extra)
                candidates_list.extend(extra[:remaining])
            if len(candidates_list) > top_k:
                cand_emb = self.item_embeddings[candidates_list]
                scores = (query @ cand_emb.T).squeeze(0)
                top_indices = torch.topk(scores, top_k).indices
                candidates_list = [candidates_list[idx] for idx in top_indices]
            
            all_candidates.append(candidates_list)
        
        candidate_ids = torch.tensor(all_candidates, device=self.device)
        candidate_embeddings = torch.stack([
            self.item_embeddings[cands] for cands in all_candidates
        ])
        
        return candidate_ids, candidate_embeddings
    
    def get_retrieval_recall(self, query_embeddings: torch.Tensor,
                            ground_truth_ids: torch.Tensor,
                            top_k: int = 100) -> float:
        candidate_ids, _ = self.query(query_embeddings, top_k)
        
        hits = 0
        for i, gt_id in enumerate(ground_truth_ids):
            if gt_id.item() in candidate_ids[i]:
                hits += 1
        
        return hits / len(ground_truth_ids)


class CandidateGenerator(nn.Module):
    
    def __init__(self, hidden_dim: int, embedding_dim: int,
                 num_hash_bits: int = 8, num_tables: int = 4,
                 candidate_size: int = 100):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.candidate_size = candidate_size
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()  
        )
        self.lsh_index = LSHIndex(
            embedding_dim=embedding_dim,
            num_hash_bits=num_hash_bits,
            num_tables=num_tables
        )
    
    def build_index(self, item_embeddings: torch.Tensor):
        self.lsh_index.build_index(item_embeddings)
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_vectors = self.query_projection(hidden_state)
        candidate_ids, candidate_embeddings = self.lsh_index.query(
            query_vectors, top_k=self.candidate_size
        )
        
        return query_vectors, candidate_ids, candidate_embeddings