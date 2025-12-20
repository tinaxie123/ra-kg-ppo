
import torch
import torch.nn as nn
from typing import Tuple, Optional


class SequenceEncoder(nn.Module):
    def __init__(self, item_embedding_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=item_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, item_embeddings: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
      
        batch_size = item_embeddings.shape[0]
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                item_embeddings, lengths.cpu(), 
                batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.gru(packed)
        else:
            outputs, hidden = self.gru(item_embeddings)
        final_hidden = hidden[-1]  
        
        return self.layer_norm(final_hidden)


class ActorNetwork(nn.Module):
   
    
    def __init__(self, item_embedding_dim: int, hidden_dim: int,
                 kg_embedding_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.encoder = SequenceEncoder(
            item_embedding_dim=item_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    def forward(self, item_embeddings: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(item_embeddings, lengths)
    
    def compute_action_logits(self, query_vectors: torch.Tensor,
                              candidate_embeddings: torch.Tensor) -> torch.Tensor:
        logits = torch.bmm(
            query_vectors.unsqueeze(1),
            candidate_embeddings.transpose(1, 2)
        ).squeeze(1)
        
        return logits


class CriticNetwork(nn.Module):
    
    
    def __init__(self, item_embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2):
        super().__init__()
        self.encoder = SequenceEncoder(
            item_embedding_dim=item_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, item_embeddings: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_state = self.encoder(item_embeddings, lengths)
        values = self.value_head(hidden_state)
        
        return values


class RAPolicyValueNet(nn.Module):
    
    
    def __init__(self, item_embedding_dim: int, hidden_dim: int,
                 kg_embedding_dim: int, num_layers: int = 2,
                 shared_encoder: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.shared_encoder = shared_encoder
        
        if shared_encoder:
            self.encoder = SequenceEncoder(
                item_embedding_dim=item_embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            self.actor = ActorNetwork(
                item_embedding_dim=item_embedding_dim,
                hidden_dim=hidden_dim,
                kg_embedding_dim=kg_embedding_dim,
                num_layers=num_layers
            )
            self.actor.encoder = self.encoder
            self.critic = CriticNetwork(
                item_embedding_dim=item_embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            self.critic.encoder = self.encoder
        else:
            self.actor = ActorNetwork(
                item_embedding_dim, hidden_dim, 
                kg_embedding_dim, num_layers
            )
            self.critic = CriticNetwork(
                item_embedding_dim, hidden_dim, num_layers
            )
    
    def get_hidden_state(self, item_embeddings: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.shared_encoder:
            return self.encoder(item_embeddings, lengths)
        else:
            return self.actor(item_embeddings, lengths)
    
    def evaluate_actions(self, item_embeddings: torch.Tensor,
                        query_vectors: torch.Tensor,
                        candidate_embeddings: torch.Tensor,
                        actions: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None):
      
        logits = self.actor.compute_action_logits(
            query_vectors, candidate_embeddings
        )
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(item_embeddings, lengths).squeeze(-1)
        
        return log_probs, values, entropy
if __name__ == '__main__':
    batch_size = 4
    seq_len = 10
    item_emb_dim = 64
    hidden_dim = 128
    kg_emb_dim = 128
    candidate_size = 100
    
    print("Testing Policy-Value Network...")
    net = RAPolicyValueNet(
        item_embedding_dim=item_emb_dim,
        hidden_dim=hidden_dim,
        kg_embedding_dim=kg_emb_dim,
        num_layers=2,
        shared_encoder=True
    )
    #
    item_embs = torch.randn(batch_size, seq_len, item_emb_dim)
    lengths = torch.randint(5, seq_len, (batch_size,))
    hidden = net.get_hidden_state(item_embs, lengths)
    print(f" Hidden state shape: {hidden.shape}")
    query_vectors = torch.randn(batch_size, kg_emb_dim)
    candidate_embs = torch.randn(batch_size, candidate_size, kg_emb_dim)
    actions = torch.randint(0, candidate_size, (batch_size,))
    
    log_probs, values, entropy = net.evaluate_actions(
        item_embs, query_vectors, candidate_embs, actions, lengths
    )
    
    print(f" Log probs shape: {log_probs.shape}")
    print(f" Values shape: {values.shape}")
    print(f" Entropy shape: {entropy.shape}")
    
    print("\n All tests passed!")