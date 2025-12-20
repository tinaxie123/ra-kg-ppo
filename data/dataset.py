

import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from collections import defaultdict


def load_kgat_data(dataset_name: str, data_path: str = './data/') -> Dict:
   
    dataset_path = os.path.join(data_path, dataset_name)
    
    
    required_files = ['train.txt', 'test.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(dataset_path, file)):
            raise FileNotFoundError(
                f"[ERROR] Missing {file}!\n"
                f"Please download KGAT data from:\n"
                f"https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data\n"
                f"And extract to: {dataset_path}/"
            )
    
   
    print("1. Loading user-item interactions...")
    train_data = load_rating_file(os.path.join(dataset_path, 'train.txt'))
    test_data = load_rating_file(os.path.join(dataset_path, 'test.txt'))
    
    print(f"   [OK] Train users: {len(train_data)}")
    print(f"   [OK] Test users: {len(test_data)}")

    n_users = max(max(train_data.keys()), max(test_data.keys())) + 1
    n_items = max(
        max([max(items) for items in train_data.values()]),
        max([max(items) for items in test_data.values()])
    ) + 1

    print("\n2. Loading Knowledge Graph...")
    kg_file = os.path.join(dataset_path, 'kg_final.txt')

    if os.path.isfile(kg_file):
        print(f"   Loading from {kg_file}...")
        kg_data = load_kg(kg_file)
        print(f"   [OK] Entities: {kg_data['n_entities']}")
        print(f"   [OK] Relations: {kg_data['n_relations']}")
        print(f"   [OK] Triples: {len(kg_data['triples'])}")
    else:
        print("   [WARN] kg_final.txt not found or is not a file")
        print("   Creating synthetic KG data based on item interactions...")
        kg_data = create_synthetic_kg(train_data, n_items)
        print(f"   [OK] Entities: {kg_data['n_entities']} (synthetic)")
        print(f"   [OK] Relations: {kg_data['n_relations']} (synthetic)")
        print(f"   [OK] Triples: {len(kg_data['triples'])} (synthetic)")

    print(f"\n3. Dataset Statistics:")
    print(f"   [OK] Users: {n_users}")
    print(f"   [OK] Items: {n_items}")
 
    print(f"\n4. Initializing embeddings...")

    item_emb_file = os.path.join(dataset_path, 'item_embeddings.npy')
    kg_emb_file = os.path.join(dataset_path, 'kg_embeddings.npy')

    if os.path.exists(item_emb_file) and os.path.exists(kg_emb_file):
        print("   Loading cached embeddings...")
        item_embeddings = np.load(item_emb_file)
        kg_embeddings = np.load(kg_emb_file)
        print(f"   [OK] Loaded item embeddings: {item_embeddings.shape}")
        print(f"   [OK] Loaded KG embeddings: {kg_embeddings.shape}")
    else:
        print("   Initializing new embeddings...")

        item_embeddings = xavier_init(n_items, 64)

        if len(kg_data['triples']) > 0:
            print("   Training TransE embeddings on KG...")
            kg_embeddings = train_transe_embeddings(
                kg_data, embedding_dim=128, epochs=50, lr=0.01
            )
        else:
            print("   Using Xavier initialization for KG embeddings...")
            kg_embeddings = xavier_init(kg_data['n_entities'], 128)

        np.save(item_emb_file, item_embeddings)
        np.save(kg_emb_file, kg_embeddings)
        print(f"   [OK] Saved to {dataset_path}/")
    
   
    print(f" Dataset loaded successfully!")
   
    return {
        'train_data': train_data,
        'test_data': test_data,
        'n_users': n_users,
        'n_items': n_items,
        'item_embeddings': torch.FloatTensor(item_embeddings),
        'kg_embeddings': torch.FloatTensor(kg_embeddings),
        'kg_data': kg_data
    }


def load_rating_file(file_path: str) -> Dict[int, List[int]]:
   
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    user_dict = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            user_dict[user_id] = item_ids
    
    return user_dict


def load_kg(file_path: str) -> Dict:
   
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    triples = []
    entities = set()
    relations = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            
            h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            
            triples.append((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)
    
    return {
        'triples': triples,
        'n_entities': len(entities),
        'n_relations': len(relations),
        'entity_list': sorted(list(entities)),
        'relation_list': sorted(list(relations))
    }


def xavier_init(n_items: int, dim: int) -> np.ndarray:
    
    scale = np.sqrt(6.0 / (n_items + dim))
    embeddings = np.random.uniform(-scale, scale, (n_items, dim))
    return embeddings.astype(np.float32)


def create_user_sequences(user_dict: Dict[int, List[int]],
                          min_seq_len: int = 5) -> Dict[int, List[int]]:
    """
    
    """
    sequences = {}

    for user_id, items in user_dict.items():
        if len(items) >= min_seq_len:
            sequences[user_id] = items

    return sequences


def create_synthetic_kg(train_data: Dict[int, List[int]],
                       n_items: int) -> Dict:
    """
    

    
    - "similar_to"
    - TF-IDF

    Args:
        train_data: 
        n_items: 

    Returns:
        kg_data: 
    """
    from collections import Counter

    print("   Building co-occurrence graph...")

    
    cooccurrence = defaultdict(Counter)

    for user_id, items in train_data.items():
       
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                if item1 != item2:
                    cooccurrence[item1][item2] += 1
                    cooccurrence[item2][item1] += 1

    triples = []
    relation_types = {
        'co_purchased': 0,
        'highly_related': 1,
        'weakly_related': 2
    }

    for item1, related_items in cooccurrence.items():
      
        top_related = related_items.most_common(20)

        for item2, count in top_related:
          
            if count >= 10:
                relation = relation_types['highly_related']
            elif count >= 5:
                relation = relation_types['co_purchased']
            else:
                relation = relation_types['weakly_related']

            triples.append((item1, relation, item2))

    entities = set(range(n_items))

    print(f"   Created {len(triples)} triples from co-occurrence")

    return {
        'triples': triples,
        'n_entities': len(entities),
        'n_relations': len(relation_types),
        'entity_list': sorted(list(entities)),
        'relation_list': list(relation_types.values())
    }


def train_transe_embeddings(kg_data: Dict, embedding_dim: int = 128,
                            epochs: int = 50, lr: float = 0.01,
                            margin: float = 1.0, batch_size: int = 1024) -> np.ndarray:
    """
    TransE

    TransEh + r ≈ t
    max(0, d(h+r, t) - d(h'+r, t') + margin)

    Args:
        kg_data: 
        embedding_dim: 
        epochs: 
        lr: 
        margin: 
        batch_size: 

    Returns:
        entity_embeddings:  [n_entities, embedding_dim]
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    n_entities = kg_data['n_entities']
    n_relations = kg_data['n_relations']
    triples = np.array(kg_data['triples'])

    entity_emb = nn.Embedding(n_entities, embedding_dim)
    relation_emb = nn.Embedding(n_relations, embedding_dim)

    nn.init.xavier_uniform_(entity_emb.weight.data)
    nn.init.xavier_uniform_(relation_emb.weight.data)

    entity_emb.weight.data = torch.nn.functional.normalize(
        entity_emb.weight.data, p=2, dim=1
    )

    optimizer = optim.Adam(
        list(entity_emb.parameters()) + list(relation_emb.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
       
        np.random.shuffle(triples)
        total_loss = 0

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]

            h = torch.LongTensor(batch[:, 0])
            r = torch.LongTensor(batch[:, 1])
            t = torch.LongTensor(batch[:, 2])

          
            neg_batch = batch.copy()
            for j in range(len(neg_batch)):
                if np.random.rand() < 0.5:
                   
                    neg_batch[j, 0] = np.random.randint(0, n_entities)
                else:
                   
                    neg_batch[j, 2] = np.random.randint(0, n_entities)

            h_neg = torch.LongTensor(neg_batch[:, 0])
            t_neg = torch.LongTensor(neg_batch[:, 2])

            h_emb = entity_emb(h)
            r_emb = relation_emb(r)
            t_emb = entity_emb(t)

            h_neg_emb = entity_emb(h_neg)
            t_neg_emb = entity_emb(t_neg)

            pos_score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
            neg_score = torch.norm(h_neg_emb + r_emb - t_neg_emb, p=2, dim=1)

            loss = torch.mean(torch.relu(pos_score - neg_score + margin))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          
            with torch.no_grad():
                entity_emb.weight.data = torch.nn.functional.normalize(
                    entity_emb.weight.data, p=2, dim=1
                )

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return entity_emb.weight.data.numpy()


def quick_check_data(data_path: str = './data/'):
    datasets = ['amazon-book', 'last-fm', 'yelp2018']
    required_files = ['train.txt', 'test.txt', 'kg_final.txt']
    print("Checking data availability...")
    
    all_ok = True
    
    for dataset in datasets:
        print(f"Checking {dataset}:")
        dataset_path = os.path.join(data_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"  Directory not found: {dataset_path}")
            all_ok = False
            continue
        
        for file in required_files:
            file_path = os.path.join(dataset_path, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   {file} ({size_mb:.2f} MB)")
            else:
                print(f"   Missing: {file}")
                all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("[OK] All datasets are ready!")
    else:
        print("[ERROR] Some datasets are missing.")
        print("\nPlease download from:")
        print("https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data")
    print("="*60 + "\n")
    
    return all_ok


if __name__ == '__main__':
    quick_check_data('./data/')
    try:
        data = load_kgat_data('amazon-book', './data/')
        
        print("\nData loaded successfully!")
        print(f"Item embeddings shape: {data['item_embeddings'].shape}")
        print(f"KG embeddings shape: {data['kg_embeddings'].shape}")
        print(f"Train users: {len(data['train_data'])}")
        print(f"Test users: {len(data['test_data'])}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease download the dataset first!")