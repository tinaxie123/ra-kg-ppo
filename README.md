# RA-KG-PPO: Retrieval-Augmented Knowledge Graph PPO for Sequential Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official implementation** of "RA-KG-PPO: Retrieval-Augmented Knowledge Graph PPO for Sequential Recommendation"

---

## 🌟 Highlights

- **Knowledge Enhancement**: Integrates knowledge graph embeddings for rich semantic understanding
- **Retrieval Augmentation**: LSH-based candidate generation for scalable action space
- **Policy Optimization**: PPO with GAE for stable and sample-efficient learning
- **State-of-the-Art Results**: Outperforms 10+ baselines by 6-7% on three datasets
- **Production-Ready**: Real-time inference (<10ms) with efficient LSH retrieval

---

## 📊 Main Results

| Dataset | Metric | Best Baseline | RA-KG-PPO | Improvement |
|---------|--------|---------------|-----------|-------------|
| Amazon-Book | Recall@20 | 0.0798 | **0.0856** | **+7.3%** |
| Amazon-Book | NDCG@20 | 0.0603 | **0.0645** | **+7.0%** |
| Last-FM | Recall@20 | 0.0891 | **0.0945** | **+6.1%** |
| Yelp2018 | Recall@20 | 0.0856 | **0.0912** | **+6.5%** |

See [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for complete results.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ra_kg_ppo.git
cd ra_kg_ppo

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download and prepare data
python scripts/prepare_data.py --dataset amazon-book
```

**Data Source**: [KGAT Repository](https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data)

### Training

**Quick Test** (10 minutes):
```bash
python train.py --dataset amazon-book --total-timesteps 10000
```

**Full Training** (recommended, reproduces paper results):
```bash
python train.py --dataset amazon-book --total-timesteps 1000000 --seed 42
```

---

## 📂 Project Structure

```
ra_kg_ppo/
├── models/              # Neural network models
│   └── policy_net.py    # Actor-Critic policy network
├── algorithms/          # RL algorithms
│   ├── trainer.py       # PPO trainer
│   └── rollout_buffer.py
├── retrieval/           # Retrieval components
│   └── lsh.py          # LSH-based candidate generation
├── envs/               # RL environments
│   └── rec_env.py      # Recommendation environment
├── data/               # Data loading
│   └── dataset.py      # KGAT data loader
├── utils/              # Utilities
│   └── metrics.py      # Evaluation metrics
├── scripts/            # Helper scripts
│   └── prepare_data.py
├── experiments/        # Experiment scripts
│   ├── ablation.py     # Ablation study (No-KG, No-LSH, No-PPO)
│   └── baselines.py    # Baseline comparisons (BPR, GRU4Rec, SASRec)
├── train.py            # Main training script
├── EXPERIMENTAL_RESULTS.md     # Complete experimental results
└── README.md           # This file
```

---

## 🎯 Key Features

### 1. Knowledge Graph Enhancement

- **Pre-trained KG Embeddings**: TransE embeddings capture semantic relationships
- **Entity-Item Alignment**: Maps items to KG entities
- **Rich Side Information**: Genres, authors, attributes, etc.

### 2. Retrieval Augmentation

- **LSH-based Candidate Generation**: Reduces action space from O(N) to O(K)
- **Policy-Conditioned Queries**: Adapts retrieval to current state
- **Efficient & Scalable**: Handles 100K+ item catalogs

### 3. Policy Optimization

- **PPO Algorithm**: Stable and sample-efficient RL
- **GAE**: Better credit assignment
- **Multi-Epoch Updates**: Improves sample efficiency

---

## 🔧 Configuration

### Model Configuration

```python
{
    "item_emb_dim": 128,        # Item embedding dimension
    "kg_emb_dim": 256,          # KG embedding dimension
    "hidden_dim": 256,          # Hidden state dimension
    "num_layers": 3,            # GRU layers
    "candidate_size": 200,      # Number of candidates
    "lsh_hash_bits": 10,        # LSH hash bits
    "lsh_tables": 8             # LSH tables
}
```

### Training Configuration

```python
{
    "learning_rate": 3e-4,
    "batch_size": 512,
    "n_steps": 4096,
    "n_epochs": 15,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2
}
```

See [configuration details](docs/) for more options.

---

## 📈 Performance

### Comparison with Baselines

![Main Results](figures/main_results.png)

Our method consistently outperforms:
- **Collaborative Filtering**: BPR, NCF
- **Sequential Models**: GRU4Rec, SASRec, BERT4Rec
- **Knowledge-Enhanced**: KGAT, KGIN
- **Reinforcement Learning**: DRR, TPGR, UNICORN

### Efficiency Analysis

| Metric | Value |
|--------|-------|
| Training Time | 4.2 hours (RTX 5090) |
| Inference Latency | 5.3 ms/user |
| Model Size | 4.7M parameters |
| GPU Memory | 5.8 GB |

---

## 📚 Documentation

- [**EXPERIMENTAL_RESULTS.md**](EXPERIMENTAL_RESULTS.md) - Complete experimental results and analysis
- [**docs/DATA_LOADING.md**](docs/DATA_LOADING.md) - Data preparation guide
- [**data/README.md**](data/README.md) - Dataset information

---

## 🎓 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rakgppo2025,
  title={RA-KG-PPO: Retrieval-Augmented Knowledge Graph PPO for Sequential Recommendation},
  author={[Haotong Xie]},
  booktitle={[]},
  year={2025}
}
```

---

## 📊 Reproducibility

### Main Experiment

```bash
# Reproduce paper results
python train.py --dataset amazon-book --total-timesteps 1000000 --seed 42
```

### Ablation Study

Test the contribution of each component:

```bash
# Full model (all components)
python experiments/ablation.py --dataset amazon-book --variant full

# Without knowledge graph embeddings
python experiments/ablation.py --dataset amazon-book --variant no-kg

# Without LSH retrieval (random sampling instead)
python experiments/ablation.py --dataset amazon-book --variant no-lsh

# Without PPO (simple policy gradient instead)
python experiments/ablation.py --dataset amazon-book --variant no-ppo
```

### Baseline Comparison

Compare with existing methods:

```bash
# Run all baselines
python experiments/baselines.py --dataset amazon-book --method all

# Run specific baseline
python experiments/baselines.py --dataset amazon-book --method bpr      # BPR
python experiments/baselines.py --dataset amazon-book --method gru4rec  # GRU4Rec
python experiments/baselines.py --dataset amazon-book --method pop      # Popularity
```

**Settings:**
- **Random Seeds**: All experiments use seed=42 for reproducibility
- **Statistical Testing**: Results reported with standard deviation over 5 independent runs
- **Hardware**: Experiments run on NVIDIA RTX 5090 (24GB) or CPU

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- KGAT dataset: [Link](https://github.com/xiangwang1223/knowledge_graph_attention_network)
- PyTorch: [Link](https://pytorch.org/)
- Tianshou RL library: [Link](https://github.com/thu-ml/tianshou)

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [18558718333@163.com]

---

## 🔖 Related Projects

- [KGAT](https://github.com/xiangwang1223/knowledge_graph_attention_network)
- [SASRec](https://github.com/kang205/SASRec)
- [KGIN](https://github.com/huangjianxian/KGIN)

---

**Last Updated**: December 2024
**Status**: Active Development
