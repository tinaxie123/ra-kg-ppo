# RA-KG-PPO 数据加载与KG嵌入初始化

本文档介绍RA-KG-PPO框架中的KGAT格式数据加载和知识图谱嵌入初始化功能。

## 功能概述

### KGAT格式数据加载

支持从KGAT格式数据集加载用户-物品交互数据和知识图谱：

**用户-物品交互**: 从 `train.txt` 和 `test.txt` 加载
**知识图谱**: 从 `kg_final.txt` 加载（可选）
**合成KG**: 当KG文件不存在时，基于物品共现关系自动构建合成知识图谱

### KG嵌入初始化

提供两种KG嵌入初始化方法：

**TransE预训练**: 使用TransE算法在知识图谱上预训练实体嵌入
**Xavier初始化**: 当无KG数据时，使用Xavier均匀分布初始化

## 使用方法

### 基础用法

```python
from data.dataset import load_kgat_data

data = load_kgat_data('amazon-book', './data/')

# 返回的数据包含:
# train_data: 训练集用户-物品交互
# test_data: 测试集用户-物品交互
# n_users: 用户数量
# n_items: 物品数量
# item_embeddings: 物品嵌入 [n_items, 64]
# kg_embeddings: KG实体嵌入 [n_entities, 128]
# kg_data: 知识图谱数据
```

### 使用数据预处理脚本

```bash
python scripts/prepare_data.py --dataset amazon-book
python scripts/prepare_data.py --dataset amazon-book --force_rebuild
python scripts/prepare_data.py --dataset amazon-book --transe_epochs 50
```

### 用户-物品交互文件格式 (train.txt / test.txt)

每行表示一个用户及其交互过的物品：

```
user_id item_id1 item_id2 item_id3 ...
```

示例：
```
0 0 1 2 3 4 5
1 32 33 34 35
2 36 37 38 39 40
```

### 知识图谱文件格式 (kg_final.txt)

每行表示一个三元组 (头实体, 关系, 尾实体)：

```
head_entity_id relation_id tail_entity_id
```

示例：
```
0 0 100
0 1 200
100 2 300
```

## 合成知识图谱

当 `kg_final.txt` 不存在时，系统会自动基于用户-物品交互构建合成KG：

**实体**: 所有物品
**关系类型**:
 `highly_related` (共现次数 >= 10)
 `co_purchased` (共现次数 >= 5)
 `weakly_related` (共现次数 < 5)
  **三元组**: 基于物品在用户序列中的共现关系生成

## TransE嵌入训练

TransE (Translating Embeddings) ：


**训练过程**:
1. 随机初始化实体和关系嵌入
2. 对每个正样本三元组，随机替换头或尾实体生成负样本
3. 优化损失函数: `max(0, ||h+r-t|| - ||h'+r-t'|| + margin)`
4. L2归一化实体嵌入

**参数说明**:
- `embedding_dim`: 嵌入维度 (默认: 128)
- `epochs`: 训练轮数 (默认: 50)
- `lr`: 学习率 (默认: 0.01)
- `margin`: 边界值 (默认: 1.0)
- `batch_size`: 批大小 (默认: 1024)

## 数据统计示例


```
【用户-物品交互】
  训练用户数: 70679
  测试用户数: 70591
  总用户数: 70679
  总物品数: 24915
  训练交互数: 652514
  测试交互数: 193920
  平均每用户交互数: 9.23

【知识图谱】
  实体数: 24915
  关系数: 3
  三元组数: 495866
  KG密度: 0.026627%

【嵌入】
  物品嵌入维度: [24915, 64]
  KG嵌入维度: [24915, 128]
```


### `load_kgat_data(dataset_name, data_path)`

加载KGAT格式数据集。

**参数**:
`dataset_name` (str): 数据集名称，如 'amazon-book'
`data_path` (str): 数据根目录路径，默认 './data/'

**返回**:`dict`: 包含所有数据和嵌入的字典

### `create_synthetic_kg(train_data, n_items)`

基于用户交互创建合成知识图谱。

**参数**:
`train_data` (dict): 用户-物品交互数据
`n_items` (int): 物品总数

**返回**:
`dict`: 知识图谱数据

### `train_transe_embeddings(kg_data, embedding_dim, epochs, lr, margin, batch_size)`

使用TransE算法训练KG嵌入。

**参数**:
`kg_data` (dict): 知识图谱数据
`embedding_dim` (int): 嵌入维度
`epochs` (int): 训练轮数
`lr` (float): 学习率
`margin` (float): 边界值
`batch_size` (int): 批大小

**返回**:
`np.ndarray`: 实体嵌入矩阵 [n_entities, embedding_dim]

## 文件结构

```
ra_kg_ppo/
├── data/
│   ├── dataset.py              
│   └── amazon-book/
│       ├── train.txt           
│       ├── test.txt           
│       ├── kg_final.txt        
│       ├── item_embeddings.npy
│       └── kg_embeddings.npy   
├── scripts/
│   └── prepare_data.py         
└── docs/
    └── DATA_LOADING.md         
```

## 相关论文

TransE: [Translating Embeddings for Modeling Multi-relational Data (NIPS 2013)](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
KGAT: [KGAT: Knowledge Graph Attention Network for Recommendation (KDD 2019)](https://arxiv.org/abs/1905.07854)
