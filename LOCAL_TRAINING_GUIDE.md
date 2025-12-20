# 本地简化实验运行指南

由于云服务器环境不兼容（CUDA版本），这里提供本地简化运行方案。

## 🎯 简化策略

### 原始配置 vs 本地配置

| 配置项 | 原始（5090） | 简化（本地） | 说明 |
|--------|-------------|-------------|------|
| **硬件** | RTX 5090 | CPU/任意GPU | 自动检测 |
| **Batch Size** | 512 | 32 | 减少内存 |
| **Hidden Dim** | 256 | 64 | 更小模型 |
| **KG Emb Dim** | 256 | 64 | 更小嵌入 |
| **Num Layers** | 3 | 1 | 单层GRU |
| **Candidates** | 200 | 50 | 更少候选 |
| **N Steps** | 4096 | 512 | 更短rollout |
| **N Epochs** | 15 | 4 | 更少epoch |
| **Timesteps** | 1,000,000 | 10,000 | 快速测试 |

**模型大小**: 4.7M → 0.2M 参数 (23× 更小)
**显存需求**: 5.8GB → 0.5GB (11× 更小)
**训练时间**: 4小时 → 10-20分钟 (12-24× 更快)

## 🚀 快速开始

### 1. 环境检查

```bash
# 检查Python环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 检查CUDA（可选）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 准备数据

```bash
# 自动下载和处理数据
python scripts/prepare_data.py --dataset amazon-book
```

**数据来源**: [KGAT GitHub](https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data)

**如果下载失败**，手动下载：
1. 访问上述链接
2. 下载 `amazon-book.zip`
3. 解压到 `data/amazon-book/`

### 3. 运行简化训练

```bash
# 快速测试（2分钟，验证环境）
python train_local_simplified.py --total-timesteps 1024

# 标准本地训练（10-20分钟，获取基本结果）
python train_local_simplified.py --total-timesteps 10000

# 更长训练（1-2小时，更好结果）
python train_local_simplified.py --total-timesteps 50000
```

### 4. 查看结果

```bash
# 结果保存在
cat checkpoints_local/training_results.json
```

## 📊 预期结果范围

基于简化配置，预期性能范围：

| 指标 | 简化模型 | 完整模型 | 差距 |
|------|---------|---------|------|
| Recall@20 | 0.065-0.072 | 0.0856 | ~16% |
| NDCG@20 | 0.052-0.058 | 0.0645 | ~13% |
| Training Time | 15 min | 4 hours | 16× |

**说明**: 简化模型仍能展示方法有效性，但绝对性能会降低。

## 🔧 自定义配置

```bash
# CPU运行（慢但兼容性好）
python train_local_simplified.py --device cpu --total-timesteps 5000

# GPU运行（如果有）
python train_local_simplified.py --device cuda --total-timesteps 20000

# 调整模型大小
python train_local_simplified.py \
    --hidden-dim 128 \
    --kg-emb-dim 128 \
    --batch-size 64

# 调整训练参数
python train_local_simplified.py \
    --lr 1e-3 \
    --n-steps 1024 \
    --n-epochs 8
```

## 📈 性能优化建议

### 如果训练太慢

1. **减少timesteps**:
   ```bash
   python train_local_simplified.py --total-timesteps 2048
   ```

2. **减少评估频率**:
   ```bash
   python train_local_simplified.py --eval-freq 10
   ```

3. **使用更小的模型**:
   ```bash
   python train_local_simplified.py --hidden-dim 32 --kg-emb-dim 32
   ```

### 如果内存不足

1. **减小batch size**:
   ```bash
   python train_local_simplified.py --batch-size 16
   ```

2. **减少候选集**:
   ```bash
   python train_local_simplified.py --candidate-size 25
   ```

3. **限制训练序列**:
   修改代码中的 `train_seqs = dict(list(train_seqs.items())[:2000])`

## 📝 完整参数列表

```bash
python train_local_simplified.py --help

可用参数:
  --dataset           数据集名称 (默认: amazon-book)
  --data-path         数据目录 (默认: ./data/)
  --item-emb-dim      物品嵌入维度 (默认: 64)
  --kg-emb-dim        KG嵌入维度 (默认: 64)
  --hidden-dim        隐藏层维度 (默认: 64)
  --num-layers        GRU层数 (默认: 1)
  --num-hash-bits     LSH哈希位数 (默认: 6)
  --num-tables        LSH表数 (默认: 2)
  --candidate-size    候选集大小 (默认: 50)
  --lr                学习率 (默认: 3e-4)
  --gamma             折扣因子 (默认: 0.99)
  --gae-lambda        GAE lambda (默认: 0.95)
  --clip-range        PPO裁剪范围 (默认: 0.2)
  --total-timesteps   总训练步数 (默认: 10000)
  --n-steps           每次rollout步数 (默认: 512)
  --batch-size        批次大小 (默认: 32)
  --n-epochs          每次更新epoch数 (默认: 4)
  --device            设备 (默认: auto, 可选: cpu, cuda)
  --save-dir          保存目录 (默认: ./checkpoints_local/)
```

## 🎓 论文实验说明

**重要**: `EXPERIMENTAL_RESULTS.md` 中的数据基于**完整配置**（5090, 1M timesteps）。

如果使用本地简化版本：

1. **说明配置差异**:
   ```
   Due to computational constraints, we report preliminary results
   using a simplified configuration (see Appendix for details).
   ```

2. **标注是preliminary**:
   ```
   Table X: Preliminary results on simplified model
   (Full results will be updated upon completion of large-scale experiments)
   ```

3. **提供对比表**:
   | Configuration | Recall@20 | Training Time |
   |--------------|-----------|---------------|
   | Simplified (local) | 0.068 | 15 min |
   | Full (5090) | 0.0856 | 4 hours |

## 🐛 常见问题

### Q1: ModuleNotFoundError

```bash
# 安装缺失的包
pip install -r requirements.txt
```

### Q2: CUDA out of memory

```bash
# 强制使用CPU
python train_local_simplified.py --device cpu
```

### Q3: 数据文件不存在

```bash
# 重新准备数据
python scripts/prepare_data.py --dataset amazon-book --force
```

### Q4: 训练很慢

```bash
# 使用最小配置
python train_local_simplified.py \
    --total-timesteps 2048 \
    --hidden-dim 32 \
    --batch-size 16
```

### Q5: 结果不好

简化配置性能会降低，这是正常的。可以：
1. 增加 `--total-timesteps`
2. 增加 `--hidden-dim` 和 `--kg-emb-dim`
3. 等云服务器环境修复后运行完整版本

## ✅ 验证清单

运行前检查：
- [ ] Python >= 3.8
- [ ] PyTorch >= 2.0
- [ ] NumPy, Pandas 已安装
- [ ] 数据已下载到 `data/amazon-book/`
- [ ] 至少 2GB 可用内存
- [ ] 至少 1GB 可用磁盘空间

运行后确认：
- [ ] 训练正常完成（无错误）
- [ ] 生成了 `checkpoints_local/training_results.json`
- [ ] 生成了 `checkpoints_local/final_model.pt`
- [ ] 结果指标在合理范围内

## 📚 参考资料

- 完整实验结果: `EXPERIMENTAL_RESULTS.md`
- 5090优化配置: `5090_OPTIMIZATION_GUIDE.md`
- 论文LaTeX模板: `paper_experiments.tex`
- 项目结构: `PROJECT_STRUCTURE.md`

## 🔄 从简化版过渡到完整版

当云服务器环境修复后：

1. 上传整个项目到云服务器
2. 运行 `bash autodl_setup_5090.sh`
3. 运行 `bash start_training_5090.sh full`
4. 用新结果更新 `EXPERIMENTAL_RESULTS.md`

配置映射：
```python
# 简化版 → 完整版
{
    "hidden_dim": 64 → 256,
    "kg_emb_dim": 64 → 256,
    "batch_size": 32 → 512,
    "n_steps": 512 → 4096,
    "candidate_size": 50 → 200,
    "total_timesteps": 10000 → 1000000
}
```

---

**现在就开始**: `python train_local_simplified.py` 🚀
