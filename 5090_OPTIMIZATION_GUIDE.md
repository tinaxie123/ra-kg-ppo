# 5090 优化配置指南

充分利用 AutoDL 上 5090 显卡的完整指南。

## 快速开始

### 1. 初始化环境

```bash
# 上传代码到 AutoDL
# 在 AutoDL 控制台开启 Jupyter 或终端

# 进入项目目录
cd /root/autodl-tmp/ra_kg_ppo

# 一键配置环境
bash autodl_setup_5090.sh
```

### 2. 快速测试

```bash
# 10分钟快速测试
bash start_training_5090.sh quick
```

### 3. 开始训练

```bash
# 标准训练（1-2小时）
bash start_training_5090.sh medium

# 完整训练（4-8小时）
bash start_training_5090.sh full

# 超长训练（24小时+）
bash start_training_5090.sh ultra
```

## 性能优化详解

### 5090 vs 默认配置对比

| 参数 | 默认配置 | 5090优化 | 提升 |
|------|---------|---------|------|
| Batch Size | 64 | 512-1024 | 8-16x |
| Rollout Steps | 2048 | 4096-8192 | 2-4x |
| Hidden Dim | 128 | 256-512 | 2-4x |
| KG Emb Dim | 128 | 256-512 | 2-4x |
| Candidate Size | 100 | 200-300 | 2-3x |
| 混合精度 | 否 | 是 | 1.5-2x |

### 显存使用估算

**默认配置 (~2GB)**
```bash
python train.py --batch-size 64 --hidden-dim 128
```

**5090 标准配置 (~8-12GB)**
```bash
python train_5090_optimized.py
# batch_size=512, hidden_dim=256
```

**5090 极限配置 (~20-24GB)**
```bash
python train_5090_optimized.py \
    --batch-size 1024 \
    --hidden-dim 512 \
    --kg-emb-dim 512 \
    --n-steps 8192
```

## 实时监控

### GPU 监控

```bash
# 方式1: 实时监控（推荐）
python monitor_5090.py --mode monitor

# 方式2: 查看摘要
python monitor_5090.py --mode summary

# 方式3: nvidia-smi
watch -n 1 nvidia-smi
```

### 训练监控

```bash
# 在另一个终端启动 TensorBoard
tensorboard --logdir tensorboard_logs --host 0.0.0.0 --port 6006

# 在 AutoDL 控制台设置端口映射，然后访问：
# http://localhost:6006
```

## 训练模式详解

### Quick Mode（快速测试）
- **时长**: 10-15分钟
- **显存**: ~6GB
- **用途**: 验证环境和代码

```bash
bash start_training_5090.sh quick
```

### Medium Mode（标准训练）
- **时长**: 1-2小时
- **显存**: ~10GB
- **用途**: 完整训练流程，获得基线结果

```bash
bash start_training_5090.sh medium
```

### Full Mode（完整训练）
- **时长**: 4-8小时
- **显存**: ~16GB
- **用途**: 充分训练，追求更好性能

```bash
bash start_training_5090.sh full
```

### Ultra Mode（极限训练）
- **时长**: 24小时+
- **显存**: ~24GB
- **用途**: 最大模型容量，榨干5090性能

```bash
bash start_training_5090.sh ultra
```

## 自定义配置

### 基本参数

```bash
python train_5090_optimized.py \
    --dataset amazon-book \
    --batch-size 512 \
    --total-timesteps 1000000 \
    --save-dir ./checkpoints_5090/
```

### 模型大小

```bash
# 小模型（快速训练）
python train_5090_optimized.py \
    --hidden-dim 128 \
    --kg-emb-dim 128 \
    --item-emb-dim 64

# 中等模型（推荐）
python train_5090_optimized.py \
    --hidden-dim 256 \
    --kg-emb-dim 256 \
    --item-emb-dim 128

# 大模型（充分利用5090）
python train_5090_optimized.py \
    --hidden-dim 512 \
    --kg-emb-dim 512 \
    --item-emb-dim 256 \
    --num-layers 4
```

### 批次大小优化

```bash
# 根据可用显存选择batch size：

# 16GB显存
--batch-size 512 --n-steps 4096

# 24GB显存
--batch-size 1024 --n-steps 4096

# 32GB显存（如果有更大显存）
--batch-size 1024 --n-steps 8192
```

### 混合精度训练

```bash
# 启用混合精度（默认开启，提速1.5-2x）
python train_5090_optimized.py --use-amp

# FP16（更快，默认）
python train_5090_optimized.py --use-amp --amp-dtype float16

# BF16（更稳定，5090支持）
python train_5090_optimized.py --use-amp --amp-dtype bfloat16
```

## 断点续传

### 保存检查点

```bash
# 自动保存（每10个update）
python train_5090_optimized.py --save-freq 10
```

### 恢复训练

```bash
# 从检查点恢复
python train_5090_optimized.py \
    --resume ./checkpoints_5090/checkpoint_500000.pt
```

## 性能优化技巧

### 1. 充分利用显存

```bash
# 逐步增加batch size直到显存接近满载
python train_5090_optimized.py --batch-size 256  # 测试
python train_5090_optimized.py --batch-size 512  # 测试
python train_5090_optimized.py --batch-size 1024 # 测试
```

### 2. 梯度累积（显存不足时）

```bash
# 等效于更大的batch size，但使用更少显存
python train_5090_optimized.py \
    --batch-size 256 \
    --gradient-accumulation-steps 4  # 等效batch=1024
```

### 3. 数据并行（多GPU）

```bash
# 如果租了多张卡
CUDA_VISIBLE_DEVICES=0,1 python train_5090_optimized.py
```

### 4. 优化数据加载

```bash
# 增加数据加载workers
python train_5090_optimized.py --num-workers 4
```

## 常见问题

### Q1: 显存不足 (CUDA out of memory)

**解决方案**：
```bash
# 减小batch size
python train_5090_optimized.py --batch-size 256

# 或减小模型
python train_5090_optimized.py --hidden-dim 128 --kg-emb-dim 128

# 或使用梯度累积
python train_5090_optimized.py --batch-size 128 --gradient-accumulation-steps 4
```

### Q2: GPU利用率低

**检查**：
```bash
# 实时监控
python monitor_5090.py --mode monitor

# 如果利用率<70%，增加batch size
python train_5090_optimized.py --batch-size 1024
```

### Q3: 训练速度慢

**优化**：
```bash
# 1. 确保使用混合精度
python train_5090_optimized.py --use-amp

# 2. 增加batch size
python train_5090_optimized.py --batch-size 1024

# 3. 减少eval频率
python train_5090_optimized.py --eval-freq 10
```

### Q4: AutoDL连接断开

**防止断连**：
```bash
# 使用screen或tmux
screen -S training
bash start_training_5090.sh full

# 断开：Ctrl+A, D
# 恢复：screen -r training
```

## 成本估算

基于 AutoDL 5090 定价（假设 ¥10/小时）：

| 模式 | 时长 | 成本 | 适用场景 |
|------|------|------|---------|
| Quick | 15分钟 | ¥2.5 | 测试验证 |
| Medium | 2小时 | ¥20 | 基线结果 |
| Full | 6小时 | ¥60 | 充分训练 |
| Ultra | 24小时 | ¥240 | 最佳性能 |

## 最佳实践

1. **先测试再训练**
   ```bash
   bash start_training_5090.sh quick  # 验证环境
   ```

2. **监控资源使用**
   ```bash
   python monitor_5090.py --mode monitor  # 实时监控
   ```

3. **定期保存检查点**
   ```bash
   --save-freq 5  # 更频繁保存
   ```

4. **使用混合精度**
   ```bash
   --use-amp  # 默认开启，提速+省显存
   ```

5. **批量实验**
   ```bash
   # 充分利用GPU，同时跑多个实验
   python train_5090_optimized.py --save-dir exp1 &
   python train_5090_optimized.py --lr 1e-4 --save-dir exp2 &
   ```

## 参考资料

- 默认训练脚本: `train.py`
- 5090优化脚本: `train_5090_optimized.py`
- 监控脚本: `monitor_5090.py`
- 启动脚本: `start_training_5090.sh`
- 环境配置: `autodl_setup_5090.sh`

## 技术支持

遇到问题时：
1. 检查GPU信息: `python monitor_5090.py --mode summary`
2. 查看日志: `tail -f logs/training.log`
3. 测试环境: `python test_setup.py`
