# 云服务器部署指南

## 快速开始（3步完成）

### Step 1: 选择并开通云服务器

#### 推荐方案：AutoDL（最便宜、最快）

1. **注册账号**：https://www.autodl.com/
2. **选择实例**：
   - 地区：任意（建议选延迟低的）
   - GPU：RTX 3090 (24GB)
   - 镜像：PyTorch 2.1.0 / Python 3.10 / CUDA 11.8
   - 费用：约2.5元/小时
3. **启动实例**
4. **获取连接信息**：
   - SSH地址：`root@region-x.autodl.com -p xxxxx`
   - 密码：会自动生成

#### 备选方案：阿里云/腾讯云

1. 选择GPU实例：gn6v (V100) 或 gn7i (A10)
2. 操作系统：Ubuntu 20.04 / 22.04
3. 安全组：开放22端口（SSH）
4. 购买时长：按量付费

---

## Step 2: 上传代码到服务器

### 方法1：使用打包脚本（推荐）⭐

在本地运行：

```bash
# Windows (运行这个)
python package_for_cloud.py

# 会生成: ra_kg_ppo_deploy.tar.gz
```

然后上传：

```bash
# 使用scp上传（替换为你的服务器地址）
scp -P 端口号 ra_kg_ppo_deploy.tar.gz root@服务器地址:~

# AutoDL示例：
scp -P 12345 ra_kg_ppo_deploy.tar.gz root@region-1.autodl.com:~
```

### 方法2：使用Git（如果已上传GitHub）

```bash
# SSH登录服务器后
git clone https://github.com/你的用户名/ra_kg_ppo.git
cd ra_kg_ppo
```

### 方法3：使用Web界面上传（AutoDL）

AutoDL提供JupyterLab，可以直接拖拽上传文件

---

## Step 3: 在服务器上运行

SSH登录服务器：

```bash
# 连接服务器
ssh -p 端口号 root@服务器地址

# 进入目录
cd ~
tar -xzf ra_kg_ppo_deploy.tar.gz
cd ra_kg_ppo

# 运行部署脚本
bash deploy_cloud.sh
```

部署脚本会自动：
1. ✓ 检查GPU环境
2. ✓ 安装依赖
3. ✓ 预处理数据
4. ✓ 开始训练

---

## 实验配置建议

### 配置1：快速验证（1-2小时）

```bash
python train.py --dataset amazon-book \
    --epochs 10 \
    --batch_size 256 \
    --n_steps 512 \
    --eval_interval 2
```

**预计时间**：1-2小时
**费用**：AutoDL约5元

### 配置2：标准实验（4-6小时）

```bash
python train.py --dataset amazon-book \
    --epochs 30 \
    --batch_size 256 \
    --n_steps 2048 \
    --eval_interval 5
```

**预计时间**：4-6小时
**费用**：AutoDL约10-15元

### 配置3：完整实验（8-10小时）

```bash
python train.py --dataset amazon-book \
    --epochs 50 \
    --batch_size 256 \
    --n_steps 2048 \
    --eval_interval 5
```

**预计时间**：8-10小时
**费用**：AutoDL约20-25元

---

## 监控训练进度

### 方法1：实时查看日志

```bash
# 训练时打开新终端
tail -f log/train_*.log
```

### 方法2：使用tmux（推荐）

```bash
# 创建会话
tmux new -s training

# 运行训练
python train.py --dataset amazon-book --epochs 50

# 断开会话（训练继续）: Ctrl+B 然后按 D

# 重新连接
tmux attach -t training
```

### 方法3：后台运行

```bash
nohup python train.py --dataset amazon-book --epochs 50 > output.log 2>&1 &

# 查看进度
tail -f output.log
```

---

## 下载结果

训练完成后，下载结果到本地：

```bash
# 在本地电脑运行
scp -P 端口号 -r root@服务器地址:~/ra_kg_ppo/checkpoints ./results
scp -P 端口号 -r root@服务器地址:~/ra_kg_ppo/log ./results
```

---

## 成本估算

### AutoDL（RTX 3090）

| 实验配置 | 预计时间 | 费用 |
|---------|---------|-----|
| 快速验证 | 1-2小时 | 5元 |
| 标准实验 | 4-6小时 | 10-15元 |
| 完整实验 | 8-10小时 | 20-25元 |
| **运行3个数据集** | 24-30小时 | **60-75元** |

### 阿里云/腾讯云（V100）

| 实验配置 | 预计时间 | 费用 |
|---------|---------|-----|
| 快速验证 | 1-2小时 | 20-30元 |
| 标准实验 | 4-6小时 | 60-90元 |
| 完整实验 | 8-10小时 | 120-150元 |

---

## 运行多个数据集

如果要跑amazon-book, last-fm, yelp2018三个数据集：

```bash
# 创建批量运行脚本
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash

datasets=("amazon-book" "last-fm" "yelp2018")

for dataset in "${datasets[@]}"; do
    echo "======================================================================"
    echo "Training on $dataset"
    echo "======================================================================"

    # 预处理数据
    python scripts/prepare_data.py --dataset $dataset --force_rebuild

    # 训练
    python train.py --dataset $dataset --epochs 30 --batch_size 256

    echo "[OK] $dataset completed!"
    echo ""
done

echo "All experiments completed!"
EOF

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## 故障排查

### 问题1：GPU不可用

```bash
# 检查GPU
nvidia-smi

# 检查PyTorch是否识别GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题2：内存不足

```bash
# 减小batch_size
python train.py --dataset amazon-book --batch_size 128
```

### 问题3：连接断开

使用tmux或nohup运行，避免SSH断开导致训练中断

---

## 时间规划（今晚完成）

假设现在是下午6点：

| 时间 | 任务 | 预计耗时 |
|-----|------|---------|
| 18:00-18:30 | 开通服务器、上传代码 | 30分钟 |
| 18:30-19:00 | 部署环境、预处理数据 | 30分钟 |
| 19:00-01:00 | 运行实验（amazon-book） | 6小时 |
| 01:00-07:00 | 运行实验（last-fm） | 6小时 |
| 07:00-13:00 | 运行实验（yelp2018） | 6小时 |

**总耗时**：约19小时
**费用**：AutoDL约50元

---

## 联系方式

遇到问题可以：
1. 查看日志：`log/train_*.log`
2. 运行测试：`python test_training.py`
3. 检查环境：`python -c "import torch; print(torch.cuda.is_available())"`
