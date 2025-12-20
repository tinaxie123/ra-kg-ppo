# 云服务器快速参考卡片

## 🚀 最快部署方案（AutoDL）

### 1分钟开始

```bash
# 本地：打包代码
python package_for_cloud.py

# 本地：上传
scp -P 端口 ra_kg_ppo_deploy.tar.gz root@服务器:~

# 服务器：部署
tar -xzf ra_kg_ppo_deploy.tar.gz && cd ra_kg_ppo && bash deploy_cloud.sh
```

---

## 💰 推荐配置和费用

| 平台 | GPU | 费用/小时 | 完整实验(50 epochs) |
|------|-----|----------|---------------------|
| **AutoDL** ⭐ | RTX 3090 | 2.5元 | 20-25元 |
| 恒源云 | RTX 3090 | 3元 | 25-30元 |
| 阿里云 | V100 | 12元 | 100-120元 |
| 腾讯云 | A10 | 10元 | 80-100元 |

**推荐AutoDL**：便宜、快速、即开即用！

---

## 📋 完整部署步骤

### 本地准备（5分钟）

```bash
# Windows
package_for_cloud.bat

# Linux/Mac
python package_for_cloud.py
```

### 开通服务器（5分钟）

**AutoDL**：
1. 注册：https://www.autodl.com/
2. 选择：RTX 3090 + PyTorch 2.1
3. 启动实例
4. 复制SSH命令

### 上传代码（2分钟）

```bash
# 方法1: SCP上传
scp -P 端口 ra_kg_ppo_deploy.tar.gz root@服务器:~

# 方法2: JupyterLab直接拖拽上传
```

### 上传数据（根据网速）

```bash
# 打包本地数据
cd data/amazon-book
tar -czf amazon-book-data.tar.gz train.txt test.txt kg_final.txt

# 上传
scp -P 端口 amazon-book-data.tar.gz root@服务器:~/

# 服务器解压
tar -xzf amazon-book-data.tar.gz -C ra_kg_ppo/data/amazon-book/
```

### 运行实验（6-8小时）

```bash
# SSH登录
ssh -p 端口 root@服务器

# 解压并运行
cd ~
tar -xzf ra_kg_ppo_deploy.tar.gz
cd ra_kg_ppo
bash deploy_cloud.sh
```

---

## 🎯 实验配置选择

### 快速测试（1-2小时，5元）

```bash
python train.py --dataset amazon-book --epochs 10 --batch_size 256
```

### 标准实验（4-6小时，15元）

```bash
python train.py --dataset amazon-book --epochs 30 --batch_size 256
```

### 完整实验（8-10小时，25元）

```bash
python train.py --dataset amazon-book --epochs 50 --batch_size 256
```

### 批量运行3个数据集（24-30小时，75元）

```bash
bash run_all_experiments.sh
```

---

## 📊 监控训练

### 实时查看日志

```bash
tail -f log/train_*.log
```

### 使用tmux（防止断连）

```bash
# 创建会话
tmux new -s train

# 运行训练
python train.py --dataset amazon-book --epochs 50

# 断开: Ctrl+B 然后 D
# 重连: tmux attach -t train
```

### 后台运行

```bash
nohup python train.py --dataset amazon-book --epochs 50 > output.log 2>&1 &

# 查看进度
tail -f output.log

# 查看进程
ps aux | grep python
```

---

## 💾 下载结果

### 训练完成后

```bash
# 在本地运行
scp -P 端口 -r root@服务器:~/ra_kg_ppo/checkpoints ./results
scp -P 端口 -r root@服务器:~/ra_kg_ppo/log ./results
```

### 或者打包下载

```bash
# 服务器上
cd ~/ra_kg_ppo
tar -czf results.tar.gz checkpoints/ log/

# 本地下载
scp -P 端口 root@服务器:~/ra_kg_ppo/results.tar.gz .
```

---

## 🔧 常用命令

### GPU检查

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 磁盘空间

```bash
df -h
du -sh *
```

### 内存使用

```bash
free -h
htop  # 或 top
```

### 杀死进程

```bash
# 查找
ps aux | grep python

# 杀死
kill -9 PID
```

---

## ⚠️ 常见问题

### Q1: GPU不可用

```bash
# 检查
nvidia-smi

# 重启服务器
# AutoDL控制台：停止 -> 启动
```

### Q2: 内存不足

```bash
# 减小batch_size
python train.py --dataset amazon-book --batch_size 128
```

### Q3: SSH断开

使用tmux或nohup，防止训练中断

### Q4: 数据加载失败

```bash
# 检查文件
ls -lh data/amazon-book/

# 重新预处理
python scripts/prepare_data.py --dataset amazon-book --force_rebuild
```

---

## ⏰ 今晚完成时间表

| 时间 | 任务 | 耗时 |
|-----|------|------|
| 18:00 | 开通服务器、打包上传 | 30min |
| 18:30 | 部署环境、预处理数据 | 30min |
| 19:00 | **开始训练（amazon-book, 50 epochs）** | **8小时** |
| 03:00 | 训练完成，下载结果 | - |

**费用**：AutoDL约20-25元

如果跑3个数据集：
- 开始时间：19:00
- 预计完成：次日15:00-19:00
- 费用：约60-75元

---

## 🎓 关键提示

1. ✅ **使用tmux**：防止SSH断开导致训练中断
2. ✅ **保存checkpoint**：定期保存模型检查点
3. ✅ **监控日志**：确保训练正常进行
4. ✅ **及时下载**：训练完成立即下载结果
5. ✅ **停止实例**：完成后立即停止，避免浪费

---

## 📞 紧急救援

### 训练中断

```bash
# 查看最新checkpoint
ls -lt checkpoints/

# 从checkpoint继续（需要修改train.py支持resume）
# 或重新开始
```

### 联系支持

- AutoDL客服：网站右下角在线客服
- 阿里云：工单系统
- 腾讯云：工单系统

---

**祝实验顺利！🚀**
