# AutoDL 代码上传指南

三种方法上传代码到 AutoDL，选择最适合你的方式。

## 方法1: JupyterLab 直接上传（最简单）⭐

### 步骤：

1. **启动 AutoDL 实例**
   - 登录 AutoDL 控制台
   - 创建/启动你的 5090 实例
   - 点击 "JupyterLab" 按钮

2. **打开 JupyterLab**
   - 浏览器会打开 JupyterLab 界面
   - 左侧是文件浏览器

3. **上传文件**

   **方式A: 上传单个文件**
   - 点击左侧文件浏览器顶部的上传按钮（↑图标）
   - 选择文件上传

   **方式B: 上传压缩包（推荐）**
   ```bash
   # 在本地打包项目
   cd C:\Users\谢昊彤\
   tar -czf ra_kg_ppo.tar.gz ra_kg_ppo/
   # 或者用 zip
   # 右键项目文件夹 -> 发送到 -> 压缩文件
   ```

   - 上传 `ra_kg_ppo.tar.gz` 或 `ra_kg_ppo.zip`
   - 在 JupyterLab 中打开终端
   - 解压：
     ```bash
     cd /root/autodl-tmp
     tar -xzf ra_kg_ppo.tar.gz
     # 或 unzip ra_kg_ppo.zip
     ```

4. **完成**
   ```bash
   cd /root/autodl-tmp/ra_kg_ppo
   ls  # 查看文件
   ```

---

## 方法2: GitHub Clone（适合有 Git 仓库）

### 步骤：

1. **本地推送到 GitHub**
   ```bash
   # 在本地项目目录
   cd C:\Users\谢昊彤\ra_kg_ppo

   # 初始化 Git（如果还没有）
   git init
   git add .
   git commit -m "Initial commit for AutoDL"

   # 关联远程仓库（替换成你的仓库地址）
   git remote add origin https://github.com/你的用户名/ra_kg_ppo.git
   git push -u origin main
   ```

2. **在 AutoDL 上 Clone**
   - 打开 AutoDL 终端或 JupyterLab 终端
   ```bash
   cd /root/autodl-tmp

   # 公开仓库
   git clone https://github.com/你的用户名/ra_kg_ppo.git

   # 私有仓库（需要 token）
   git clone https://你的token@github.com/你的用户名/ra_kg_ppo.git
   ```

3. **配置 Git 忽略大文件**

   创建 `.gitignore`：
   ```bash
   # 数据文件
   data/amazon-book/
   data/*.pkl
   *.npy

   # 模型文件
   checkpoints/
   checkpoints_5090/
   *.pt
   *.pth

   # 日志
   logs/
   tensorboard_logs/

   # Python
   __pycache__/
   *.pyc
   .ipynb_checkpoints/

   # 系统文件
   .DS_Store
   ```

### GitHub 优点：
- ✅ 版本控制
- ✅ 代码同步方便
- ✅ 适合团队协作
- ⚠️ 需要处理大文件（数据、模型）

---

## 方法3: AutoDL 文件同步（官方工具）

### 步骤：

1. **安装 AutoDL 客户端**（可选）
   - 访问 AutoDL 文档查看客户端下载

2. **使用 SSH + SCP**

   在 AutoDL 控制台查看连接信息：
   ```
   SSH地址: root@connect.xxx.autodl.com
   端口: 12345
   密码: xxx
   ```

   上传文件：
   ```bash
   # Windows (使用 PowerShell 或 Git Bash)
   scp -P 12345 -r C:\Users\谢昊彤\ra_kg_ppo root@connect.xxx.autodl.com:/root/autodl-tmp/

   # Linux/Mac
   scp -P 12345 -r ~/ra_kg_ppo root@connect.xxx.autodl.com:/root/autodl-tmp/
   ```

3. **使用 FileZilla 或 WinSCP（图形界面）**
   - 下载 WinSCP (Windows) 或 FileZilla
   - 配置连接：
     - 协议: SFTP
     - 主机: connect.xxx.autodl.com
     - 端口: 12345
     - 用户名: root
     - 密码: [AutoDL提供的密码]
   - 拖拽上传文件

---

## 推荐方案对比

| 方法 | 适用场景 | 速度 | 难度 |
|------|---------|------|------|
| **JupyterLab上传** | 小项目、快速测试 | ⭐⭐⭐ | ⭐ 简单 |
| **GitHub Clone** | 版本控制、团队协作 | ⭐⭐ | ⭐⭐ 中等 |
| **SCP/SFTP** | 大文件、批量上传 | ⭐⭐⭐ | ⭐⭐⭐ 较难 |

---

## 完整流程示例（推荐）

### 使用 JupyterLab + 压缩包上传

```bash
# === 本地操作 ===
# 1. 打包项目（排除大文件）
cd C:\Users\谢昊彤\

# Windows: 右键项目文件夹 -> 发送到 -> 压缩(zipped)文件夹
# 或使用命令（需要安装 7zip 或 tar）
tar -czf ra_kg_ppo.tar.gz ra_kg_ppo/ --exclude=data --exclude=checkpoints

# === AutoDL 操作 ===
# 2. 打开 JupyterLab，上传 ra_kg_ppo.tar.gz

# 3. 在 JupyterLab 终端解压
cd /root/autodl-tmp
tar -xzf ra_kg_ppo.tar.gz
cd ra_kg_ppo

# 4. 配置环境
bash autodl_setup_5090.sh

# 5. 开始训练
bash start_training_5090.sh quick
```

---

## 大文件处理

### 数据文件不要上传！

数据文件通常很大，应该在 AutoDL 上直接下载：

```bash
# 在 AutoDL 上执行
cd /root/autodl-tmp/ra_kg_ppo
python scripts/prepare_data.py --dataset amazon-book
```

### 模型检查点

如果需要上传预训练模型：

```bash
# 方案1: 通过云存储
# 上传到百度网盘/阿里云盘，在AutoDL上wget下载

# 方案2: GitHub Release（<2GB）
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add checkpoint.pt
git commit -m "Add model checkpoint"
git push
```

---

## 常见问题

### Q1: 上传速度慢？

**A:** 先压缩再上传
```bash
# 压缩率能达到 50-90%
tar -czf project.tar.gz ra_kg_ppo/
```

### Q2: 上传中断？

**A:** 使用断点续传工具
```bash
# 使用 rsync
rsync -avz -P --rsh="ssh -p 12345" ra_kg_ppo/ root@connect.xxx.autodl.com:/root/autodl-tmp/ra_kg_ppo/
```

### Q3: 文件权限问题？

**A:** 修复权限
```bash
chmod +x *.sh
chmod -R 755 /root/autodl-tmp/ra_kg_ppo/
```

### Q4: Git Clone 速度慢？

**A:** 使用国内镜像
```bash
# GitHub 加速（使用镜像）
git clone https://ghproxy.com/https://github.com/用户名/仓库名.git

# 或使用 Gitee（国内）
# 先在 Gitee 导入 GitHub 仓库
git clone https://gitee.com/用户名/仓库名.git
```

---

## 实战：完整部署流程

```bash
# ========== 本地准备 ==========
# 1. 整理项目文件
cd C:\Users\谢昊彤\ra_kg_ppo

# 2. 创建 .gitignore（如果用 Git）
# （参考上面的 .gitignore 内容）

# 3. 压缩项目
# 右键 -> 发送到 -> 压缩文件

# ========== AutoDL 部署 ==========
# 4. 登录 AutoDL，启动实例

# 5. 打开 JupyterLab，上传 ra_kg_ppo.zip

# 6. 打开 Terminal，执行：
cd /root/autodl-tmp
unzip ra_kg_ppo.zip
cd ra_kg_ppo

# 7. 给脚本执行权限
chmod +x *.sh

# 8. 一键配置
bash autodl_setup_5090.sh

# 9. 快速测试
bash start_training_5090.sh quick

# 10. 监控GPU（新开终端）
python monitor_5090.py --mode monitor

# ========== 开始正式训练 ==========
# 11. 使用 screen 防止断连
screen -S training
bash start_training_5090.sh medium

# 断开：Ctrl+A, 然后按 D
# 重连：screen -r training
```

---

## 小贴士

1. **首次上传建议**：JupyterLab + 压缩包（最简单）
2. **经常修改代码**：使用 GitHub（方便同步）
3. **大文件传输**：使用 SCP/SFTP
4. **数据文件**：不要上传，在 AutoDL 上直接下载
5. **使用 screen**：防止 SSH 断开导致训练中断

---

## 下一步

上传完成后：
1. 阅读：`5090_OPTIMIZATION_GUIDE.md`
2. 配置：`bash autodl_setup_5090.sh`
3. 训练：`bash start_training_5090.sh medium`
4. 监控：`python monitor_5090.py --mode monitor`
