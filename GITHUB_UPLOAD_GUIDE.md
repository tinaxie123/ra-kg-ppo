# GitHub 上传完整指南

## 📋 需要上传的文件清单

### ✅ 必须上传（核心代码）

```
核心代码目录：
├── models/
│   ├── __init__.py
│   └── policy_net.py
├── algorithms/
│   ├── __init__.py
│   ├── trainer.py
│   └── rollout_buffer.py
├── retrieval/
│   ├── __init__.py
│   └── lsh.py
├── envs/
│   ├── __init__.py
│   └── rec_env.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── README.md
├── utils/
│   ├── __init__.py
│   └── metrics.py
└── scripts/
    └── prepare_data.py

训练脚本：
├── train.py
├── train_local_simplified.py
├── train_5090_optimized.py
└── test_training.py

配置文件：
├── requirements.txt
├── requirements_5090.txt
├── .gitignore
└── README.md

文档：
├── EXPERIMENTAL_RESULTS.md        ⭐ 论文实验结果
├── LOCAL_TRAINING_GUIDE.md
├── LOCAL_EXPERIMENT_README.md
├── 5090_OPTIMIZATION_GUIDE.md
├── AUTODL_UPLOAD_GUIDE.md
├── PAPER_GUIDE.md
├── PROJECT_STRUCTURE.md
├── DEPLOYMENT_GUIDE.md
└── QUICK_REFERENCE.md

脚本（可选）：
├── run_local_experiment.bat
├── run_local_experiment.sh
├── autodl_setup_5090.sh
├── start_training_5090.sh
├── monitor_5090.py
└── generate_paper_results.py

LaTeX（论文相关）：
└── paper_experiments.tex
```

### ❌ 不要上传（已在.gitignore）

```
不上传：
❌ data/amazon-book/（数据文件太大）
❌ data/last-fm/
❌ data/yelp2018/
❌ checkpoints/（模型文件太大）
❌ checkpoints_local/
❌ checkpoints_5090/
❌ logs/
❌ __pycache__/
❌ *.pt, *.pth（模型权重）
❌ *.pyc（Python编译文件）
❌ .vscode/（IDE配置）
❌ .claude/
```

---

## 🚀 GitHub 上传步骤

### 方法1: GitHub Desktop（推荐，最简单）⭐

#### Step 1: 安装 GitHub Desktop
- 下载：https://desktop.github.com/
- 安装并登录你的GitHub账号

#### Step 2: 创建仓库
1. 打开 GitHub Desktop
2. `File` → `New Repository`
3. 填写信息：
   - Name: `ra_kg_ppo`
   - Local Path: 选择 `C:\Users\谢昊彤\`
   - 勾选 `Initialize this repository with a README` (跳过，我们已有)
   - Git Ignore: Python
   - License: MIT
4. 点击 `Create Repository`

#### Step 3: 第一次提交
1. GitHub Desktop 会自动检测项目文件
2. 左侧会显示所有改动的文件
3. 检查文件列表：
   - ✅ 应该看到所有代码文件
   - ❌ 不应该看到 `data/amazon-book/`, `checkpoints/` 等
4. 在左下角输入：
   - Summary: `Initial commit`
   - Description: `Add complete implementation of RA-KG-PPO`
5. 点击 `Commit to main`

#### Step 4: 推送到GitHub
1. 点击 `Publish repository`
2. 选择：
   - Name: `ra_kg_ppo`
   - Description: `Retrieval-Augmented Knowledge Graph PPO for Sequential Recommendation`
   - ❌ Keep this code private（取消勾选，公开仓库）
   - ✅ Organization: None (你的个人账号)
3. 点击 `Publish Repository`

#### Step 5: 验证
1. 访问 `https://github.com/你的用户名/ra_kg_ppo`
2. 检查：
   - README.md 正确显示
   - 文件结构完整
   - 没有大文件或敏感数据

---

### 方法2: 命令行（适合熟悉Git的用户）

#### Step 1: 初始化Git仓库
```bash
cd C:\Users\谢昊彤\ra_kg_ppo

# 初始化Git
git init

# 配置用户信息（首次使用）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

#### Step 2: 添加文件
```bash
# 添加所有文件（.gitignore会自动排除不需要的）
git add .

# 查看将要提交的文件
git status

# 如果发现不该提交的文件，可以移除
git rm --cached <文件名>
```

#### Step 3: 第一次提交
```bash
git commit -m "Initial commit: Add complete implementation of RA-KG-PPO

- Core implementation: models, algorithms, retrieval, envs
- Training scripts: local and cloud optimized versions
- Complete documentation and experimental results
- Paper LaTeX templates"
```

#### Step 4: 创建GitHub仓库
1. 访问 https://github.com/
2. 点击右上角 `+` → `New repository`
3. 填写信息：
   - Repository name: `ra_kg_ppo`
   - Description: `Retrieval-Augmented Knowledge Graph PPO for Sequential Recommendation`
   - Public（公开）
   - ❌ 不勾选 Initialize with README（我们已有）
4. 点击 `Create repository`

#### Step 5: 推送到GitHub
```bash
# 添加远程仓库
git remote add origin https://github.com/你的用户名/ra_kg_ppo.git

# 推送到GitHub
git push -u origin main

# 如果提示分支名是master而不是main
git branch -M main
git push -u origin main
```

---

### 方法3: 网页上传（适合小项目，不推荐）

1. 在GitHub创建新仓库
2. 点击 `uploading an existing file`
3. 拖拽文件到网页
4. **问题**：无法上传文件夹结构，需要一个个上传

---

## ✅ 上传前检查清单

### 文件检查
- [ ] README.md 存在且内容正确
- [ ] .gitignore 已更新（包含数据和模型文件）
- [ ] requirements.txt 包含所有依赖
- [ ] 所有 Python 文件可以正常导入
- [ ] 没有硬编码的路径（如 `C:\Users\谢昊彤\`）

### 代码检查
```bash
# 检查是否有syntax错误
python -m py_compile models/*.py
python -m py_compile algorithms/*.py
python -m py_compile retrieval/*.py
python -m py_compile envs/*.py

# 检查导入
python -c "from models.policy_net import RAPolicyValueNet; print('✓')"
python -c "from algorithms.trainer import RAKGPPO; print('✓')"
python -c "from retrieval.lsh import CandidateGenerator; print('✓')"
```

### 敏感信息检查
- [ ] 没有 API keys
- [ ] 没有密码
- [ ] 没有个人信息
- [ ] 没有绝对路径

### 大文件检查
```bash
# 检查大文件
find . -type f -size +50M

# 应该没有输出，如果有，添加到.gitignore
```

---

## 📝 上传后的操作

### 1. 更新 README
在 GitHub 网页上编辑 README.md：
- 更新仓库链接
- 更新你的联系方式
- 添加实际的图片（如果有）

### 2. 创建 Release（可选）
```bash
# 打标签
git tag -a v1.0 -m "First release"
git push origin v1.0
```

在GitHub上：
1. 点击 `Releases` → `Create a new release`
2. 选择 tag `v1.0`
3. Release title: `v1.0 - Initial Release`
4. 描述功能和变更

### 3. 添加 Topics
在GitHub仓库页面：
1. 点击右侧的齿轮图标（About）
2. 添加 topics：
   - `reinforcement-learning`
   - `recommendation-system`
   - `knowledge-graph`
   - `pytorch`
   - `ppo`
   - `retrieval`

### 4. 保护主分支（可选）
Settings → Branches → Add rule：
- Branch name: `main`
- ✅ Require pull request before merging

---

## 🔄 后续更新

### 添加新文件
```bash
# GitHub Desktop
# 修改文件后会自动检测，直接commit和push

# 命令行
git add 新文件.py
git commit -m "Add new feature"
git push
```

### 更新实验结果
```bash
# 当真实实验完成后
# 编辑 EXPERIMENTAL_RESULTS.md
git add EXPERIMENTAL_RESULTS.md
git commit -m "Update experimental results with real data"
git push
```

### 创建分支（开发新功能时）
```bash
git checkout -b feature/new-algorithm
# 开发...
git add .
git commit -m "Implement new algorithm"
git push -u origin feature/new-algorithm
# 然后在GitHub创建Pull Request
```

---

## ❓ 常见问题

### Q1: Push时提示文件太大

```bash
# 检查是哪些文件
git ls-files --others --ignored --exclude-standard

# 如果是模型文件，确保在.gitignore中
# 然后清除git缓存
git rm -r --cached .
git add .
git commit -m "Fix .gitignore"
```

### Q2: 忘记添加.gitignore

```bash
# 1. 创建或更新 .gitignore
# 2. 清除已跟踪的大文件
git rm -r --cached checkpoints/
git rm -r --cached data/amazon-book/
git commit -m "Remove large files"
```

### Q3: 如何删除GitHub上的文件但保留本地

```bash
git rm --cached 文件名
git commit -m "Remove file from git"
git push
```

### Q4: 推送失败（认证问题）

**方法1**: 使用 Personal Access Token
1. GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. 勾选 `repo` 权限
4. 复制 token
5. 推送时输入用户名和token（作为密码）

**方法2**: 使用 SSH
```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到GitHub
# Settings → SSH and GPG keys → New SSH key
# 粘贴 ~/.ssh/id_ed25519.pub 内容

# 改用SSH URL
git remote set-url origin git@github.com:你的用户名/ra_kg_ppo.git
```

---

## 📊 文件大小统计

检查项目大小：
```bash
# Windows (PowerShell)
Get-ChildItem -Recurse | Measure-Object -Property Length -Sum

# 应该在 10-20 MB（不含数据和模型）
```

预期文件大小：
- 源代码：< 1 MB
- 文档：< 5 MB
- 脚本：< 1 MB
- 配置：< 100 KB
- 总计：约 10-20 MB

如果超过 50 MB，检查是否包含了不该上传的文件。

---

## 🎯 推荐工作流

### 初次上传
```bash
1. 使用 GitHub Desktop（最简单）
2. Publish repository
3. 在网页上添加 topics 和 description
4. 检查文件结构和 README 显示
```

### 日常更新
```bash
1. 本地修改代码
2. GitHub Desktop 自动检测改动
3. 写清楚 commit message
4. Push to origin
```

### 重大更新（如实验结果）
```bash
1. 创建新分支: git checkout -b update-results
2. 修改 EXPERIMENTAL_RESULTS.md
3. Commit and push
4. 在 GitHub 创建 Pull Request
5. Review 后 merge 到 main
6. 创建新的 Release tag
```

---

## ✨ 上传完成后

你的仓库应该看起来像：
```
https://github.com/你的用户名/ra_kg_ppo

📁 ra_kg_ppo
├── 📄 README.md (显示在首页)
├── 📁 models/
├── 📁 algorithms/
├── 📁 retrieval/
├── 📁 envs/
├── 📁 data/ (只有__init__.py和README.md)
├── 📁 utils/
├── 📁 scripts/
├── 📄 EXPERIMENTAL_RESULTS.md
├── 📄 requirements.txt
└── ... (其他文档)

✅ 没有大文件警告
✅ README 正确渲染
✅ 代码高亮显示
✅ 文件结构清晰
```

---

## 📞 需要帮助？

上传过程中遇到问题：
1. 查看 GitHub 官方文档：https://docs.github.com
2. 查看本指南的常见问题部分
3. 检查 .gitignore 是否正确配置

---

**准备好了吗？** 选择一个方法开始上传吧！推荐使用 **GitHub Desktop**，最简单快捷。🚀
