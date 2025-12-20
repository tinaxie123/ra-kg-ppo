
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import argparse
from tqdm import tqdm
from data.dataset import load_kgat_data
from models.policy_net import RAPolicyValueNet
from retrieval.lsh import CandidateGenerator
from envs.rec_env import RecommendationEnv
from utils.metrics import evaluate_policy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon-book')
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--item-emb-dim', type=int, default=64)
    parser.add_argument('--kg-emb-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-hash-bits', type=int, default=8)
    parser.add_argument('--num-tables', type=int, default=4)
    parser.add_argument('--candidate-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-test-episodes', type=int, default=10)
    
    return parser.parse_args()


def simple_train_episode(env, policy_net, candidate_gen, optimizer, device, gamma=0.99):
  
    obs = env.reset()
    states = []
    actions = []
    rewards = []
    log_probs = []
    
    done = False
    steps = 0
    max_steps = 20
    
    while not done and steps < max_steps:
        # 转换观察为tensor
        item_emb = torch.FloatTensor(obs['item_embeddings']).unsqueeze(0).to(device)
        length = torch.tensor([obs['length']]).to(device)
        
        # 获取隐状态
        hidden = policy_net.get_hidden_state(item_emb, length)
        
        # 生成候选
        query, cand_ids, cand_embs = candidate_gen(hidden)
        
        # 计算logits
        logits = policy_net.actor.compute_action_logits(query, cand_embs)
        
        # 采样动作
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        # 映射到真实物品ID
        real_action = cand_ids[0, action_idx].item()
        
        # 执行动作
        next_obs, reward, done, info = env.step(real_action)
        
        # 保存
        states.append((item_emb, length, query, cand_embs))
        actions.append(action_idx)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        obs = next_obs
        steps += 1
    
    if len(rewards) == 0:
        return 0.0
    
    # 计算returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, device=device)
    
    # 策略梯度更新
    policy_loss = 0
    for log_prob, G in zip(log_probs, returns):
        policy_loss -= log_prob * G
    
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer.step()
    
    return sum(rewards)


def main():
    args = get_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*70)
    print("RA-KG-PPO 简化训练")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载数据...")
    data = load_kgat_data(args.dataset, args.data_path)
    
    item_embeddings = data['item_embeddings'].to(args.device)
    kg_embeddings = data['kg_embeddings'].to(args.device)
    train_data = data['train_data']
    test_data = data['test_data']
    
    print(f"✓ 数据加载完成")
    print(f"  训练用户: {len(train_data)}")
    print(f"  测试用户: {len(test_data)}")
    
    # 创建环境
    print("\n创建环境...")
    env = RecommendationEnv(
        user_sequences=train_data,
        item_embeddings=item_embeddings,
        kg_embeddings=kg_embeddings,
        max_seq_len=50,
        device=args.device
    )
    print("✓ 环境创建完成")
    
    # 构建模型
    print("\n构建模型...")
    policy_net = RAPolicyValueNet(
        item_embedding_dim=args.item_emb_dim,
        hidden_dim=args.hidden_dim,
        kg_embedding_dim=args.kg_emb_dim,
        num_layers=args.num_layers,
        shared_encoder=True
    ).to(args.device)
    
    candidate_gen = CandidateGenerator(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.kg_emb_dim,
        num_hash_bits=args.num_hash_bits,
        num_tables=args.num_tables,
        candidate_size=args.candidate_size
    ).to(args.device)
    
    # 构建LSH索引
    print("构建LSH索引...")
    candidate_gen.build_index(kg_embeddings)
    print("✓ 模型构建完成")
    
    # 优化器
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(candidate_gen.parameters()),
        lr=args.lr
    )
    
    # 训练
    print(f"\n开始训练 (共{args.epochs}轮)...")
    print("="*70)
    
    for epoch in range(args.epochs):
        # 训练
        policy_net.train()
        candidate_gen.train()
        
        epoch_rewards = []
        num_episodes = 50  # 每轮训练50个episode
        
        for _ in tqdm(range(num_episodes), desc=f"Epoch {epoch+1}/{args.epochs}"):
            reward = simple_train_episode(
                env, policy_net, candidate_gen, optimizer, 
                args.device, args.gamma
            )
            epoch_rewards.append(reward)
        
        avg_reward = np.mean(epoch_rewards)
        
        print(f"\nEpoch {epoch+1}: 平均奖励 = {avg_reward:.4f}")
        
        # 每5轮评估一次
        if (epoch + 1) % 5 == 0:
            print("评估中...")
            # 这里可以添加完整评估代码
    
    print("\n" + "="*70)
    print("✓ 训练完成！")
    print("="*70)
    
    # 保存模型
    torch.save({
        'policy_net': policy_net.state_dict(),
        'candidate_gen': candidate_gen.state_dict(),
    }, f'checkpoint_epoch{args.epochs}.pth')
    
    print(f"\n模型已保存到: checkpoint_epoch{args.epochs}.pth")


if __name__ == '__main__':
    main()