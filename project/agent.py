import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks import HVACActorCritic

class RolloutBuffer:
    """
    经验回放池：用于暂存智能体在环境中交互的数据。
    PPO 是 On-policy 算法，用完这批数据更新完网络后，就会清空。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.is_terminals.clear()


class PPOAgent:
    """
    PPO 算法核心逻辑
    """
    def __init__(self, obs_dim=17, action_dim=2, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma          # 折扣因子 (越接近1，越看重长期能耗和舒适度)
        self.eps_clip = eps_clip    # PPO 的核心：截断范围，防止策略更新步子迈太大
        self.K_epochs = K_epochs    # 每次收集完数据后，拿这批数据反复训练多少次
        
        self.buffer = RolloutBuffer()

        # 初始化你的创新网络 (三通道 + Gate + DCN)
        self.policy = HVACActorCritic(obs_dim, action_dim)
        # PPO 需要一个"旧策略"网络来计算概率比率 (Ratio)
        self.policy_old = HVACActorCritic(obs_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # 均方误差，用于更新 Critic 网络
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """
        根据当前状态，采样出一个动作，并记录必要的梯度信息
        """
        # 将 numpy 数组转为 Tensor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0) # 增加 batch 维度
            
            # 通过旧策略网络获取分布参数
            action_mean, action_std, value, gate_weights = self.policy_old(state_tensor)
            
            # 构建正态分布
            dist = Normal(action_mean, action_std)
            # 从分布中采样一个动作
            action = dist.sample()
            # 计算该动作的对数概率
            action_logprob = dist.log_prob(action).sum(dim=-1)
            
        # 记录到 Buffer 中
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(value.flatten())
        
        # 将 Tensor 转回 numpy，并限制在 [-1, 1] 范围内送给环境
        action_np = action.squeeze(0).numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        
        return action_np

    def update(self):
        """
        使用收集到的数据，执行 PPO 的网络更新
        """
        # 1. 把 Buffer 里的 List 转换成 Tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach()
        old_values = torch.squeeze(torch.stack(self.buffer.values, dim=0), dim=1).detach()

        # 2. 计算回报 (Returns) 和优势 (Advantages)
        # 这里使用经典的蒙特卡洛估计，也可以换成 GAE
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # # 归一化奖励，提升训练稳定性
        # rewards = torch.tensor(rewards, dtype=torch.float32)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # # 优势函数 A(s,a) = R - V(s)
        # advantages = rewards - old_values

        # ==================== ✅ 替换为正确的 PPO 优势归一化 ====================
        returns = torch.tensor(rewards, dtype=torch.float32)
        
        # 绝对回报(Returns)原封不动地给 Critic 做目标，绝对不能归一化！
        advantages = returns - old_values

        # 只对优势函数(Advantages)进行归一化，保证策略梯度更新平稳
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # ====================================================================

        # 3. K_epochs 次的迭代优化
        for _ in range(self.K_epochs):
            # 获取当前策略对这些旧数据的评估
            action_mean, action_std, state_values, _ = self.policy(old_states)
            dist = Normal(action_mean, action_std)
            
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = torch.squeeze(state_values)

            # 计算概率比率 Ratio = pi_theta(a|s) / pi_theta_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算 PPO 的双重 Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor Loss (取最小值作为悲观估计)
            loss_actor = -torch.min(surr1, surr2)
            # Critic Loss (均方误差)
            loss_critic = self.MseLoss(state_values, returns)
            
            # 总 Loss = ActorLoss + 0.5 * CriticLoss - 0.01 * 探索奖励(熵)
            loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy

            # 反向传播，更新网络
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. 更新完成后，将当前策略硬拷贝给旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 5. 清空经验池，准备下一轮收集
        self.buffer.clear()