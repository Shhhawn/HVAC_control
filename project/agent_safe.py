import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks import HVACActorCritic

# ==========================================
# 1. 拆分后的 Buffer
# ==========================================
class SafeRolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        # 👑 核心：剥离 reward，独立记录能耗和温度
        self.energy_costs, self.temp_costs = [], [] 
        self.values, self.is_terminals = [], []
    def clear(self):
        self.states.clear(); self.actions.clear(); self.logprobs.clear()
        self.energy_costs.clear(); self.temp_costs.clear()
        self.values.clear(); self.is_terminals.clear()

class SafePPOAgent:
    def __init__(self, obs_dim=17, action_dim=2, lr=1e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, 
                 temporal_type='gru', stack_size=4, extractor_type='full'):
        self.gamma = gamma          
        self.eps_clip = eps_clip    
        self.K_epochs = K_epochs    
        self.buffer = SafeRolloutBuffer() # 使用新 Buffer

        self.policy = HVACActorCritic(obs_dim, action_dim, temporal_type, stack_size, extractor_type)
        self.policy_old = HVACActorCritic(obs_dim, action_dim, temporal_type, stack_size, extractor_type)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        # ==========================================
        # 👑 2. 拉格朗日乘子定义
        # ==========================================
        self.log_lambda = torch.tensor([-1.0], requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=0.05)
        self.target_temp_dev = 0.50 # 🎯 我们的终极红线：0.5℃

    @property
    def lagrangian_multiplier(self):
        return torch.exp(self.log_lambda).detach().item()

    def select_action(self, state):
        state_array = np.array(state) 
        if state_array.ndim == 1: state_array = np.expand_dims(state_array, axis=0)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0) 
            action_mean, action_std, value, gate_weights = self.policy_old(state_tensor)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(value.flatten())
        
        action_np = action.squeeze(0).numpy()
        return np.clip(action_np, -1.0, 1.0)

    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach()
        old_values = torch.squeeze(torch.stack(self.buffer.values, dim=0), dim=1).detach()

        current_lambda = self.lagrangian_multiplier

        # ==========================================
        # 👑 3. 动态生成 Reward (市长算账)
        # ==========================================
        rewards = []
        discounted_reward = 0
        for e_cost, t_cost, is_terminal in zip(reversed(self.buffer.energy_costs), reversed(self.buffer.temp_costs), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            
            # 核心公式：总奖励 = -能耗 - λ * 温度偏差
            step_reward = -e_cost - current_lambda * t_cost
            
            discounted_reward = step_reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        returns = torch.tensor(rewards, dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # ====== PPO Actor-Critic 更新 ======
        for _ in range(self.K_epochs):
            action_mean, action_std, state_values, _ = self.policy(old_states)
            dist = Normal(action_mean, action_std)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss_actor = -torch.min(surr1, surr2)
            loss_critic = self.MseLoss(state_values, returns)
            loss = loss_actor + 0.5 * loss_critic - 0.05 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # ==========================================
        # 👑 4. 对偶梯度上升更新 Lambda (局长开罚单)
        # ==========================================
        mean_temp_cost = torch.tensor(self.buffer.temp_costs, dtype=torch.float32).mean()
        
        # PyTorch 优化器是最小化，所以前面加负号实现最大化
        loss_lambda = - torch.exp(self.log_lambda) * (mean_temp_cost - self.target_temp_dev)
        
        self.lambda_optimizer.zero_grad()
        loss_lambda.backward()
        torch.nn.utils.clip_grad_norm_([self.log_lambda], max_norm=0.5) 
        self.lambda_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()