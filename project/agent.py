import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks import HVACActorCritic

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.values, self.is_terminals = [], [], [], [], [], []
    def clear(self):
        self.states.clear(); self.actions.clear(); self.logprobs.clear()
        self.rewards.clear(); self.values.clear(); self.is_terminals.clear()

class PPOAgent:
    def __init__(self, obs_dim=17, action_dim=2, lr=1e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, 
                 temporal_type='mlp', stack_size=1, extractor_type='full'): # ✅ 新增参数
        self.gamma = gamma          
        self.eps_clip = eps_clip    
        self.K_epochs = K_epochs    
        self.buffer = RolloutBuffer()

        # ✅ 将模式传给大脑
        self.policy = HVACActorCritic(obs_dim, action_dim, temporal_type, stack_size, extractor_type)
        self.policy_old = HVACActorCritic(obs_dim, action_dim, temporal_type, stack_size, extractor_type)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # ✅ Gym 的 FrameStack 会返回 LazyFrames，必须用 np.array 强制转换
        state_array = np.array(state) 
        # 如果是单帧 (mlp)，强行加一个时间维度以兼容网络 (Stack_Size=1)
        if state_array.ndim == 1:
            state_array = np.expand_dims(state_array, axis=0)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0) # 最终 shape: (1, S, 17)
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
        # Buffer 里的 shape 是 (1, S, 17)，堆叠并挤掉 dim=1
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach()
        old_values = torch.squeeze(torch.stack(self.buffer.values, dim=0), dim=1).detach()

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        returns = torch.tensor(rewards, dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

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

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()