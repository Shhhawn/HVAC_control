import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random

# -----------------------------------------
# 1. 经验回放池 (Off-policy 必备)
# -----------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------
# 2. Critic 网络 (Twin Q-Networks)
# -----------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# -----------------------------------------
# 3. Actor 网络 (Gaussian Policy with Reparameterization)
# -----------------------------------------
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # 截断防止数值爆炸
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 核心：重参数化技巧 (Reparameterization trick)
        y_t = torch.tanh(x_t)   # 将动作压缩到 [-1, 1]
        action = y_t
        # 计算 Tanh 变换后的 log 概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean

# -----------------------------------------
# 4. SAC 智能体核心逻辑
# -----------------------------------------
class SACAgent:
    def __init__(self, obs_dim=17, action_dim=2, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        
        self.actor = GaussianPolicy(obs_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = QNetwork(obs_dim, action_dim)
        self.critic_target = QNetwork(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 自动熵调整 (Auto-Entropy Tuning)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(torch.device('cpu'))).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha

        self.buffer = ReplayBuffer()
        self.batch_size = 256

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            _, _, mean = self.actor.sample(state)
            action = torch.tanh(mean).detach().cpu().numpy()[0]
        else:
            action, _, _ = self.actor.sample(state)
            action = action.detach().cpu().numpy()[0]
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target)

        # 更新 Critic
        qf1, qf2 = self.critic(state, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新 Alpha (温度系数)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新 Target Critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)