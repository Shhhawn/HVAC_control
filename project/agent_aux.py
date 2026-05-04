import torch
import torch.nn as nn
from agent import PPOAgent, RolloutBuffer # 复用你之前的代码

class AuxRolloutBuffer(RolloutBuffer):
    def __init__(self):
        super().__init__()
        self.next_states = [] # 👑 新增：必须记住真实的下一步状态
    
    def clear(self):
        super().clear()
        self.next_states = []

class AuxPPOAgent(PPOAgent):
    def __init__(self, obs_dim=17, action_dim=2, temporal_type='gru', stack_size=4, extractor_type='full', aux_weight=0.5):
        super().__init__(obs_dim, action_dim, temporal_type=temporal_type, stack_size=stack_size, extractor_type=extractor_type)
        self.buffer = AuxRolloutBuffer()
        self.aux_weight = aux_weight # 辅助任务的权重系数，通常 0.5 或 1.0

    def update(self):
        # 1. 整理数据 (和标准 PPO 一样)
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs)).to(self.device)
        
        # 👑 取出真实的下一时刻状态 (展平用于计算 MSE)
        B, S, D = old_states.shape
        old_next_states = torch.FloatTensor(np.array(self.buffer.next_states)).to(self.device).view(B, S*D)
        
        # 计算优势函数 Advantages... (此处省略，完全复用你原本 agent.py 里的 GAE/Discount 逻辑)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. PPO 迭代更新
        for _ in range(self.K_epochs):
            # 👑 调用我们新写的 evaluate_with_dynamics，不仅拿策略，还拿对未来的预测
            logprobs, state_values, dist_entropy, pred_next_states = self.policy.evaluate_with_dynamics(old_states, old_actions)
            
            state_values = state_values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.MseLoss(state_values, rewards)
            
            # 👑 核心创新：物理预测损失 (MSE Loss)
            loss_aux = self.MseLoss(pred_next_states, old_next_states)
            
            # 终极混合损失 = PPO策略损失 + 价值损失 + β * 物理预测损失
            loss_total = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy.mean() + self.aux_weight * loss_aux
            
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()