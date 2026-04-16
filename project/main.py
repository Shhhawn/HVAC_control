import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import sinergym
import numpy as np
import torch
from agent import PPOAgent

def train_ppo(env_name='Eplus-5zone-hot-continuous-v1', max_episodes=50):
    base_env = gym.make(env_name)
    env = NormalizeObservation(base_env)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 动作空间的物理边界，用于将神经网络的 [-1, 1] 映射为真实温度
    a_low = env.action_space.low
    a_high = env.action_space.high
    
    # 初始化你的创新 PPO Agent
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=1e-4, gamma=0.99)
    
    # PPO 超参数：每积攒多少步经验，就更新一次网络
    update_timestep = 2000 
    time_step = 0
    
    print("========== 🚀 开始训练 多通道 Gate PPO 智能体 ==========")
    print(f"环境: {env_name} | 状态维度: {obs_dim} | 动作维度: {action_dim}")
    
    for ep in range(1, max_episodes + 1):
        state, info = env.reset()
        ep_reward = 0
        ep_energy = 0
        ep_comfort = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            
            # 1. 智能体根据状态输出动作 [-1, 1]
            action_tanh = agent.select_action(state)
            
            # 2. 核心！将 [-1, 1] 的动作线性映射到环境的真实物理边界
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)

            deadband_penalty = 0
            if action_env[1] - action_env[0] < 2.0:
                # 给一个极大的负反馈，让它知道这么做代价惨重
                deadband_penalty = -50.0

            action_env = np.clip(action_env, a_low, a_high) # 双重保险
            
            # 3. 与环境交互
            next_state, default_reward, terminated, truncated, info = env.step(action_env)

            # ========== 自定义奖励函数 =============
            # 获取当前步真实的物理指标
            step_power_w = info.get('total_power_demand', 0.0) # 瞬时功率 (瓦特)
            step_temp_viol = info.get('total_temperature_violation', 0.0) # 温度超标度数
            
            # 将功率转换为千瓦 (kW)，缩小数值量级，方便与温度对齐
            step_power_kw = step_power_w / 1000.0
            
            # 【调节天平】
            weight_temp_linear = 10.0
            # weight_temp: 温度超标的惩罚权重 (调大这个值，AI 就会怕热，拼命开空调)
            # weight_energy: 耗电的惩罚权重 (调大这个值，AI 就会省电)
            weight_temp_sq = 2.0   # 之前环境默认可能太低了，我们直接拉高到 5 倍惩罚！
            weight_energy = 0.02 # 适度惩罚耗电

            temp_penalty = (weight_temp_linear * step_temp_viol) + (weight_temp_sq * (step_temp_viol ** 2))
            power_penalty = weight_energy * step_power_kw
            
            # 计算我们自己的核心 Reward
            custom_reward = - temp_penalty - power_penalty + deadband_penalty
            reward = custom_reward / 100.0

            # ====== 新增这 3 行探针代码 ======
            # if time_step == 1:
            #     print("\n【终极调试】第一步环境吐出的 info 字典里到底有什么？")
            #     for k, v in info.items():
            #         print(f"键: '{k}' -> 值类型: {type(v)}")
            # ==================================
            
            # 4. 记录奖励和终止信号 (供 PPO 计算 Advantage)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            ep_reward += reward
            # ep_energy += info.get('energies', [info.get('total_energy', 0)])[0]
            # ep_energy += info.get('total_electricity_HVAC', 0)
            current_power = info.get('total_power_demand', 0)
            ep_energy += current_power * 900
            ep_comfort += info.get('total_temperature_violation', 0)
            
            # 5. 如果经验池满了，执行一次 PPO 网络更新
            if time_step % update_timestep == 0:
                print(f"   [PPO Update] 正在更新网络权重... (当前经验池大小: {len(agent.buffer.states)})")
                agent.update()
                
        # 打印当前回合的最终成绩
        print(f"Episode {ep:02d} | 奖励: {ep_reward:8.2f} | 能耗: {ep_energy/1e6:8.2f} MJ | 温度超标总计: {ep_comfort:8.2f} ℃")
        
    env.close()
    
    # 训练结束后，保存你引以为傲的模型权重
    torch.save(agent.policy.state_dict(), "./results/hvac_multi_channel_ppo.pth")
    print("✅ 训练完成！模型权重已保存至 hvac_multi_channel_ppo.pth")

if __name__ == "__main__":
    train_ppo(max_episodes=300)