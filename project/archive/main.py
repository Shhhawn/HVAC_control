import os
import shutil
import glob
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import sinergym
import numpy as np
import torch
from agent import PPOAgent

def train_ppo(env_name='Eplus-5zone-hot-continuous-v1', max_episodes=50,
              weight_temp_linear=10.0, weight_temp_sq=2.0, weight_energy=0.02,
              run_name="default_run"):
    
    base_env = gym.make(env_name)
    env = NormalizeObservation(base_env)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    a_low = env.action_space.low
    a_high = env.action_space.high
    
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=1e-4, gamma=0.99)
    update_timestep = 2000 
    time_step = 0
    
    # ======== 早停机制 (Early Stopping) 设置 ========
    patience = 8               # 考察过去 8 个 Episode
    recent_rewards = []        # 记录最近的奖励
    std_threshold = 80.0       # 如果这 8 次的奖励标准差小于 80，判定为收敛
    
    # 确保保存目录存在
    save_dir = f"./results/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"========== 🚀 启动训练: {run_name} ==========")
    print(f"权重配置 -> 线性温度: {weight_temp_linear} | 平方温度: {weight_temp_sq} | 能耗: {weight_energy}")
    
    # 用于记录最终指标
    final_energy_kwh = 0
    final_avg_temp_viol = 0

    for ep in range(1, max_episodes + 1):
        state, info = env.reset()
        ep_reward = 0
        ep_energy = 0
        ep_comfort = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            
            action_tanh = agent.select_action(state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)

            deadband_penalty = 0
            if action_env[1] - action_env[0] < 2.0:
                deadband_penalty = -50.0

            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, default_reward, terminated, truncated, info = env.step(action_env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            step_power_kw = step_power_w / 1000.0
            
            # 使用外部传入的超参数
            temp_penalty = (weight_temp_linear * step_temp_viol) + (weight_temp_sq * (step_temp_viol ** 2))
            power_penalty = weight_energy * step_power_kw
            
            custom_reward = - temp_penalty - power_penalty + deadband_penalty
            reward = custom_reward / 100.0
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            ep_reward += reward
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol

            if time_step % update_timestep == 0:
                agent.update()
                
        # 计算核心指标
        avg_temp_violation = ep_comfort / 35040.0
        ep_energy_kwh = (ep_energy / 1e6) / 3.6
        
        print(f"Ep {ep:02d} | 奖励: {ep_reward:8.2f} | 能耗: {ep_energy_kwh:8.2f} 度 | 单步均偏: {avg_temp_violation:4.3f} ℃")
        
        # ======== 执行早停检查 ========
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > patience:
            recent_rewards.pop(0)
            
        if ep >= patience:
            current_std = np.std(recent_rewards)
            if current_std < std_threshold:
                print(f"\n🛑 触发早停机制！过去 {patience} 轮奖励标准差为 {current_std:.2f}，策略已稳定收敛。")
                final_energy_kwh = ep_energy_kwh
                final_avg_temp_viol = avg_temp_violation
                break
                
        # 如果一直跑到最后都没触发早停，记录最后一次的数据
        if ep == max_episodes:
            final_energy_kwh = ep_energy_kwh
            final_avg_temp_viol = avg_temp_violation

    env.close()
    
    # 保存权重并清理战场
    torch.save(agent.policy.state_dict(), f"{save_dir}/ppo_weights.pth")
    
    # 自动清理当前目录下所有 Sinergym 产生的冗余文件夹
    for f in glob.glob(f"{env_name}-res*"):
        shutil.rmtree(f, ignore_errors=True)
        
    print(f"✅ [{run_name}] 训练完成，清理完毕！")
    return final_energy_kwh, final_avg_temp_viol

if __name__ == "__main__":
    train_ppo(run_name="manual_test")