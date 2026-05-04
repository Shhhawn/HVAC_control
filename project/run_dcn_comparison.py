# ======================
# 用于比较DCN-V1和DCN-V2的效果
# seed = 42，V1的最高平均reward为-8612.22；V2的最高平均reward为-9473.88；
# seed = 10，V1的最高平均reward为-8654.13；V2的最高平均reward为-8300.32；
# ======================


import os
import shutil
import glob
import json
import numpy as np
import torch
import random
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import sinergym
from agent import PPOAgent

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def static_normalize(state_array, env):
    high = env.observation_space.high
    low = env.observation_space.low
    high = np.where(high > 1e10, 100.0, high) 
    low = np.where(low < -1e10, -100.0, low)
    norm_state = 2.0 * (state_array - low) / (high - low) - 1.0
    return np.clip(norm_state, -5.0, 5.0)

def train_dcn_version(dcn_version, max_episodes=50): # 放宽最大轮次，交给早停来判断
    env_name = "Eplus-5zone-hot-continuous-v1"
    run_name = f"DCN_Comparison_{dcn_version}"
    save_dir = f"./results/DCN_Ablation/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🥊 启动特征交叉架构之战 | 参赛选手: DCN-{dcn_version}")
    print(f"{'='*60}")
    
    base_env = gym.make(env_name)
    env = FrameStackObservation(base_env, stack_size=4)
        
    obs_dim = base_env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]
    a_low, a_high = env.action_space.low, env.action_space.high
    
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, 
                     temporal_type='gru', stack_size=4, extractor_type='full', 
                     dcn_version=dcn_version)
    
    update_timestep = 2000 
    training_log = []
    
    # ==========================================
    # 👑 引入极其公平的竞赛机制：早停与最高分记录
    # ==========================================
    best_ma_reward = -float('inf')
    recent_rewards = []
    patience_limit = 10 # 连续 10 轮滑动平均不破纪录，则判定为已达极限
    patience_counter = 0

    for ep in range(1, max_episodes + 1):
        state, info = env.reset(seed=42+ep) 
        time_step = 0
        ep_reward, ep_energy, ep_comfort = 0, 0, 0
        terminated = False; truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            norm_state = static_normalize(np.array(state), base_env)
            
            action_tanh = agent.select_action(norm_state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            
            deadband_penalty = -50.0 if (action_env[1] - action_env[0] < 2.0) else 0
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            
            temp_penalty = (10.0 * step_temp_viol) + (2.0 * (step_temp_viol ** 2))
            power_penalty = 10.0 * (step_power_w / 1000.0)
            reward = (- temp_penalty - power_penalty + deadband_penalty) / 100.0
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            ep_reward += reward
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol

            if time_step % update_timestep == 0:
                agent.update()
                
        actual_steps = max(time_step, 1)
        if actual_steps >= 35000:
            ep_energy_kwh = (ep_energy / 1e6) / 3.6
            avg_temp_violation = ep_comfort / actual_steps
            
            training_log.append({
                "Episode": ep, "Reward": ep_reward, 
                "Energy_kWh": ep_energy_kwh, "Avg_Temp_Dev": avg_temp_violation
            })
            print(f"🎯 Ep {ep:02d} | 单轮奖励: {ep_reward:8.2f} | 能耗: {ep_energy_kwh:8.2f}度 | 均偏: {avg_temp_violation:4.3f}℃")

            # ====== 公平的裁判系统 ======
            recent_rewards.append(ep_reward)
            if len(recent_rewards) > 5: 
                recent_rewards.pop(0) # 只看最近 5 轮的表现，过滤极端波动
            
            if ep >= 5:
                ma_reward = np.mean(recent_rewards)
                if ma_reward > best_ma_reward:
                    print(f"  🌟 [破纪录] 滑动平均奖励达到 {ma_reward:.2f}！保存最优权重。")
                    best_ma_reward = ma_reward
                    patience_counter = 0
                    # 只保存巅峰时刻的权重
                    torch.save(agent.policy.state_dict(), f"{save_dir}/{dcn_version}_best_weights.pth")
                else:
                    patience_counter += 1
                    print(f"  ⚠️ 未破纪录 (当前 MA: {ma_reward:.2f})。耐心值: {patience_counter}/{patience_limit}")
                
                if patience_counter >= patience_limit:
                    print(f"🛑 选手 DCN-{dcn_version} 潜力耗尽，触发早停！训练轮数：{ep}，最高 MA 成绩为: {best_ma_reward:.2f}")
                    break
    print(f"🛑 选手 DCN-{dcn_version} 训练结束，训练轮数：{ep}，最高 MA 成绩为: {best_ma_reward:.2f}")
    env.close()
    for f in glob.glob(f"{env_name}-res*"): shutil.rmtree(f, ignore_errors=True)
    
    with open(f"{save_dir}/training_log.json", 'w') as f:
        json.dump(training_log, f, indent=4)
        
    print(f"✅ {run_name} 测评圆满结束！")

if __name__ == "__main__":
    set_seed(42)
    versions = ['V1', 'V2']
    for v in versions:
        train_dcn_version(v)
        
    print("\n🎉 巅峰对决执行完毕，去检查 results/DCN_Ablation 里的最佳权重吧！")