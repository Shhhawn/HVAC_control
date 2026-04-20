import os
import shutil
import glob
import json
import numpy as np
import torch
import random
import gymnasium as gym
import sinergym
from agent_sac import SACAgent

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

def run_sac_baseline(max_episodes=50):
    env_name = "Eplus-5zone-hot-continuous-v1"
    run_name = "Baseline_SAC_Vanilla"
    save_dir = f"./results/Baselines/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🥊 启动外部强敌基线测试: {run_name}")
    print(f"{'='*60}")
    
    # 注意：SAC 作为外部 Baseline，不加 FrameStack，仅用 17 维原生数据
    env = gym.make(env_name)
        
    obs_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]
    a_low, a_high = env.action_space.low, env.action_space.high
    
    # 实例化手写的 SAC Agent
    agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim)
    
    training_log = []
    best_ma_reward = -float('inf')
    patience_limit = 10
    patience_counter = 0
    recent_rewards = []

    # SAC 预热：让它先随机在环境里走几千步，填满 Buffer
    print("⏳ 正在预填充经验回放池 (Warm-up)...")
    state, _ = env.reset(seed=42)
    for _ in range(5000):
        action_env = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action_env)
        agent.buffer.push(static_normalize(np.array(state), env), 
                          action_env, 0, # 这里只存状态，reward 先给 0 无所谓
                          static_normalize(np.array(next_state), env), 
                          terminated or truncated)
        state = next_state if not (terminated or truncated) else env.reset()[0]
    print("✅ 预填充完毕，正式开始训练！")

    for ep in range(1, max_episodes + 1):
        state, info = env.reset(seed=42+ep) 
        time_step = 0
        ep_reward, ep_energy, ep_comfort = 0, 0, 0
        terminated = False; truncated = False
        
        norm_state = static_normalize(np.array(state), env)
        
        while not (terminated or truncated):
            time_step += 1
            
            action_tanh = agent.select_action(norm_state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            
            deadband_penalty = -50.0 if (action_env[1] - action_env[0] < 2.0) else 0
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)
            norm_next_state = static_normalize(np.array(next_state), env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            
            # 🔥 铁律：奖励权重死死锁在 10.0
            temp_penalty = (10.0 * step_temp_viol) + (2.0 * (step_temp_viol ** 2))
            power_penalty = 10.0 * (step_power_w / 1000.0)
            reward = (- temp_penalty - power_penalty + deadband_penalty) / 100.0
            
            # Off-policy 存储经验
            agent.buffer.push(norm_state, action_tanh, reward, norm_next_state, terminated or truncated)
            
            # 每一步都更新网络 (Off-policy 的特征)
            agent.update()
            
            norm_state = norm_next_state
            ep_reward += reward
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol
                
        actual_steps = max(time_step, 1)
        
        if actual_steps >= 35000:
            ep_energy_kwh = (ep_energy / 1e6) / 3.6
            avg_temp_violation = ep_comfort / actual_steps
            
            training_log.append({
                "Episode": ep, "Reward": ep_reward, 
                "Energy_kWh": ep_energy_kwh, "Avg_Temp_Dev": avg_temp_violation
            })
            print(f"🎯 Ep {ep:02d} | 奖励: {ep_reward:8.2f} | 能耗: {ep_energy_kwh:8.2f}度 | 均偏: {avg_temp_violation:4.3f}℃")

            recent_rewards.append(ep_reward)
            if len(recent_rewards) > 5: recent_rewards.pop(0)
            
            if ep >= 5:
                ma_reward = np.mean(recent_rewards)
                if ma_reward > best_ma_reward:
                    best_ma_reward = ma_reward
                    patience_counter = 0
                    torch.save(agent.actor.state_dict(), f"{save_dir}/sac_actor_weights.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    print(f"🛑 触发早停机制。")
                    break

    env.close()
    for f in glob.glob(f"{env_name}-res*"): shutil.rmtree(f, ignore_errors=True)
    
    with open(f"{save_dir}/training_log.json", 'w') as f:
        json.dump(training_log, f, indent=4)
        
    print(f"✅ {run_name} 训练完成！")

if __name__ == "__main__":
    set_seed(42)
    run_sac_baseline()