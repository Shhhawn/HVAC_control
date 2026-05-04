# ===============================
# 用于比较 Gate + DCN-V1 + GRU 和 ODE + DCN 的效果
# seed = 42: 
# # Gate + DCN-V1 + GRU: -7617.40
# # ODE + DCN: -8710.11
# ===============================


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

def evaluate_architecture(agent, env_name):
    eval_base_env = gym.make(env_name)
    eval_env = FrameStackObservation(eval_base_env, stack_size=4)
    state, _ = eval_env.reset(seed=100) 
    
    ep_reward, ep_energy, ep_comfort, steps = 0, 0, 0, 0
    terminated = False; truncated = False
    a_low, a_high = eval_env.action_space.low, eval_env.action_space.high
    agent.policy.eval() 
    
    with torch.no_grad():
        while not (terminated or truncated):
            steps += 1
            norm_state = static_normalize(np.array(state), eval_base_env)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0)
            
            action_mean, _, _, _ = agent.policy(state_tensor)
            action_tanh = action_mean.squeeze(0).numpy()
            
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            deadband_penalty = -50.0 if (action_env[1] - action_env[0] < 2.0) else 0.0
            action_env = np.clip(action_env, a_low, a_high) 
            
            state, _, terminated, truncated, info = eval_env.step(action_env)
            
            step_power_w = info.get('total_power_demand', 0.0)
            step_temp_viol = info.get('total_temperature_violation', 0.0)
            
            temp_penalty = (10.0 * step_temp_viol) + (2.0 * (step_temp_viol ** 2))
            power_penalty = 10.0 * (step_power_w / 1000.0)
            pure_reward = (- temp_penalty - power_penalty + deadband_penalty) / 100.0
            
            ep_reward += pure_reward
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol
            
    eval_env.close()
    agent.policy.train() 
    return ep_reward, (ep_energy / 1e6) / 3.6, ep_comfort / max(steps, 1)

def train_architecture(architecture_type, max_episodes=50):
    env_name = "Eplus-5zone-hot-continuous-v1"
    run_name = f"ODE_Ablation_{architecture_type}"
    save_dir = f"./results/Architecture_Ablation/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🥊 连续物理世界的大考 | 参赛模型: {architecture_type.upper()}")
    if architecture_type == 'full':
        print("   [时空观] 离散跳跃: DCN交叉 + 离散GRU")
    else:
        print("   [时空观] 连续积分: ODE求解器 + 连续向量场 (ODE-DCN)")
    print(f"{'='*70}")
    
    base_env = gym.make(env_name)
    env = FrameStackObservation(base_env, stack_size=4)
        
    obs_dim = base_env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, 
                     temporal_type='gru', stack_size=4, 
                     extractor_type=architecture_type) 
    
    update_timestep = 2000 
    training_log = []
    
    best_eval_reward = -float('inf')
    patience_limit = 10 
    patience_counter = 0
    eval_interval = 1 

    for ep in range(1, max_episodes + 1):
        state, info = env.reset(seed=42+ep) 
        time_step, ep_reward = 0, 0
        terminated = False; truncated = False
        a_low, a_high = env.action_space.low, env.action_space.high
        
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
            
            reward = (- (10.0 * step_temp_viol + 2.0 * step_temp_viol**2) - 10.0 * (step_power_w / 1000.0) + deadband_penalty) / 100.0
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(terminated or truncated)
            state = next_state
            ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()
                
        actual_steps = max(time_step, 1)
        if actual_steps >= 35000:
            print(f"🔄 [训练 Ep {ep:02d}] 带噪声得分: {ep_reward:8.2f}")

        # 👑 严格推断裁判系统
        if ep % eval_interval == 0:
            print(f"  🔍 正在进行纯净推断评估 (ODE积分将进行极其精确的推演)...")
            eval_reward, eval_energy, eval_temp = evaluate_architecture(agent, env_name)
            
            training_log.append({"Episode": ep, "Eval_Reward": eval_reward, "Energy_kWh": eval_energy, "Avg_Temp_Dev": eval_temp})
            print(f"  📊 [裁判打分] 纯净奖励: {eval_reward:8.2f} | 能耗: {eval_energy:8.2f}度 | 均偏: {eval_temp:4.3f}℃")
            
            if eval_reward > best_eval_reward:
                print(f"  🌟 [破纪录] 得分飙升至 {eval_reward:.2f}！保存神级权重。")
                best_eval_reward = eval_reward
                patience_counter = 0
                torch.save(agent.policy.state_dict(), f"{save_dir}/best_weights.pth")
            else:
                patience_counter += 1
                print(f"  ⚠️ 未破纪录。最高纯净得分: {best_eval_reward:.2f}，耐心值: {patience_counter}/{patience_limit}")
            
            if patience_counter >= patience_limit:
                print(f"🛑 模型收敛，触发早停！训练轮数: {ep}，最高纯净得分: {best_eval_reward:.2f}")
                break
    
    print(f"🛑 训练结束！最高纯净得分: {best_eval_reward:.2f}")
    env.close()
    for f in glob.glob(f"{env_name}-res*"): shutil.rmtree(f, ignore_errors=True)
    with open(f"{save_dir}/eval_log.json", 'w') as f: json.dump(training_log, f, indent=4)

if __name__ == "__main__":
    set_seed(42)
    # 对比 1: 离散流派
    train_architecture('full')
    # 对比 2: 连续流派 (Neural ODEs)
    train_architecture('ode_coupled')