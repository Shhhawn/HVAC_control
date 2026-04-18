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

# ==========================================
# 铁律：锁死随机性
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 静态物理归一化
def static_normalize(state_array, env):
    high = env.observation_space.high
    low = env.observation_space.low
    high = np.where(high > 1e10, 100.0, high) 
    low = np.where(low < -1e10, -100.0, low)
    norm_state = 2.0 * (state_array - low) / (high - low) - 1.0
    return np.clip(norm_state, -5.0, 5.0)

def train_pareto_point(weight_energy, max_episodes=40):
    env_name = "Eplus-5zone-hot-continuous-v1"
    run_name = f"Pareto_GRU_WE_{weight_energy}"
    save_dir = f"./results/Pareto/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 启动帕累托前沿搜索 | 当前能耗权重 (WE): {weight_energy}")
    print(f"{'='*60}")
    
    base_env = gym.make(env_name)
    env = FrameStackObservation(base_env, stack_size=4)
        
    obs_dim = base_env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]
    a_low, a_high = env.action_space.low, env.action_space.high
    
    # 使用 GRU 架构
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=1e-4, 
                     temporal_type='gru', stack_size=4)
    
    update_timestep = 2000 
    
    # 早停机制
    best_ma_reward = -float('inf')
    patience_limit = 10
    patience_counter = 0
    recent_rewards = []
    
    final_energy = None
    final_temp_dev = None

    for ep in range(1, max_episodes + 1):
        state, info = env.reset(seed=42+ep) # 确保每个 episode 初始化状态一致的扰动
        time_step = 0
        ep_reward, ep_energy, ep_comfort = 0, 0, 0
        terminated = False; truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            state_array = np.array(state)
            norm_state = static_normalize(state_array, base_env)
            
            action_tanh = agent.select_action(norm_state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            
            deadband_penalty = -50.0 if (action_env[1] - action_env[0] < 2.0) else 0
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            
            # 🔥 核心：动态调整的 Reward 公式
            temp_penalty = (10.0 * step_temp_viol) + (2.0 * (step_temp_viol ** 2))
            power_penalty = weight_energy * (step_power_w / 1000.0)
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
        
        # 只记录没有中途崩溃的有效数据
        if actual_steps >= 35000:
            ep_energy_kwh = (ep_energy / 1e6) / 3.6
            avg_temp_violation = ep_comfort / actual_steps
            
            print(f"🎯 Ep {ep:02d} | 奖励: {ep_reward:8.2f} | 能耗: {ep_energy_kwh:8.2f}度 | 均偏: {avg_temp_violation:4.3f}℃")

            # 早停逻辑判断
            recent_rewards.append(ep_reward)
            if len(recent_rewards) > 5: recent_rewards.pop(0)
            
            if ep >= 5:
                ma_reward = np.mean(recent_rewards)
                if ma_reward > best_ma_reward:
                    best_ma_reward = ma_reward
                    patience_counter = 0
                    final_energy = ep_energy_kwh
                    final_temp_dev = avg_temp_violation
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    print(f"🛑 触发早停机制，锁定最优成绩！")
                    break
        else:
            print(f"⚠️ Episode {ep} 未跑满，丢弃数据。")

    env.close()
    for f in glob.glob(f"{env_name}-res*"): shutil.rmtree(f, ignore_errors=True)
    
    # 如果全军覆没（物理引擎全部崩溃），返回极端惩罚值防止画图出错
    if final_energy is None:
        return 20000.0, 1.5 
    
    return round(final_energy, 2), round(final_temp_dev, 3)

if __name__ == "__main__":
    set_seed(42)
    
    # 对数量级网格搜索配置表
    # 0.0001: 极端舒适度优先，0.5: 极端省电优先
    # weight_candidates = [1.0, 5.0, 8.0, 12.0, 15.0, 18.0, 20.0, 50.0]
    # weight_candidates = [7.0, 12.0, 16.0, 17.0]
    weight_candidates = [6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 15.3, 15.6, 16.0, 17.0, 20.0, 50.0]
    pareto_results = {}

    # 读取已有结果
    file_path = './results/Pareto/pareto_frontier_data.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("文件损坏或格式错误，将创建新文件")
                results = {}
    print(f'results: {results}')
    
    for we in weight_candidates:
        energy, temp_dev = train_pareto_point(we)
        pareto_results[str(we)] = {
            "Energy_kWh": energy,
            "Avg_Temp_Dev_C": temp_dev
        }
        
    # 保存结果，用于后续画图
    results.update(pareto_results)

    os.makedirs("./results/Pareto", exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'pareto_results: {pareto_results}')
        
    print("\n🎉 所有网格搜索任务圆满结束！数据已保存至 ./results/Pareto/pareto_frontier_data.json")