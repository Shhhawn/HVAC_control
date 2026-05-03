import os
import shutil
import glob
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import sinergym
import numpy as np
import torch
import random
from agent_safe import SafePPOAgent

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

# ==========================================
# 👑 新增：独立无噪声评估函数 (The Evaluation Phase)
# ==========================================
def evaluate_policy(agent, env_name, target_temp_dev):
    eval_base_env = gym.make(env_name)
    eval_env = FrameStackObservation(eval_base_env, stack_size=4)
    state, _ = eval_env.reset(seed=100) # 用不同的种子作为验证集
    
    ep_energy, ep_comfort, steps = 0, 0, 0
    terminated = False; truncated = False
    
    a_low, a_high = eval_env.action_space.low, eval_env.action_space.high
    agent.policy.eval() # 开启评估模式
    
    with torch.no_grad():
        while not (terminated or truncated):
            steps += 1
            norm_state = static_normalize(np.array(state), eval_base_env)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0)
            
            # ⚠️ 纯粹的推断：直接拿 Mean，不采样！
            action_mean, _, _, _ = agent.policy(state_tensor)
            action_tanh = action_mean.squeeze(0).numpy()
            
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            action_env = np.clip(action_env, a_low, a_high) 
            
            state, _, terminated, truncated, info = eval_env.step(action_env)
            ep_energy += info.get('total_power_demand', 0.0) * 900
            ep_comfort += info.get('total_temperature_violation', 0.0)
            
    eval_env.close()
    agent.policy.train() # 恢复训练模式
    
    final_energy_kwh = (ep_energy / 1e6) / 3.6
    final_temp_dev = ep_comfort / max(steps, 1)
    return final_energy_kwh, final_temp_dev

# ==========================================
# 主训练循环
# ==========================================
def train_safe_ppo(max_episodes=50):
    env_name = 'Eplus-5zone-hot-continuous-v1'
    run_name = "SafePPO_FULL_Eps0.5"
    
    set_seed(42)
    base_env = gym.make(env_name)
    env = FrameStackObservation(base_env, stack_size=4)
    
    obs_dim = base_env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    a_low, a_high = env.action_space.low, env.action_space.high
    
    agent = SafePPOAgent(obs_dim=obs_dim, action_dim=action_dim, 
                         temporal_type='gru', stack_size=4, extractor_type='full')
    update_timestep = 2000 
    
    save_dir = f"./results/SafeRL/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    final_weights_path = f"{save_dir}/safe_ppo_full_weights.pth"
    
    # 👑 早停与最优模型记录变量
    best_eval_energy = float('inf')
    patience_limit = 8 # 如果连续 8 次评估都没有破记录，则早停
    patience_counter = 0
    eval_interval = 3 # 每训练 3 轮评估一次
    
    print(f"\n========== 🚀 启动拉格朗日安全强化学习 ==========")
    print(f"目标：最小化能耗，同时严格保证全年平均温度偏差 <= {agent.target_temp_dev} ℃\n")
    
    for ep in range(1, max_episodes + 1):
        state, info = env.reset(seed=42+ep)
        ep_energy, ep_comfort, time_step = 0, 0, 0
        terminated = False; truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            norm_state = static_normalize(np.array(state), base_env)
            
            action_tanh = agent.select_action(norm_state) # 训练时带探索噪声
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            
            deadband_penalty = 50.0 if (action_env[1] - action_env[0] < 2.0) else 0.0
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            
            e_cost = (step_power_w / 1000.0) * 10.0 + deadband_penalty
            t_cost = step_temp_viol 
            
            agent.buffer.energy_costs.append(e_cost / 100.0)
            agent.buffer.temp_costs.append(t_cost)
            agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol

            if time_step % update_timestep == 0:
                agent.update()
                
        actual_steps = max(time_step, 1)
        if actual_steps >= 35000:
            avg_temp_violation = ep_comfort / actual_steps
            ep_energy_kwh = (ep_energy / 1e6) / 3.6
            current_lam = agent.lagrangian_multiplier
            print(f"🔄 [训练 Ep {ep:02d}] 罚款 Lambda: {current_lam:.4f} | 训练能耗: {ep_energy_kwh:8.2f}度 | 训练均偏: {avg_temp_violation:4.3f}℃")

        # ==========================================
        # 👑 独立评估与保存最好模型 (Validation & Save)
        # ==========================================
        if ep % eval_interval == 0:
            print(f"  🔍 正在进行无噪声严格评估...")
            eval_energy, eval_temp = evaluate_policy(agent, env_name, agent.target_temp_dev)
            print(f"  📊 [评估成绩] 能耗: {eval_energy:8.2f}度 | 均偏: {eval_temp:4.3f}℃")
            
            # 判断逻辑：1. 首先均偏必须达标 (给 0.01 的极小容忍度)；2. 然后能耗必须比历史最好成绩低
            if eval_temp <= agent.target_temp_dev + 0.01:
                if eval_energy < best_eval_energy:
                    print(f"  🌟 [破纪录] 发现新的合法最优策略！能耗从 {best_eval_energy if best_eval_energy!=float('inf') else 'inf'} 降至 {eval_energy:.2f}！保存权重。")
                    best_eval_energy = eval_energy
                    torch.save(agent.policy.state_dict(), final_weights_path)
                    patience_counter = 0 # 重置耐心
                else:
                    print(f"  ⚠️ 虽然安全达标，但不够省电。耐心值: {patience_counter+1}/{patience_limit}")
                    patience_counter += 1
            else:
                print(f"  ❌ 评估越界！策略不安全。耐心值: {patience_counter+1}/{patience_limit}")
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print(f"\n🛑 连续 {patience_limit} 次评估未找到更好的合法策略，触发早停！")
                break

    env.close()
    for f in glob.glob(f"{env_name}-res*"): shutil.rmtree(f, ignore_errors=True)
    print(f"✅ 训练与评估结束。最优权重位于: {final_weights_path}")

if __name__ == "__main__":
    train_safe_ppo()