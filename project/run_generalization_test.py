import os
import numpy as np
import torch
import random
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import sinergym
from networks import HVACActorCritic

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def static_normalize(state_array, env):
    high = env.observation_space.high
    low = env.observation_space.low
    high = np.where(high > 1e10, 100.0, high) 
    low = np.where(low < -1e10, -100.0, low)
    norm_state = 2.0 * (state_array - low) / (high - low) - 1.0
    return np.clip(norm_state, -5.0, 5.0)

# 测试 RBC 在指定环境的表现
def test_rbc(env_name):
    env = gym.make(env_name)
    state, _ = env.reset(seed=42)
    ep_energy, ep_comfort, steps = 0, 0, 0
    terminated = False; truncated = False
    
    while not (terminated or truncated):
        steps += 1
        hour = int(state[3])
        if 8 <= hour <= 19:
            action = np.array([21.0, 24.0], dtype=np.float32)
        else:
            action = np.array([15.0, 27.0], dtype=np.float32)
            
        next_state, _, terminated, truncated, info = env.step(action)
        ep_energy += info.get('total_power_demand', 0.0) * 900
        ep_comfort += info.get('total_temperature_violation', 0.0)
        state = next_state
        
    env.close()
    return (ep_energy / 1e6) / 3.6, ep_comfort / max(steps, 1)

# 测试我们的 FULL 完全体网络
def test_our_model(env_name, weights_path):
    base_env = gym.make(env_name)
    env = FrameStackObservation(base_env, stack_size=4)
    obs_dim = base_env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    a_low, a_high = env.action_space.low, env.action_space.high
    
    # 实例化 FULL 完全体网络
    policy = HVACActorCritic(obs_dim, action_dim, temporal_type='gru', stack_size=4, extractor_type='full')
    policy.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    policy.eval() # 开启测试模式
    
    state, _ = env.reset(seed=42)
    ep_energy, ep_comfort, steps = 0, 0, 0
    terminated = False; truncated = False
    
    with torch.no_grad():
        while not (terminated or truncated):
            steps += 1
            norm_state = static_normalize(np.array(state), base_env)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0)
            
            # 零样本测试：直接取动作的均值 (mean)，不加入高斯探索噪声！
            action_mean, _, _, _ = policy(state_tensor)
            action_tanh = action_mean.squeeze(0).numpy()
            
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)
            ep_energy += info.get('total_power_demand', 0.0) * 900
            ep_comfort += info.get('total_temperature_violation', 0.0)
            state = next_state
            
    env.close()
    return (ep_energy / 1e6) / 3.6, ep_comfort / max(steps, 1)

if __name__ == "__main__":
    set_seed(42)
    
    # 填入你消融实验中跑出来的 FULL 完全体的权重路径
    # (确保该路径存在，如果是别的位置请自行修改)
    weights_path = "./results/SafeRL/SafePPO_FULL_Eps0.5/safe_ppo_full_weights.pth"
    
    # 三大考场：炎热、寒冷、混合
    environments = [
        "Eplus-5zone-hot-continuous-v1",
        "Eplus-5zone-cool-continuous-v1", 
        "Eplus-5zone-mixed-continuous-v1"
    ]
    
    print(f"\n{'='*60}")
    print("🌍 启动终极跨气候泛化大考 (Zero-Shot Generalization)")
    print(f"{'='*60}")
    
    for env_name in environments:
        climate = env_name.split('-')[2].upper()
        print(f"\n--- 正在测试气候: {climate} ---")
        
        # 跑 RBC 底线
        print("⏳ 正在计算 RBC 基准...")
        rbc_eng, rbc_comf = test_rbc(env_name)
        
        # 跑 你的 AI
        print("⏳ 正在测试 FULL 完全体 AI (Zero-Shot)...")
        ai_eng, ai_comf = test_our_model(env_name, weights_path)
        
        print(f"📊 成绩单 [{climate}]:")
        print(f"  [RBC] 能耗: {rbc_eng:8.2f} 度 | 均偏: {rbc_comf:4.3f} ℃")
        print(f"  [ AI] 能耗: {ai_eng:8.2f} 度 | 均偏: {ai_comf:4.3f} ℃")
        
        # 简单的表现评判
        eng_save = (rbc_eng - ai_eng) / rbc_eng * 100
        if ai_comf < rbc_comf and ai_eng < rbc_eng:
            print(f"  🏆 结论: 绝对碾压！既省电 ({eng_save:.1f}%)，又更舒服！")
        elif ai_comf < rbc_comf:
            print(f"  ✅ 结论: 舒适度提升，能耗属于帕累托置换范畴。")
        else:
            print(f"  ⚠️ 结论: 舒适度略逊，面临域偏移挑战。")