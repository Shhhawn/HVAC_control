import os
import shutil
import glob
import json
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import sinergym
from agent import PPOAgent
from gymnasium.wrappers import FrameStackObservation

# =====================================================================
# 🎛️ 全局配置区
# =====================================================================
# 请确保这里填入的是你之前帕累托实验中保存的、表现最好的那个权重！
BEST_MODEL_PATH = {
    "MLP": "./results/Ablation/Ablation_MLP/best_ppo_weights.pth", 
    "STACK": "./results/Ablation/Ablation_STACK/best_ppo_weights.pth", 
    "LSTM": "./results/Ablation/Ablation_LSTM/best_ppo_weights.pth", 
    "GRU": "./results/Ablation/Ablation_GRU/best_ppo_weights.pth", 
    "Transformer": "./results/Ablation/Ablation_TRANSFORMER/best_ppo_weights.pth"
}

SAVE_DIR = "./results/Ablation/Generalization"

ENV_HOT = "Eplus-5zone-hot-continuous-v1"     # 亚利桑那 (极热)
ENV_COOL = "Eplus-5zone-cool-continuous-v1"   # 华盛顿 (凉爽)
ENV_MIXED = "Eplus-5zone-mixed-continuous-v1" # 混合气候


# 提取 Sinergym 状态空间的物理上下限 (EnergyPlus 底层有默认的硬边界)
# 注意：这里的维度大小 (obs_dim) 必须跟你的实际状态维度匹配
def static_normalize(state, env):
    # 获取底层环境定义的物理上下边界
    high = env.observation_space.high
    low = env.observation_space.low
    
    # 防止除以 0 的极端情况，并替换掉无穷大的边界
    high = np.where(high > 1e10, 100.0, high) # 比如能耗没有上限，强行给个 100kW 的上限
    low = np.where(low < -1e10, -100.0, low)
    
    # 标准的 Min-Max 归一化，将所有物理量强行压缩到 [-1, 1] 之间
    norm_state = 2.0 * (state - low) / (high - low) - 1.0
    
    # 防止溢出
    return np.clip(norm_state, -5.0, 5.0)

# =====================================================================
# 🛠️ 第一重保险：落盘读写预检
# =====================================================================
def pre_flight_check():
    print("========== 🛫 执行起飞前系统预检 ==========")
    os.makedirs(SAVE_DIR, exist_ok=True)
    test_file = os.path.join(SAVE_DIR, ".test_write.tmp")
    try:
        # 测试写权限
        with open(test_file, 'w') as f:
            f.write("test")
        # 测试读权限
        with open(test_file, 'r') as f:
            _ = f.read()
        # 测试删除权限
        os.remove(test_file)
        print("✅ 硬盘读写权限测试通过！")
        
        # 测试模型文件是否存在
        for model_path in BEST_MODEL_PATH.values():
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到预训练权重：{model_path}")
            print("✅ 预训练权重文件确认存在！\n")
        
    except Exception as e:
        print(f"❌ 预检失败！请立刻检查报错，不要去睡觉！\n错误信息: {e}")
        exit(1)

# =====================================================================
# 🌍 阶段一：Zero-Shot 零样本泛化测试
# =====================================================================
def run_zero_shot(model, model_path):
    print(f"========== 🌍 阶段一：开始 {model} 的 Zero-Shot 泛化测试 ==========")
    test_envs = [ENV_HOT, ENV_COOL, ENV_MIXED]
    results = {}

    # 1. 核心修复：根据模型名称，映射对应的网络结构和时间窗口参数
    architecture_map = {
        "MLP": {"temporal_type": "mlp", "stack_size": 1},
        "STACK": {"temporal_type": "stack", "stack_size": 4},
        "LSTM": {"temporal_type": "lstm", "stack_size": 4},
        "GRU": {"temporal_type": "gru", "stack_size": 4},
        "Transformer": {"temporal_type": "transformer", "stack_size": 4}
    }
    
    arch_config = architecture_map[model]
    temp_type = arch_config["temporal_type"]
    stack_size = arch_config["stack_size"]

    for env_name in test_envs:
        print(f"\n🏢 正在测试环境: {env_name}, 使用模型架构: {model} (Stack: {stack_size})")
        
        # 2. 核心修复：根据模型需求，动态添加时间堆叠包装器
        base_env = gym.make(env_name)
        if stack_size > 1:
            env = FrameStackObservation(base_env, stack_size=stack_size)
        else:
            env = base_env
        
        # 注意：obs_dim 必须取 base_env 的原始维度 (17)
        obs_dim = base_env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        a_low = env.action_space.low
        a_high = env.action_space.high
        
        # 3. 核心修复：用正确的参数初始化包含特定架构的 Agent
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, 
                         temporal_type=temp_type, stack_size=stack_size)
        
        # 现在锁孔（网络）和钥匙（权重）匹配了，可以安全加载
        agent.policy.load_state_dict(torch.load(model_path, weights_only=True))
        agent.policy.eval() # 开启测试模式
        
        state, info = env.reset()
        ep_energy = 0
        ep_comfort = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 将 numpy array 送入静态归一化
            state_array = np.array(state)
            norm_state = static_normalize(state_array, base_env)
            
            action_tanh = agent.select_action(norm_state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)
            state = next_state
            
            ep_energy += info.get('total_power_demand', 0.0) * 900
            ep_comfort += info.get('total_temperature_violation', 0.0)

        ep_energy_kwh = (ep_energy / 1e6) / 3.6
        avg_temp_violation = ep_comfort / 35040.0
        
        print(f"✅ 测试完成 -> 模型： {model} | 能耗: {ep_energy_kwh:.2f} 度 | 均偏: {avg_temp_violation:.3f} ℃")
        
        results[env_name] = {
            "Energy_kWh": round(ep_energy_kwh, 2),
            "Avg_Temp_Dev_C": round(avg_temp_violation, 3)
        }
        
        env.close()
        cleanup_sinergym(env_name)

    # 将各模型的成绩独立保存
    with open(os.path.join(SAVE_DIR, f"{model}_zero_shot_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 阶段一 {model} 完成！数据已安全保存\n")

# =====================================================================
# 🌪️ 阶段二：多气候混合训练 (Domain Randomization)
# =====================================================================
def run_mixed_training(max_episodes=60):
    print("========== 🌪️ 阶段二：开始多气候混合训练 ==========")
    train_envs = [ENV_HOT, ENV_COOL, ENV_MIXED]
    
    # 获取维度信息 (三个环境维度一致，取一个即可)
    dummy_env = gym.make(ENV_HOT)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    a_low = dummy_env.action_space.low
    a_high = dummy_env.action_space.high
    dummy_env.close()
    
    # 重新初始化一个全新的 Agent，从头开始学习应对所有气候
    # (如果想基于预训练微调，可在这里 load_state_dict)
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=1e-4, gamma=0.99)
    update_timestep = 2000
    
    # 使用均衡型奖励权重
    w_t_linear, w_t_sq, w_e = 10.0, 2.0, 0.02 
    
    training_log = []

    for ep in range(1, max_episodes + 1):
        # 核心：每次 Episode 随机抽取一份“气候考卷”
        current_env_name = random.choice(train_envs)
        print(f"\n[Ep {ep:02d}/{max_episodes}] 🎲 抽取训练环境: {current_env_name}")
        
        env = gym.make(current_env_name)
        # env = NormalizeObservation(base_env)
        
        state, info = env.reset()
        time_step = 0
        ep_reward = 0
        ep_energy = 0
        ep_comfort = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            time_step += 1
            norm_state = static_normalize(state, env)
            action_tanh = agent.select_action(norm_state)
            # action_tanh = agent.select_action(state)
            action_env = a_low + (action_tanh + 1.0) * 0.5 * (a_high - a_low)
            action_env[0] = np.clip(action_env[0], 15.0, 22.0)
            action_env[1] = np.clip(action_env[1], 23.0, 27.0)

            deadband_penalty = -50.0 if (action_env[1] - action_env[0] < 2.0) else 0
            action_env = np.clip(action_env, a_low, a_high) 
            
            next_state, _, terminated, truncated, info = env.step(action_env)

            step_power_w = info.get('total_power_demand', 0.0) 
            step_temp_viol = info.get('total_temperature_violation', 0.0) 
            
            # 奖励计算
            temp_penalty = (w_t_linear * step_temp_viol) + (w_t_sq * (step_temp_viol ** 2))
            power_penalty = w_e * (step_power_w / 1000.0)
            reward = (- temp_penalty - power_penalty + deadband_penalty) / 100.0
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            ep_reward += reward
            ep_energy += step_power_w * 900
            ep_comfort += step_temp_viol

            if time_step % update_timestep == 0:
                agent.update()
                
        ep_energy_kwh = (ep_energy / 1e6) / 3.6
        actual_steps = max(time_step, 1)
        avg_temp_violation = ep_comfort / actual_steps
        
        print(f"🎯 结算[存活 {actual_steps} 步] -> 奖励: {ep_reward:8.2f} | 能耗: {ep_energy_kwh:8.2f} 度 | 均偏: {avg_temp_violation:4.3f} ℃")
        
        training_log.append({
            "Episode": ep,
            "Environment": current_env_name,
            "Reward": round(ep_reward, 2),
            "Energy_kWh": round(ep_energy_kwh, 2),
            "Avg_Temp_Dev": round(avg_temp_violation, 3)
        })
        
        env.close()
        cleanup_sinergym(current_env_name)
        
        # 【即时保存】每 5 轮保存一次 Checkpoint，防止意外中断
        if ep % 5 == 0 or ep == max_episodes:
            checkpoint_path = os.path.join(SAVE_DIR, f"mixed_ppo_ep{ep}.pth")
            torch.save(agent.policy.state_dict(), checkpoint_path)
            
            with open(os.path.join(SAVE_DIR, "mixed_training_log.json"), 'w') as f:
                json.dump(training_log, f, indent=4)
            print(f"💾 Checkpoint 已保存至 Ep {ep}")

    print("\n🎉 阶段二：多气候混合训练圆满完成！通用型 AI 诞生！")

# =====================================================================
# 🧹 工具函数：清扫战场
# =====================================================================
def cleanup_sinergym(env_name):
    # 扫描当前目录下该环境产生的所有 res 文件夹并强制删除
    for f in glob.glob(f"{env_name}-res*"):
        shutil.rmtree(f, ignore_errors=True)

if __name__ == "__main__":
    pre_flight_check()
    
    # 执行阶段一：Zero-Shot (大概需要 5-10 分钟)
    for model, model_path in BEST_MODEL_PATH.items():
        print(f'model: {model}, model_path: {model_path}')
        run_zero_shot(model, model_path)
    
    # 执行阶段二：混合训练 (跑 60 轮，由于是 3 个气候交替，相当于每个气候练 20 轮。大概需要几小时)
    # run_mixed_training(max_episodes=200)
    
    print("\n✅ 所有任务执行完毕！你可以安心查看 results/Overnight_Generalization 目录下的丰硕成果了！")