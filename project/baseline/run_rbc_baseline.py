import os
import shutil
import glob
import numpy as np
import gymnasium as gym
import sinergym
import torch
import random

'''
🎯 RBC 最终成绩单 [存活 35040 步]
⚡ 年总能耗: 14695.96 kWh
🌡️ 平均温度偏差: 0.752 ℃
'''

# ==========================================
# 铁律：锁死随机数种子，保证实验的绝对可复现性
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_rbc_baseline(env_name="Eplus-5zone-hot-continuous-v1"):
    set_seed(42)
    
    print(f"\n{'='*50}")
    print(f"🏢 启动规则控制基准 (RBC) 测试: {env_name}")
    print(f"{'='*50}")
    
    # 初始化环境 (不需要 NormalizeObservation，因为这里不用神经网络)
    env = gym.make(env_name)
    state, info = env.reset(seed=42)
    
    ep_energy = 0
    ep_comfort = 0
    time_step = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        time_step += 1
        
        # Sinergym 5-zone 默认 observation 的 index 3 是当前小时 (hour, 0-23)
        hour = int(state[3])
        
        # 🧠 传统 HVAC 规则控制逻辑 (Rule-Based Control)
        if 8 <= hour <= 19:
            # 白天工作时间 (8:00 - 19:00)：收紧温度死区，优先保证员工舒适度
            action_env = np.array([21.0, 24.0], dtype=np.float32)
        else:
            # 夜间下班时间 (19:00 - 次日 8:00)：放宽温度死区，极限省电
            action_env = np.array([15.0, 27.0], dtype=np.float32)
            
        next_state, _, terminated, truncated, info = env.step(action_env)
        
        # 累加指标
        step_power_w = info.get('total_power_demand', 0.0) 
        step_temp_viol = info.get('total_temperature_violation', 0.0) 
        
        ep_energy += step_power_w * 900
        ep_comfort += step_temp_viol
        
        state = next_state

    # 结算
    actual_steps = max(time_step, 1)
    ep_energy_kwh = (ep_energy / 1e6) / 3.6
    avg_temp_violation = ep_comfort / actual_steps
    
    print(f"\n🎯 RBC 最终成绩单 [存活 {actual_steps} 步]")
    print(f"⚡ 年总能耗: {ep_energy_kwh:.2f} kWh")
    print(f"🌡️ 平均温度偏差: {avg_temp_violation:.3f} ℃")
    print(f"{'='*50}\n")
    
    env.close()
    
    # 清扫 Sinergym 产生的冗余文件夹
    for f in glob.glob(f"{env_name}-res*"):
        shutil.rmtree(f, ignore_errors=True)

if __name__ == "__main__":
    run_rbc_baseline()

