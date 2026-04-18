import gymnasium as gym
import sinergym
import numpy as np

def run_rule_based_baseline(env_name='Eplus-5zone-hot-continuous-v1', episodes=1):
    """
    运行基于规则的基准测试 (Rule-Based Controller)
    策略：全年采用固定的供暖和制冷设定点，模拟最传统的中央空调管理方式。
    """
    env = gym.make(env_name)
    print(f"========== 开始运行基准模型 (Baseline) ==========")
    print(f"环境: {env_name}")
    
    # 获取动作空间的真实物理边界 (例如: 供暖 15~22.5℃, 制冷 22.5~30℃)
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"动作空间边界: Low={action_low}, High={action_high}")

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        ep_reward = 0
        ep_energy = 0
        ep_comfort = 0
        step = 0
        
        while not (terminated or truncated):
            # 制定传统规则：供暖设定为 20℃，制冷设定为 24℃
            # 注意：必须确保动作在环境允许的边界内
            rbc_action = np.array([20.0, 24.0], dtype=np.float32)
            rbc_action = np.clip(rbc_action, action_low, action_high)
            
            obs, reward, terminated, truncated, info = env.step(rbc_action)
            
            ep_reward += reward
            # 累加能耗 (取自 info 字典)
            # ep_energy += info.get('energies', [info.get('total_energy', 0)])[0]
            current_power = info.get('total_power_demand', 0)
            ep_energy += current_power * 900
            ep_comfort += info.get('total_temperature_violation', 0)
            step += 1
            
        avg_temp = ep_comfort / 35040.0
        ep_energy_kwh = (ep_energy / 1e6) / 3.6
        print(f"Baseline Episode {ep+1} 结束 | 总步数: {step} | 累计奖励: {ep_energy_kwh:.2f} | 能耗: {ep_energy_kwh:8.2f} 度 | 温度超标总计: {ep_comfort:8.2f} ℃ | 单步平均偏离: {avg_temp:4.3f} ℃")
        
    env.close()

if __name__ == "__main__":
    run_rule_based_baseline()