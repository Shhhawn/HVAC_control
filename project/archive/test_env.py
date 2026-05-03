import os
import sys

# ==========================================
# 核心修复：强制注入 EnergyPlus 安装路径
# 必须在 import sinergym 之前执行！
# ==========================================
EPLUS_PATH = "/usr/local/EnergyPlus-25-1-0"  # 我们第一步默认安装的路径
os.environ["EPLUS_PATH"] = EPLUS_PATH
sys.path.append(EPLUS_PATH)  # 把 EPlus 的原生 Python API 暴露给系统

import gymnasium as gym
import sinergym
import torch

def test_setup():
    print("="*40)
    print("1. 检查 PyTorch 与 GPU 加速")
    print("="*40)
    print(f"PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
        
    print("\n" + "="*40)
    print("2. 检查 Sinergym 与 EnergyPlus 引擎")
    print("="*40)
    
    # 获取注册表
    all_envs = list(gym.envs.registry.keys())
    sinergym_envs = [env for env in all_envs if 'eplus' in str(env).lower()]
    
    if not sinergym_envs:
        print(f"❌ 依然找不到环境！请检查 {EPLUS_PATH} 目录是否存在。")
        return
        
    # 寻找连续控制的 5 区办公室环境
    target_env_id = next((env for env in sinergym_envs if '5zone' in env.lower() and 'hot' in env.lower() and 'continuous' in env.lower()), None)
    
    print(f"🎯 底层自动匹配到真实环境名: {target_env_id}")
    
    try:
        env = gym.make(target_env_id)
        obs, info = env.reset()
        print(f"✅ 环境初始化成功！初始状态(Observation)维度: {obs.shape}")
        
        # 打印物理指标列表（方便后续我们做特征通道切分）
        print("\n=== 环境暴露的物理传感器指标 ===")
        variables = env.get_wrapper_attr('observation_variables')
        for idx, var_name in enumerate(variables): 
            print(f" 索引 [{idx}]: {var_name}")
        
        # 随机执行动作，跑 3 步测试
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            energy = info.get('energies', [info.get('total_energy', 0)])[0]
            print(f"Step {i+1} | 奖励: {reward:.2f} | 功率: {energy:.2f} W | 室内温度: {obs[1]:.2f} °C")
            
        env.close()
        print("\n✅ 环境交互循环测试彻底通过！")
    except Exception as e:
        print(f"❌ 环境交互失败，报错: {e}")

if __name__ == "__main__":
    test_setup()