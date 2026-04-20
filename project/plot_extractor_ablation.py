import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 绘图配置区
# ==========================================
VARIANTS = ["VANILLA", "CHANNEL", "GATE", "FULL"]

# 论文级别的渐进配色方案 (体现从基础到高级的进化)
COLORS = {
    "VANILLA": "#7f7f7f",  # 灰色 (基准)
    "CHANNEL": "#1f77b4",  # 蓝色 (初步结构)
    "GATE": "#ff7f0e",     # 橙色 (加入动态门控)
    "FULL": "#d62728"      # 红色 (完全体)
}

LABELS = {
    "VANILLA": "Vanilla GRU (17D Flat)",
    "CHANNEL": "+ Physical Channels",
    "GATE": "+ Attention Gate",
    "FULL": "Full (Gate + DCN + GRU)"
}

SMOOTH_WINDOW = 5

def smooth_curve(points, window=SMOOTH_WINDOW):
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window + 1)
        segment = points[start:i+1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed

def load_data():
    data = {}
    for var in VARIANTS:
        path = f"./results/Architecture_Ablation/Extractor_{var}/training_log.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                log = json.load(f)
                if log:
                    data[var] = {
                        "episodes": [item["Episode"] for item in log],
                        "rewards": [item["Reward"] for item in log],
                        "energies": [item["Energy_kWh"] for item in log],
                        "temp_devs": [item["Avg_Temp_Dev"] for item in log]
                    }
        else:
            print(f"⚠️ Warning: Data not found for {var} at {path}")
    return data

def plot_ablation():
    print("========== 📊 正在生成架构消融进化图 ==========")
    data = load_data()
    if not data:
        print("❌ 未加载到任何数据！")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # -----------------------------------------
    # 左图：训练奖励收敛曲线
    # -----------------------------------------
    for var in VARIANTS:
        if var in data:
            eps = data[var]["episodes"]
            rews_raw = data[var]["rewards"]
            rews_smooth = smooth_curve(rews_raw)
            
            ax1.plot(eps, rews_smooth, label=LABELS[var], color=COLORS[var], linewidth=2.5)
            ax1.plot(eps, rews_raw, color=COLORS[var], alpha=0.15, linewidth=1)
            ax1.scatter(eps[-1], rews_smooth[-1], color=COLORS[var], marker='*', s=100, zorder=5)

    ax1.set_title('Training Convergence (Reward over Episodes)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Valid Training Episodes', fontsize=12)
    ax1.set_ylabel('Total Reward (Smoothed) -> Higher is better', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=11)

    # -----------------------------------------
    # 右图：性能终局散点图 (帕累托进化轨迹)
    # -----------------------------------------
    final_temps = []
    final_energies = []
    valid_variants = []

    for var in VARIANTS:
        if var in data:
            # 取最后 5 轮的平均值作为该架构的最终真实水平，消除单轮波动
            temp = np.mean(data[var]["temp_devs"][-5:])
            energy = np.mean(data[var]["energies"][-5:])
            
            final_temps.append(temp)
            final_energies.append(energy)
            valid_variants.append(var)
            
            ax2.scatter(temp, energy, color=COLORS[var], s=250, edgecolor='black', zorder=4, label=LABELS[var])
            ax2.annotate(var, (temp, energy), xytext=(10, 5), textcoords='offset points', 
                         fontsize=10, fontweight='bold', color=COLORS[var])

    # 绘制进化轨迹箭头 (展现一步步变强的过程)
    for i in range(len(valid_variants) - 1):
        ax2.annotate('', xy=(final_temps[i+1], final_energies[i+1]), 
                     xytext=(final_temps[i], final_energies[i]),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--", alpha=0.7), zorder=2)

    ax2.set_title('Architecture Evolution (Energy vs. Comfort)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Avg Temp Deviation (C) -> Lower is better', fontsize=12)
    ax2.set_ylabel('Annual Energy Consumption (kWh) -> Lower is better', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 标注左下角的最优方向
    ax2.annotate('Optimal Direction', xy=(0.05, 0.05), xytext=(0.2, 0.2), xycoords='axes fraction', 
                 arrowprops=dict(facecolor='green', alpha=0.3, width=3, headwidth=10), 
                 color='green', alpha=0.7, fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = "./results/Architecture_Ablation/Architecture_Evolution_Plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 绘图完成！图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_ablation()