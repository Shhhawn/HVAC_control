import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNCrossLayer(nn.Module):
    """
    手工实现 DCN (Deep & Cross Network) 的交叉层。
    用于显式捕获温湿度、气象环境与能耗之间的高阶非线性交叉特征。
    """
    def __init__(self, input_dim):
        super(DCNCrossLayer, self).__init__()
        # 交叉层的权重和偏置
        self.weight = nn.Parameter(torch.Tensor(input_dim, 1))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        
        # 参数初始化
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # 核心交叉公式: x_{l+1} = x_0 * (x_l * W_l) + b_l + x_l
        # xl_w 的维度将是 (batch_size, 1)，通过广播机制与 x0 相乘
        xl_w = torch.matmul(xl, self.weight)
        return x0 * xl_w + self.bias + xl

class MultiChannelGateExtractor(nn.Module):
    """
    基于物理机理的特征提取器：物理通道切分 + Gate注意力 + DCN融合
    """
    def __init__(self, obs_dim=17, channel_dim=64):
        super(MultiChannelGateExtractor, self).__init__()
        
        # ---------------------------------------------------------
        # 【物理通道切分配置】 (严格对齐 Eplus-5zone-hot-continuous-v1)
        # ---------------------------------------------------------
        # 通道 1: 温度感知通道 (4维)
        # [3]室外温度, [9]室内空气温度, [12]供暖设定点, [13]制冷设定点
        self.idx_t = [3, 9, 12, 13]  
        
        # 通道 2: 气象环境与扰动通道 (10维)
        # [0]月, [1]日, [2]时, [4]室外湿度, [5]风速, [6]风向, [7]散辐射, [8]直辐射, [10]室内湿度, [11]室内人数
        self.idx_h = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11] 
        
        # 通道 3: 能耗反馈通道 (3维)
        # [14]碳排放, [15]瞬时HVAC功率, [16]累计HVAC耗电量
        self.idx_e = [14, 15, 16] 
        
        # ---------------------------------------------------------
        # 1. 独立通道特征提取网络
        # ---------------------------------------------------------
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        
        # ---------------------------------------------------------
        # 2. Gate 注意力网络 (全局视角分配权重)
        # ---------------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1) # 输出 3 个权重，且和为 1
        )
        
        # ---------------------------------------------------------
        # 3. DCN 交叉特征融合
        # ---------------------------------------------------------
        self.cross_dim = channel_dim * 3
        self.cross_layer1 = DCNCrossLayer(self.cross_dim)
        self.cross_layer2 = DCNCrossLayer(self.cross_dim)
        
        # 最后降维输出给 Actor 和 Critic
        self.final_linear = nn.Linear(self.cross_dim, 128)
        
    def forward(self, state):
        # --- 数据切片 ---
        x_t = state[:, self.idx_t]
        x_h = state[:, self.idx_h]
        x_e = state[:, self.idx_e]
        
        # --- 独立通道特征提取 ---
        f_t = self.ch_temp(x_t)
        f_h = self.ch_humid(x_h)
        f_e = self.ch_energy(x_e)
        
        # --- Gate 网络生成动态权重 ---
        g_weights = self.gate(state) # shape: (batch, 3)
        w_t = g_weights[:, 0].unsqueeze(1) # shape: (batch, 1) 以便与特征相乘
        w_h = g_weights[:, 1].unsqueeze(1)
        w_e = g_weights[:, 2].unsqueeze(1)
        
        # --- 动态加权 (Attention) ---
        weighted_f_t = f_t * w_t
        weighted_f_h = f_h * w_h
        weighted_f_e = f_e * w_e
        
        # --- 拼接并进入 DCN 交叉层 ---
        x0 = torch.cat([weighted_f_t, weighted_f_h, weighted_f_e], dim=1)
        # x1 = self.cross_layer1(x0, x0)
        # x2 = self.cross_layer2(x0, x1)
        
        # --- 输出融合特征 ---
        out_features = F.relu(self.final_linear(x0))
        return out_features, g_weights

class HVACActorCritic(nn.Module):
    """
    完整的 PPO 智能体大脑
    包含 Actor (动作策略) 和 Critic (价值评估)
    """
    def __init__(self, obs_dim=17, action_dim=2):
        super(HVACActorCritic, self).__init__()
        
        # 挂载你的创新特征提取器
        self.extractor = MultiChannelGateExtractor(obs_dim)
        
        # Critic 网络: 评估当前状态有多好，输出 V 值
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Actor 网络: 输出动作均值
        # Sinergym 连续动作环境需要归一化到 [-1, 1] 的动作，所以最后一层用 Tanh
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )
        # 动作的方差(Log Std)，用于控制探索力度，设为可学习的独立参数
        # self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def forward(self, state):
        # 获取特征和 Gate 权重（权重可用于你写论文时画“注意力热力图”）
        features, gate_weights = self.extractor(state)
        
        v_value = self.critic(features)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        return action_mean, action_std, v_value, gate_weights

# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟环境吐出的一个 batch 的真实维度数据: (batch_size=4, obs_dim=17)
    dummy_state = torch.randn(4, 17) 
    
    # 初始化你的模型，假设动作维度为 2 (例如：供暖设定点调节、制冷设定点调节)
    model = HVACActorCritic(obs_dim=17, action_dim=2)
    
    mean, std, v, weights = model(dummy_state)
    
    print("✅ 网络架构编译与前向传播测试通过！")
    print(f"输入状态维度: {dummy_state.shape}")
    print(f"Actor 输出动作均值维度: {mean.shape} (应当为 batch_size x 2)")
    print(f"Critic 输出价值 V 维度: {v.shape} (应当为 batch_size x 1)")
    print(f"Gate 注意力权重维度: {weights.shape} (应当为 batch_size x 3通道)")
    
    # 简单查看一下第一个样本的权重分配
    print(f"示例 Gate 权重 (通道1-温度, 通道2-环境, 通道3-能耗): {weights[0].detach().numpy()}")