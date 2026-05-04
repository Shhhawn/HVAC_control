import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNCrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(DCNCrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, 1))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        xl_w = torch.matmul(xl, self.weight)
        return x0 * xl_w + self.bias + xl

class DCNv2CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(DCNv2CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        
        # 使用 Xavier 初始化矩阵，防止训练初期梯度爆炸
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # 1. 矩阵乘法：(Batch, D) @ (D, D) -> (Batch, D)
        # 这对应公式里的 W_l * x_l
        xl_w = torch.matmul(xl, self.weight) 
        
        # 2. 加上 bias
        xl_w_b = xl_w + self.bias
        
        # 3. 哈达玛积（按位乘法）与残差连接
        # 这对应公式里的 x_0 ⊙ (W_l * x_l + b_l) + x_l
        return x0 * xl_w_b + xl
    
import torch
import torch.nn as nn
import torch.nn.functional as F
# 👑 引入神经常微分方程求解器
from torchdiffeq import odeint 

# ==========================================
# 👑 巅峰算子：连续时空向量场 (Vector Field of ODE-DCN)
# ==========================================
class ODEDCNFunc(nn.Module):
    def __init__(self, hidden_dim, feature_dim):
        super(ODEDCNFunc, self).__init__()
        self.hidden_dim = hidden_dim
        
        # ODE 内部的 DCN 投影矩阵 W
        # 负责计算状态 h(t) 与 外部物理输入 x_t 之间的偏导数关系
        self.dcn_weight = nn.Parameter(torch.Tensor(hidden_dim + feature_dim, hidden_dim))
        self.dcn_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        nn.init.xavier_uniform_(self.dcn_weight)
        # 极度关键：初始化为 0，让初始向量场极其平缓，防止初期积分发散！
        nn.init.zeros_(self.dcn_bias) 
        
        # 寄存器：用于在连续积分时，保持外部离散输入 x_t 的状态 (Zero-Order Hold)
        self.current_x = None 

    # 这就是物理上的 dh/dt
    def forward(self, t, h):
        if self.current_x is None:
            raise ValueError("积分前必须设置外部输入 current_x")
            
        # 将连续演化的历史 h(t) 与 恒定的外部输入 x_t 拼接
        hx = torch.cat([h, self.current_x], dim=-1)
        
        # 物理导数场计算：dh/dt = h(t) ⊙ (W * [h(t), x_t] + b)
        proj = torch.matmul(hx, self.dcn_weight) + self.dcn_bias
        dh_dt = h * proj 
        
        return dh_dt

# ==========================================
# 🧠 引擎盖重构：连续积分提取器
# ==========================================
class ODE_CoupledExtractor(nn.Module):
    def __init__(self, obs_dim=17, channel_dim=64, hidden_dim=64):
        super(ODE_CoupledExtractor, self).__init__()
        # 真实的物理通道索引
        self.idx_t = [3, 9, 12, 13] 
        self.idx_h = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11] 
        self.idx_e = [14, 15, 16] 
        
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(obs_dim, 3), nn.Softmax(dim=-1))
        
        self.cross_dim = channel_dim * 3
        self.hidden_dim = hidden_dim
        
        # 实例化 ODE 导数函数
        self.ode_func = ODEDCNFunc(hidden_dim=self.hidden_dim, feature_dim=self.cross_dim)

    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        # 积分的时间区间 [0, 1] 代表一个离散的 15 分钟采样周期
        integration_time = torch.tensor([0.0, 1.0], device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            f_t = self.ch_temp(x_t[:, self.idx_t])
            f_h = self.ch_humid(x_t[:, self.idx_h])
            f_e = self.ch_energy(x_t[:, self.idx_e])
            gate_weights = self.gate(x_t)
            w_t, w_h, w_e = gate_weights[:, 0].unsqueeze(1), gate_weights[:, 1].unsqueeze(1), gate_weights[:, 2].unsqueeze(1)
            
            x0 = torch.cat([f_t * w_t, f_h * w_h, f_e * w_e], dim=1)
            
            # 👑 核心魔法：将 x0 锁定在寄存器中
            self.ode_func.current_x = x0
            
            # 使用四阶龙格-库塔法 (rk4) 进行连续时间积分！
            # out 的形状是 (len(integration_time), batch, hidden_dim)
            out = odeint(self.ode_func, h_t, integration_time, method='rk4')
            
            # 取出 t=1.0 时刻（积分终点）的状态，作为下一个周期的起点
            h_t = out[-1]
            
        return h_t, gate_weights
    
"""
# ==========================================
# 核心算子：ST-DCN (时空融合深度交叉层)
# ==========================================
class STDCNv2CrossLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(STDCNv2CrossLayer, self).__init__()
        # 空间交叉权重 W_l
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # 时空投影权重 U_l (将隐藏状态投射到交叉维度)
        self.u_weight = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        
        nn.init.xavier_uniform_(self.weight)
        # 将历史调节项初始化为0，确保早期训练稳定性
        nn.init.zeros_(self.u_weight) 
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl, h_prev):
        # 1. 计算空间交叉项: W_l * x_l
        xl_w = torch.matmul(xl, self.weight) 
        # 2. 计算历史记忆干预项: U_l * h_{t-1}
        h_u = torch.matmul(h_prev, self.u_weight)
        
        # 3. 理论公式：x_{l+1} = x_0 ⊙ (W_l * x_l + U_l * h_{t-1} + b_l) + x_l
        xl_w_h_b = xl_w + h_u + self.bias
        return x0 * xl_w_h_b + xl

# ==========================================
# Extractor重构
# ==========================================
class ST_CoupledExtractor(nn.Module):
    def __init__(self, obs_dim=17, channel_dim=64, hidden_dim=64): # 修正: hidden_dim 改为 64，对接下游网络
        super(ST_CoupledExtractor, self).__init__()
        
        # 对齐真实的 Sinergym 物理通道索引
        self.idx_t = [3, 9, 12, 13] 
        self.idx_h = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11] 
        self.idx_e = [14, 15, 16] 
        
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        
        self.gate = nn.Sequential(nn.Linear(obs_dim, 3), nn.Softmax(dim=-1))
        
        self.cross_dim = channel_dim * 3
        self.hidden_dim = hidden_dim
        
        self.st_cross = STDCNv2CrossLayer(self.cross_dim, self.hidden_dim)
        self.gru_cell = nn.GRUCell(self.cross_dim, self.hidden_dim)

    def forward(self, x_seq):
        # 这里的 x_seq 原封不动保持 3D 形状: (Batch, StackSize, ObsDim)
        batch_size, seq_len, _ = x_seq.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            f_t = self.ch_temp(x_t[:, self.idx_t])
            f_h = self.ch_humid(x_t[:, self.idx_h])
            f_e = self.ch_energy(x_t[:, self.idx_e])
            
            gate_weights = self.gate(x_t)
            w_t = gate_weights[:, 0].unsqueeze(1)
            w_h = gate_weights[:, 1].unsqueeze(1)
            w_e = gate_weights[:, 2].unsqueeze(1)
            
            x0 = torch.cat([f_t * w_t, f_h * w_h, f_e * w_e], dim=1)
            
            x_cross = self.st_cross(x0, x0, h_t)
            h_t = self.gru_cell(x_cross, h_t)
            
        return h_t, gate_weights
"""

# 1. 纯小白提取器 (Vanilla): 什么物理机理都不懂，直接全连接
class VanillaExtractor(nn.Module):
    def __init__(self, obs_dim=17, out_dim=128):
        super(VanillaExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
    def forward(self, state):
        # 为了兼容统一接口，返回一个全为 1 的假 Gate 权重
        dummy_gate = torch.ones(state.shape[0], 3).to(state.device) / 3.0
        return self.net(state), dummy_gate

# 2. 物理分通道提取器 (Channel Only): 只分通道，没有 Gate 和 DCN
class ChannelExtractor(nn.Module):
    def __init__(self, obs_dim=17, channel_dim=64):
        super(ChannelExtractor, self).__init__()
        self.idx_t, self.idx_h, self.idx_e = [3, 9, 12, 13], [0, 1, 2, 4, 5, 6, 7, 8, 10, 11], [14, 15, 16] 
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        self.final_linear = nn.Linear(channel_dim * 3, 128)
        
    def forward(self, state):
        f_t = self.ch_temp(state[:, self.idx_t])
        f_h = self.ch_humid(state[:, self.idx_h])
        f_e = self.ch_energy(state[:, self.idx_e])
        x0 = torch.cat([f_t, f_h, f_e], dim=1)
        dummy_gate = torch.ones(state.shape[0], 3).to(state.device) / 3.0
        return F.relu(self.final_linear(x0)), dummy_gate

# 3. 门控分通道提取器 (Gate + Channel): 有分通道和注意力，没有 DCN
class GateExtractor(nn.Module):
    def __init__(self, obs_dim=17, channel_dim=64):
        super(GateExtractor, self).__init__()
        self.idx_t, self.idx_h, self.idx_e = [3, 9, 12, 13], [0, 1, 2, 4, 5, 6, 7, 8, 10, 11], [14, 15, 16] 
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))
        self.final_linear = nn.Linear(channel_dim * 3, 128)
        
    def forward(self, state):
        f_t, f_h, f_e = self.ch_temp(state[:, self.idx_t]), self.ch_humid(state[:, self.idx_h]), self.ch_energy(state[:, self.idx_e])
        g_weights = self.gate(state)
        w_t, w_h, w_e = g_weights[:, 0].unsqueeze(1), g_weights[:, 1].unsqueeze(1), g_weights[:, 2].unsqueeze(1)
        x0 = torch.cat([f_t * w_t, f_h * w_h, f_e * w_e], dim=1)
        return F.relu(self.final_linear(x0)), g_weights

# 4. 最终完全体 (Gate + Channel + DCN)
class MultiChannelGateExtractor(nn.Module):
    def __init__(self, obs_dim=17, channel_dim=64, dcn_version='V1'):
        super(MultiChannelGateExtractor, self).__init__()
        self.idx_t, self.idx_h, self.idx_e = [3, 9, 12, 13], [0, 1, 2, 4, 5, 6, 7, 8, 10, 11], [14, 15, 16] 
        self.ch_temp = nn.Sequential(nn.Linear(len(self.idx_t), channel_dim), nn.ReLU())
        self.ch_humid = nn.Sequential(nn.Linear(len(self.idx_h), channel_dim), nn.ReLU())
        self.ch_energy = nn.Sequential(nn.Linear(len(self.idx_e), channel_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))
        self.cross_dim = channel_dim * 3
        if dcn_version == 'V2':
            self.cross_layer1 = DCNv2CrossLayer(self.cross_dim)
            self.cross_layer2 = DCNv2CrossLayer(self.cross_dim)
        else:
            self.cross_layer1 = DCNCrossLayer(self.cross_dim)
            self.cross_layer2 = DCNCrossLayer(self.cross_dim)
        self.final_linear = nn.Linear(self.cross_dim, 128)
        
    def forward(self, state):
        f_t, f_h, f_e = self.ch_temp(state[:, self.idx_t]), self.ch_humid(state[:, self.idx_h]), self.ch_energy(state[:, self.idx_e])
        g_weights = self.gate(state)
        w_t, w_h, w_e = g_weights[:, 0].unsqueeze(1), g_weights[:, 1].unsqueeze(1), g_weights[:, 2].unsqueeze(1)
        x0 = torch.cat([f_t * w_t, f_h * w_h, f_e * w_e], dim=1)
        x1 = self.cross_layer1(x0, x0)
        x2 = self.cross_layer2(x0, x1)
        return F.relu(self.final_linear(x2)), g_weights
    



# ----------------- 升级版 ActorCritic -----------------
class HVACActorCritic(nn.Module):
    def __init__(self, obs_dim=17, action_dim=2, temporal_type='gru', stack_size=4, extractor_type='full', dcn_version='V1'):
        super(HVACActorCritic, self).__init__()
        self.temporal_type = temporal_type
        self.stack_size = stack_size
        self.extractor_type = extractor_type # 🛡️ 必须存为类的属性，forward里才能用！
        
        extracted_dim = 128
        final_dim = 64 # 后续 Actor/Critic 网络接受的标准维度
        
        # ====== 动态挂载特征提取器 & 时序处理器 ======
        if extractor_type == 'vanilla':
            self.extractor = VanillaExtractor(obs_dim)
        elif extractor_type == 'channel':
            self.extractor = ChannelExtractor(obs_dim)
        elif extractor_type == 'gate':
            self.extractor = GateExtractor(obs_dim)
        # elif extractor_type == 'st_coupled':
        #     # ST 架构自带内部 GRU，直接输出 final_dim (64)
        #     self.extractor = ST_CoupledExtractor(obs_dim=obs_dim, channel_dim=64, hidden_dim=final_dim)
        #     self.rnn = nn.Identity() # 置空外部的 GRU
        elif extractor_type == 'ode_coupled':
            self.extractor = ODE_CoupledExtractor(obs_dim=obs_dim, channel_dim=64, hidden_dim=final_dim)
            self.rnn = nn.Identity()
        else: # 'full'
            self.extractor = MultiChannelGateExtractor(obs_dim, dcn_version=dcn_version)

        # 传统的级联架构需要外部 GRU
        if extractor_type != 'st_coupled':
            if temporal_type == 'gru':
                self.rnn = nn.GRU(input_size=extracted_dim, hidden_size=final_dim, batch_first=True)
            elif temporal_type == 'lstm':
                self.rnn = nn.LSTM(input_size=extracted_dim, hidden_size=final_dim, batch_first=True)
            else: # 'mlp'
                self.rnn = nn.Identity()
                final_dim = extracted_dim

        # ====== Actor & Critic ======
        self.critic = nn.Sequential(nn.Linear(final_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.actor_mean = nn.Sequential(nn.Linear(final_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))
        
        # 新增：物理动力学预测头 (Dynamics Prediction Head)
        # 输入：提取出的高维特征 (final_dim) + 当前执行的动作 (action_dim)
        # 输出：预测下一时刻的物理状态 (假设环境堆叠展开后是 stack_size * obs_dim)
        self.dynamics_head = nn.Sequential(
            nn.Linear(final_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, stack_size * obs_dim) 
        )
    
    
    # 👑 新增一个专属方法，供 PPO 更新时计算辅助 Loss 使用
    def evaluate(self, state, action):
        B, S, D = state.shape
        
        if self.extractor_type in ['st_coupled', 'ode_coupled']:
            x, gate_w = self.extractor(state)
        else:
            state_flat = state.view(B * S, D)
            features_flat, g_weights_flat = self.extractor(state_flat)
            features = features_flat.view(B, S, -1)
            g_weights = g_weights_flat.view(B, S, 3)
            if self.temporal_type in ['lstm', 'gru']:
                rnn_out, _ = self.rnn(features)
                x = rnn_out[:, -1, :] 
                gate_w = g_weights[:, -1, :]
            else:
                x = features.squeeze(1)
                gate_w = g_weights.squeeze(1)

        v_value = self.critic(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        from torch.distributions import Normal
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        # 👑 物理预测逻辑：传入当前特征 x 和 动作 action，预测未来的环境状态
        # 注意：这里的 action 是没有被 tanh 缩放的原始网络输出动作
        pred_next_state = self.dynamics_head(torch.cat([x, action], dim=-1))
        
        return action_logprobs, torch.squeeze(v_value), dist_entropy, pred_next_state


    def forward(self, state):
        B, S, D = state.shape
        
        # 逻辑分流：时空一体 (ST/ODE) vs 时空解耦 (级联)
        # 将 ode_coupled 和 st_coupled 归为同一类处理逻辑
        if self.extractor_type in ['st_coupled', 'ode_coupled']:
            # 方案 A：时空量子纠缠 / 连续时间积分 架构
            # 它们内部自带循环或ODE求解器，直接把完整的 3D 序列塞进去，吐出最后时刻的物理记忆
            x, gate_w = self.extractor(state)
        else:
            # 方案 B：经典级联架构 (Cascade/FULL)
            # 必须先拍平，空间交叉完，再恢复成 3D 序列给外部的 GRU 处理
            state_flat = state.view(B * S, D)
            features_flat, g_weights_flat = self.extractor(state_flat)
            
            features = features_flat.view(B, S, -1)
            g_weights = g_weights_flat.view(B, S, 3)
            
            if self.temporal_type in ['lstm', 'gru']:
                rnn_out, _ = self.rnn(features)
                x = rnn_out[:, -1, :] # 只取最后一个时间步的隐状态
                gate_w = g_weights[:, -1, :] 
            else: 
                x = features.squeeze(1) 
                gate_w = g_weights.squeeze(1)
                
        # --- 决策下发 ---
        v_value = self.critic(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        return action_mean, action_std, v_value, gate_w