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
    # 新增 extractor_type 参数
    def __init__(self, obs_dim=17, action_dim=2, temporal_type='gru', stack_size=4, extractor_type='full', dcn_version='V1'):
        super(HVACActorCritic, self).__init__()
        self.temporal_type = temporal_type
        self.stack_size = stack_size
        
        # 动态挂载特征提取器
        if extractor_type == 'vanilla':
            self.extractor = VanillaExtractor(obs_dim)
        elif extractor_type == 'channel':
            self.extractor = ChannelExtractor(obs_dim)
        elif extractor_type == 'gate':
            self.extractor = GateExtractor(obs_dim)
        else: # 'full'
            self.extractor = MultiChannelGateExtractor(obs_dim, DCN=dcn_version)
            
        extracted_dim = 128
        
        # ====== 时序处理器 (默认使用最强的 GRU) ======
        if temporal_type == 'gru':
            self.rnn = nn.GRU(input_size=extracted_dim, hidden_size=64, batch_first=True)
            final_dim = 64
        elif temporal_type == 'lstm':
            self.rnn = nn.LSTM(input_size=extracted_dim, hidden_size=64, batch_first=True)
            final_dim = 64
        else: # 'mlp'
            final_dim = extracted_dim

        # ====== Actor & Critic ======
        self.critic = nn.Sequential(nn.Linear(final_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.actor_mean = nn.Sequential(nn.Linear(final_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def forward(self, state):
        B, S, D = state.shape
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
        
        return action_mean, action_std, v_value, gate_w