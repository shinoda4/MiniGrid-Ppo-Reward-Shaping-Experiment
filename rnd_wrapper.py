import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# RND 使用的小型 CNN (适配 MiniGrid 7x7x3)
class RNDNetwork(nn.Module):
    def __init__(self, input_shape, output_dim=128):
        super().__init__()
        n_input_channels = input_shape[0] # Usually 3
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算 Flatten 后的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.cnn(dummy_input).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.linear(self.cnn(x))

class RNDWrapper(gym.Wrapper):
    def __init__(self, env, learning_rate=1e-4, intrinsic_weight=0.01, output_dim=128):
        super().__init__(env)
        self.intrinsic_weight = intrinsic_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取观测形状 (C, H, W)
        # 注意: ImgObsWrapper 输出通常是 (7, 7, 3), 也就是 (H, W, C)
        # 我们需要在 forward 时转置为 (C, H, W) 以适配 PyTorch
        self.obs_shape = (3, 7, 7) 
        
        # 1. Target Network (固定，不训练)
        self.target_net = RNDNetwork(self.obs_shape, output_dim).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        # 2. Predictor Network (训练)
        self.predictor_net = RNDNetwork(self.obs_shape, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=learning_rate)
        
    def _get_obs_tensor(self, obs):
        # 将 numpy (H, W, C) -> tensor (1, C, H, W)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs = obs.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- RND 核心逻辑 ---
        obs_tensor = self._get_obs_tensor(obs)
        
        with torch.no_grad():
            target_feature = self.target_net(obs_tensor)
            
        # 前向传播预测
        predictor_feature = self.predictor_net(obs_tensor)
        
        # 计算内在奖励 (MSE Error)
        loss = nn.MSELoss()(predictor_feature, target_feature)
        intrinsic_reward = loss.item()
        
        # 训练 Predictor (在线更新，每步都更)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 组合奖励
        # 注意：这里我们只缩放 intrinsic reward，并加到原始奖励上
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        
        # 在 info 中记录内在奖励，方便调试
        info['rnd_reward'] = intrinsic_reward
        
        return obs, total_reward, terminated, truncated, info


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from minigrid.wrappers import ImgObsWrapper
from rnd_wrapper import RNDWrapper

def verify_rnd_logic():
    print("🔍 正在验证 RND 逻辑 (Curiosity Check)...")
    
    # 初始化环境
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    # 权重设大一点以便观察
    env = RNDWrapper(env, learning_rate=0.001, intrinsic_weight=1.0) 
    
    obs, _ = env.reset(seed=42)
    
    intrinsic_rewards = []
    
    # 阶段 1: 呆在原地不动 (Same State) 100次
    # 我们不执行 env.step，而是手动喂同一个 observation 给 wrapper 的网络
    print("   阶段 1: 连续观察同一个状态 100 次 (预期：奖励下降)")
    
    obs_tensor = env._get_obs_tensor(obs)
    target_feat = env.target_net(obs_tensor) # Target 固定
    
    for i in range(100):
        # 手动训练循环
        pred_feat = env.predictor_net(obs_tensor)
        loss = env.predictor_net.parameters()
        
        # 计算当前 loss (reward)
        import torch.nn as nn
        loss_val = nn.MSELoss()(pred_feat, target_feat)
        intrinsic_rewards.append(loss_val.item())
        
        # 更新网络
        env.optimizer.zero_grad()
        loss_val.backward()
        env.optimizer.step()

    # 阶段 2: 突然换一个完全不同的状态 (New State)
    print("   阶段 2: 突然观测新状态 (预期：奖励暴涨)")
    
    # 模拟一个全黑或全白的新状态 (噪音)
    # 注意 MiniGrid 观测范围是 0-255，归一化通常在内部处理，这里直接模拟数值变化
    fake_obs = np.random.randint(0, 255, (7, 7, 3), dtype=np.uint8) 
    fake_tensor = env._get_obs_tensor(fake_obs)
    
    with torch.no_grad():
        t_feat = env.target_net(fake_tensor)
        p_feat = env.predictor_net(fake_tensor)
        new_reward = nn.MSELoss()(p_feat, t_feat).item()
        
    intrinsic_rewards.append(new_reward)
    
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.plot(intrinsic_rewards, marker='o')
    plt.axvline(x=99, color='r', linestyle='--', label="Switch State")
    plt.title("RND Intrinsic Reward Verification")
    plt.xlabel("Training Steps (on same state -> new state)")
    plt.ylabel("Intrinsic Reward (MSE Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig("verification_rnd.png")
    print("✅ 验证完成！请查看 verification_rnd.png")
    print(f"   初始奖励: {intrinsic_rewards[0]:.4f}")
    print(f"   第100次奖励: {intrinsic_rewards[99]:.4f} (应显著降低)")
    print(f"   新状态奖励: {intrinsic_rewards[100]:.4f} (应显著升高)")

if __name__ == "__main__":
    verify_rnd_logic()