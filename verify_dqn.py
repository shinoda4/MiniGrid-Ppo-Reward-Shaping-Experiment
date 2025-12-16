import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 1. 必须复用之前的 CNN (否则图像维度会报错)
# ==========================================
class MiniGridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, normalized_image=False):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def verify_dqn_setup():
    print("🔍 正在验证 DQN 与 MiniGrid 的兼容性...")
    
    # 使用 5x5 快速验证
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # DQN 配置
    policy_kwargs = dict(
        features_extractor_class=MiniGridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    try:
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=1000, # 缩得很小只为测试
            learning_starts=100,
            batch_size=32,
            verbose=1
        )
        print("   ✅ DQN 模型构建成功 (CNN 维度匹配)")
        
        print("   ⏳ 尝试运行 200 步训练...")
        model.learn(total_timesteps=200)
        print("   ✅ DQN 训练循环测试通过 (Replay Buffer 正常)")
        
        # 简单测试预测
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"   ✅ 动作预测测试通过 (Action: {action})")
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    verify_dqn_setup()