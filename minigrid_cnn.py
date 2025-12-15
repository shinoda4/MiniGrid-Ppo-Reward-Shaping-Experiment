import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# MiniGrid 7x7 观测的通道数是 3 (RGB 或 One-hot 编码，通过 ImgObsWrapper 转换后是 3)
# 图像尺寸是 7x7
MINIGRID_INPUT_CHANNELS = 3 
MINIGRID_OBS_SIZE = 7 

class MiniGridFeaturesExtractor(BaseFeaturesExtractor):
    """
    定制的特征提取器，用于处理 MiniGrid 的 7x7 图像观测。
    适配 (7, 7, 3) 的原始输入，并在内部处理为 (3, 7, 7) 的卷积。
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # 1. 自动检测通道数
        # MiniGrid 通常是 (H, W, C) -> (7, 7, 3)
        if observation_space.shape[-1] == 3:
            n_input_channels = observation_space.shape[-1] # 取最后一个维度 (3)
        else:
            n_input_channels = observation_space.shape[0]  # 假设已经是 (3, H, W)

        # 2. 定义 CNN (针对 7x7 输入)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 3. 计算 Flatten 后的维度
        # 我们需要模拟一个 (Batch, Channel, Height, Width) 的输入来计算
        with th.no_grad():
            # 创建一个假的输入张量: (1, 3, 7, 7)
            # 注意：这里我们强制指定为 (1, 3, 7, 7)，因为 SB3 会在 forward 前帮我们转好
            sample_input = th.zeros((1, n_input_channels, 7, 7)).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # 4. 线性层
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # SB3 会在调用这个 forward 之前，自动把 (Batch, 7, 7, 3) 转成 (Batch, 3, 7, 7)
        # 所以我们直接归一化并传入 CNN 即可
        return self.linear(self.cnn(observations / 255.0))