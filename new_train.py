import os

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from minigrid.core.world_object import Key, Goal, Door
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from mamba_ssm import Mamba

class PotentialBasedRewardShaping(gym.Wrapper):
    """
    论文核心贡献: Potential-Based Reward Shaping (PBRS) Wrapper
    该 Wrapper 通过引入基于曼哈顿距离的势能函数，解决稀疏奖励问题。
    它不改变最优策略（Policy Invariance），但能显著加速收敛。
    """

    def __init__(self, env, shaping_weight=0.1):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.last_dist = None
        self.goal_pos = None

    def _get_distance(self):
        """
        计算智能体当前位置到目标的曼哈顿距离。
        """
        agent_pos = self.env.unwrapped.agent_pos
        # 第一次运行时寻找目标位置 (通常目标位置是固定的，缓存以提高性能)
        if self.goal_pos is None:
            grid = self.env.unwrapped.grid
            for i in range(grid.width):
                for j in range(grid.height):

                    cell = grid.get(i, j)

                    if cell is not None and cell.type == 'goal':
                        self.goal_pos = (i, j)

                        break

                if self.goal_pos: break

        # 如果依然找不到目标 (防御性编程)，返回 0

        if self.goal_pos is None:
            return 0

        # 计算曼哈顿距离: |x1-x2| + |y1-y2|

        return abs(agent_pos[0] - self.goal_pos[0]) + abs(agent_pos[1] - self.goal_pos[1])

    def reset(self, **kwargs):

        """环境重置时，记录初始距离"""

        obs, info = self.env.reset(**kwargs)

        # 重置 goal_pos 缓存，因为某些环境(如Random)每次reset目标位置会变

        self.goal_pos = None

        self.last_dist = self._get_distance()

        return obs, info

    def step(self, action):

        """环境步进时，计算额外的塑形奖励"""

        obs, reward, terminated, truncated, info = self.env.step(action)

        current_dist = self._get_distance()

        # --- 核心公式 ---

        # 势能差 = 上一步距离 - 当前距离

        # 靠近了: positive; 远离了: negative

        potential_difference = self.last_dist - current_dist

        shaping_reward = self.shaping_weight * potential_difference

        # 组合奖励

        total_reward = reward + shaping_reward

        # 更新距离

        self.last_dist = current_dist

        # 记录数据供论文绘图分析 (很重要)

        info['shaping_reward'] = shaping_reward

        info['original_reward'] = reward

        return obs, total_reward, terminated, truncated, info


class HierarchicalPotentialShaping(gym.Wrapper):

    def __init__(self, env, shaping_weight=1.0):

        super().__init__(env)

        self.shaping_weight = shaping_weight

        self.last_potential = 0.0

        self.gamma = 0.99

    def get_potential(self):

        """

        计算分层势能：

        Stage 0: 没钥匙 -> 靠近钥匙

        Stage 1: 有钥匙 -> 靠近门

        Stage 2: 门开了 -> 靠近终点

        """

        # 必须访问底层 MiniGrid 环境获取物体位置

        unwrapped = self.unwrapped

        # 获取关键物体位置

        agent_pos = np.array(unwrapped.agent_pos)

        key_pos = None

        door_pos = None

        goal_pos = None

        # 扫描网格寻找物体 (因为位置可能是随机的)

        for i in range(unwrapped.grid.width):

            for j in range(unwrapped.grid.height):

                obj = unwrapped.grid.get(i, j)

                if obj is not None:

                    if isinstance(obj, Key):
                        key_pos = np.array((i, j))

                    elif isinstance(obj, Door):
                        door_pos = np.array((i, j))

                    elif isinstance(obj, Goal):
                        goal_pos = np.array((i, j))

        # 状态判断

        has_key = unwrapped.carrying is not None and isinstance(unwrapped.carrying, Key)

        # Door 即使被打开，对象还在，只是 is_open=True

        door_open = False

        if door_pos is not None:

            cell = unwrapped.grid.get(*door_pos)

            if cell and cell.is_open:
                door_open = True

        # --- 计算势能 ---

        # 我们把整个任务进度归一化到 0.0 - 3.0 之间

        potential = 0.0

        max_dist = unwrapped.grid.width + unwrapped.grid.height  # 曼哈顿距离最大值

        if door_open:

            # Stage 2: 门开了，全力奔向终点

            # 基础分 2.0，加上距离分

            dist = np.abs(agent_pos - goal_pos).sum()

            potential = 2.0 + (1.0 - dist / max_dist)



        elif has_key:

            # Stage 1: 有钥匙，全力奔向门

            # 基础分 1.0

            if door_pos is not None:

                dist = np.abs(agent_pos - door_pos).sum()

                potential = 1.0 + (1.0 - dist / max_dist)

            else:

                # 异常情况：有钥匙但没门（理论不应发生）

                potential = 1.0



        else:

            # Stage 0: 没钥匙，全力奔向钥匙

            if key_pos is not None:

                dist = np.abs(agent_pos - key_pos).sum()

                potential = 0.0 + (1.0 - dist / max_dist)

            else:

                # 钥匙已经被捡起来了(但在carrying里没检测到？)或者还没生成

                potential = 0.0

        return potential

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.last_potential = self.get_potential()

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 计算当前的势能

        current_potential = self.get_potential()

        # PBRS 公式: F = gamma * Phi(s') - Phi(s)

        shaping_reward = self.gamma * current_potential - self.last_potential

        # 更新状态

        self.last_potential = current_potential

        # 添加到原始奖励 (权重通常给小一点，防止主导)

        total_reward = reward + self.shaping_weight * shaping_reward

        return obs, total_reward, terminated, truncated, info


class MiniGridFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):

        super().__init__(observation_space, features_dim)

        if observation_space.shape[-1] == 3:

            n_input_channels = observation_space.shape[-1]

        else:

            n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(

            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),

            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),

            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),

            nn.ReLU(),

            nn.Flatten(),

        )

        with th.no_grad():

            sample_input = th.zeros((1, n_input_channels, 7, 7)).float()

            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.linear(self.cnn(observations / 255.0))


class MambaLayer(nn.Module):
    """

    基础 Mamba 块：负责维度转换和计算

    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is not installed.")

        self.mamba = Mamba(

            d_model=dim,

            d_state=d_state,

            d_conv=d_conv,

            expand=expand

        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (Batch, Dim) -> (Batch, 1, Dim) for Mamba -> (Batch, Dim)

        x_seq = x.unsqueeze(1)

        x_out = self.mamba(self.norm(x_seq))

        return x_out.squeeze(1)


class MambaMlpExtractor(nn.Module):
    """

    [最终修复版] 适配 SB3 接口的 Mamba Extractor

    实现了 forward, forward_actor, forward_critic 以满足 SB3 的调用链

    """

    def __init__(self, features_dim, mamba_kwargs):
        super().__init__()

        self.mamba_net = MambaLayer(features_dim, **mamba_kwargs)

        # 必须显式定义这两个属性

        self.latent_dim_pi = features_dim

        self.latent_dim_vf = features_dim

    def forward(self, features):
        """

        标准前向传播，返回 (latent_pi, latent_vf)

        """

        x = self.mamba_net(features)

        return x, x

    def forward_actor(self, features):
        """

        SB3 在 evaluate_actions 时会单独调用此方法

        """

        return self.mamba_net(features)

    def forward_critic(self, features):
        """

        SB3 在 predict_values 时会单独调用此方法 (报错就是因为缺这个)

        """

        return self.mamba_net(features)


class MambaActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        self.mamba_kwargs = kwargs.pop('mamba_kwargs', {})

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """

        构建自定义的 Mamba Extractor

        """

        self.mlp_extractor = MambaMlpExtractor(

            features_dim=self.features_dim,

            mamba_kwargs=self.mamba_kwargs

        )


def run_doorkey_5x5_v0_training_with_mamba():
    ENV_ID = "MiniGrid-DoorKey-5x5-v0"

    TOTAL_TIMESTEPS = 200_000

    SEEDS = [101, 102, 103]

    LOG_DIR = "./logs_paper_mamba_comparison"  # 新的日志目录

    def make_env_fn(algo_name, seed, use_shaping):
        def _init():
            env = gym.make(ENV_ID, render_mode="rgb_array")

            env = ImgObsWrapper(env)

            log_path = os.path.join(LOG_DIR, f"{algo_name}_seed{seed}")

            os.makedirs(log_path, exist_ok=True)

            env = Monitor(env, filename=os.path.join(log_path, "0"))

            if use_shaping:
                # 依然使用高权重的分层势能

                env = HierarchicalPotentialShaping(env, shaping_weight=1.0)

            env.reset(seed=seed)

            return env

        return _init

    print(f"--- 开始实验 (含 Mamba): {ENV_ID} ---")

    print(f"数据将保存至: {LOG_DIR}")

    # 基础 CNN 参数

    policy_kwargs_cnn = dict(

        features_extractor_class=MiniGridFeaturesExtractor,

        features_extractor_kwargs=dict(features_dim=128),

    )

    # Mamba 专用 Policy 参数

    # 我们传入自定义的 Policy 类，以及 Mamba 的超参数

    policy_kwargs_mamba = dict(

        features_extractor_class=MiniGridFeaturesExtractor,

        features_extractor_kwargs=dict(features_dim=128),

        # 传递给 MambaActorCriticPolicy 的参数

        mamba_kwargs=dict(d_state=16, d_conv=4, expand=2)

    )

    # --- 定义三组实验 ---

    # 1. Baseline (PPO + CNN)

    # 2. Ours (PPO + CNN + Hierarchical Shaping)

    # 3. Ours+Mamba (MambaPPO + CNN + Hierarchical Shaping)

    experiments = [

        # (实验名, 是否用Shaping, 是否用Mamba)

        # ("Baseline", False, False),

        # ("Ours_Shaping", True, False),

        ("Ours_Shaping_Mamba", True, True)

    ]
