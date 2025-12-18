import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Key, Door, Goal

# ==========================================
# 1. 核心组件 (CNN + Wrappers) - 保持不变
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

class SimplePotentialShaping(gym.Wrapper):
    def __init__(self, env, shaping_weight=1.0, gamma=0.99):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.last_potential = 0.0
        self.gamma = gamma
    def get_potential(self):
        unwrapped = self.unwrapped
        agent_pos = np.array(unwrapped.agent_pos)
        goal_pos = None
        for i in range(unwrapped.grid.width):
            for j in range(unwrapped.grid.height):
                obj = unwrapped.grid.get(i, j)
                if isinstance(obj, Goal):
                    goal_pos = np.array((i, j))
                    break
            if goal_pos is not None: break
        if goal_pos is None: return 0.0
        max_dist = unwrapped.grid.width + unwrapped.grid.height
        dist = np.abs(agent_pos - goal_pos).sum()
        return 1.0 - dist / max_dist
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.get_potential()
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = self.get_potential()
        shaping_reward = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        total_reward = reward + self.shaping_weight * shaping_reward
        return obs, total_reward, terminated, truncated, info

class HierarchicalPotentialShaping(gym.Wrapper):
    def __init__(self, env, shaping_weight=1.0, gamma=0.99):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.last_potential = 0.0
        self.gamma = gamma
    def get_potential(self):
        unwrapped = self.unwrapped
        agent_pos = np.array(unwrapped.agent_pos)
        key_pos, door_pos, goal_pos = None, None, None
        for i in range(unwrapped.grid.width):
            for j in range(unwrapped.grid.height):
                obj = unwrapped.grid.get(i, j)
                if isinstance(obj, Key): key_pos = np.array((i, j))
                elif isinstance(obj, Door): door_pos = np.array((i, j))
                elif isinstance(obj, Goal): goal_pos = np.array((i, j))
        has_key = unwrapped.carrying is not None and isinstance(unwrapped.carrying, Key)
        door_open = False
        if door_pos is not None:
             cell = unwrapped.grid.get(*door_pos)
             if cell and cell.is_open: door_open = True
        max_dist = unwrapped.grid.width + unwrapped.grid.height
        potential = 0.0
        if door_open: 
            if goal_pos is not None:
                dist = np.abs(agent_pos - goal_pos).sum()
                potential = 2.0 + (1.0 - dist / max_dist)
        elif has_key: 
            if door_pos is not None:
                dist = np.abs(agent_pos - door_pos).sum()
                potential = 1.0 + (1.0 - dist / max_dist)
            else: potential = 1.0
        else: 
            if key_pos is not None:
                dist = np.abs(agent_pos - key_pos).sum()
                potential = 0.0 + (1.0 - dist / max_dist)
        return potential
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.get_potential()
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = self.get_potential()
        shaping_reward = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        total_reward = reward + self.shaping_weight * shaping_reward
        return obs, total_reward, terminated, truncated, info

# ==========================================
# 2. 实验配置
# ==========================================
POLICY_KWARGS = dict(
    features_extractor_class=MiniGridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

def make_dqn_env_fn(log_dir, algo_name, seed, wrapper_name):
    def _init():
        # [修改] 降级为 5x5，确保快速出结果
        env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
        env = ImgObsWrapper(env)
        
        log_path = os.path.join(log_dir, f"{algo_name}_seed{seed}")
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_path, "0"))
        
        if wrapper_name == "DQN_Ours":
            env = HierarchicalPotentialShaping(env, shaping_weight=1.0)
        elif wrapper_name == "DQN_Simple":
            env = SimplePotentialShaping(env, shaping_weight=1.0)
        
        env.reset(seed=seed)
        return env
    return _init

def run_dqn_generality():
    # 5x5 比较简单，200k 步足够 DQN 学会了
    TOTAL_TIMESTEPS = 200_000 
    SEEDS = [201, 202] 
    # [修改] 文件夹名字改一下，避免混淆
    LOG_DIR = "./logs_generality_dqn_5x5"
    
    print(f"🔥 开始通用性验证实验 (DQN): MiniGrid-DoorKey-5x5")
    print(f"📂 数据保存至: {LOG_DIR}")

    experiments = [
        ("DQN_Baseline", "None"),
        ("DQN_Simple", "Simple"), # 在 5x5 上 Simple 可能也会学出来，但 Ours 应该更快
        ("DQN_Ours", "Ours")
    ]

    for algo_name, wrapper_name in experiments:
        for seed in SEEDS:
            print(f"\n🚀 Training [{algo_name}] Seed {seed}...")
            
            env = DummyVecEnv([make_dqn_env_fn(LOG_DIR, algo_name, seed, wrapper_name)])
            
            # DQN 针对 MiniGrid 的推荐参数
            model = DQN(
                "CnnPolicy",
                env,
                verbose=1,
                seed=seed,
                policy_kwargs=POLICY_KWARGS,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000, # 先随机探索一会
                batch_size=32,
                tau=1.0,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.1, # 前10%时间进行探索衰减
                exploration_final_eps=0.05,
                gamma=0.99
            )
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            env.close()
    
    print(f"\n✅ DQN 实验结束！请修改绘图脚本的 LOG_DIR 查看结果。")

if __name__ == "__main__":
    # 1. 先验证
    # verify_dqn_setup() # 第一次运行建议打开注释
    
    # 2. 后训练
    run_dqn_generality()