import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Key, Door, Goal
from rnd_wrapper import RNDWrapper # 导入我们刚写的 RND

# ==========================================
# 1. 复用之前的 Ours 和 Simple Wrapper
# ==========================================
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
# 2. 自定义 CNN (适配 MiniGrid 7x7)
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

# ==========================================
# 3. 训练主程序
# ==========================================
POLICY_KWARGS = dict(
    features_extractor_class=MiniGridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

def make_env_fn(log_dir, algo_name, seed, wrapper_name):
    def _init():
        env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
        env = ImgObsWrapper(env)
        
        # Monitor 记录 extrinsic reward (实际任务表现)
        log_path = os.path.join(log_dir, f"{algo_name}_seed{seed}")
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_path, "0"))
        
        # 挂载 Wrapper
        if wrapper_name == "Ours":
            env = HierarchicalPotentialShaping(env, shaping_weight=1.0)
        elif wrapper_name == "RND":
            # RND 权重 0.05 是经验值，既鼓励探索又不至于完全忽略任务
            env = RNDWrapper(env, learning_rate=1e-4, intrinsic_weight=0.05)
        elif wrapper_name == "Baseline":
            pass # 纯 PPO
            
        env.reset(seed=seed)
        return env
    return _init

def run_comparison_training():
    TOTAL_TIMESTEPS = 500_000 
    SEEDS = [101, 102] # 为了演示快一点，这里只写了2个种子，论文建议跑3个
    LOG_DIR = "./logs_comparison_rnd_8x8"
    
    print(f"🔥 开始最终对比实验 (含 RND): MiniGrid-DoorKey-8x8")
    print(f"📂 数据保存至: {LOG_DIR}")

    experiments = [
        ("Baseline", "Baseline"),
        ("RND", "RND"),
        ("Ours", "Ours")
    ]

    for algo_name, wrapper_name in experiments:
        for seed in SEEDS:
            print(f"\n🚀 Training [{algo_name}] Seed {seed}...")
            
            env = DummyVecEnv([make_env_fn(LOG_DIR, algo_name, seed, wrapper_name)])
            
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                seed=seed,
                policy_kwargs=POLICY_KWARGS,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01,
                gamma=0.99
            )
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            env.close()
    
    print(f"\n✅ 实验结束！请使用 plot_success.py (需修改路径) 进行绘图。")

if __name__ == "__main__":
    run_comparison_training()