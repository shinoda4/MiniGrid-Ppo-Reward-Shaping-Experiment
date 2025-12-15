import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Key, Door, Goal

# --- 1. 简单势能 Wrapper (SimplePotentialShaping) ---
# 仅计算 Agent 到 Goal 的距离
class SimplePotentialShaping(gym.Wrapper):
    def __init__(self, env, shaping_weight=1.0, gamma=0.99):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.last_potential = 0.0
        self.gamma = gamma
        
    def get_potential(self):
        """仅计算 Agent 到 Goal 的势能 (0-1)"""
        unwrapped = self.unwrapped
        agent_pos = np.array(unwrapped.agent_pos)
        goal_pos = None
        
        # 寻找 Goal 位置 (假设只有一个 Goal)
        for i in range(unwrapped.grid.width):
            for j in range(unwrapped.grid.height):
                obj = unwrapped.grid.get(i, j)
                if isinstance(obj, Goal):
                    goal_pos = np.array((i, j))
                    break
            if goal_pos is not None: break
        
        if goal_pos is None:
            return 0.0 # 目标丢失
            
        max_dist = unwrapped.grid.width + unwrapped.grid.height
        dist = np.abs(agent_pos - goal_pos).sum()
        
        # 势能：靠近目标距离越短，势能越高 (归一化到 0-1)
        potential = 1.0 - dist / max_dist
        return potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.get_potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_potential = self.get_potential()
        # PBRS 公式: F = gamma * Phi(s') - Phi(s)
        shaping_reward = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        
        total_reward = reward + self.shaping_weight * shaping_reward
        
        # 在 info 中存储原始奖励，方便验证和绘图 (可选，但推荐)
        info['raw_reward'] = reward 
        
        return obs, total_reward, terminated, truncated, info


# --- 2. 分层势能 Wrapper (HierarchicalPotentialShaping) ---
# 这个类应该就是我们之前修改和验证过的版本
class HierarchicalPotentialShaping(gym.Wrapper):
    def __init__(self, env, shaping_weight=1.0, gamma=0.99):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.last_potential = 0.0
        self.gamma = gamma
        
    def get_potential(self):
        """
        计算分层势能：Stage 0 (Key), Stage 1 (Door), Stage 2 (Goal)
        总势能归一化到 0.0 - 3.0 之间。
        """
        unwrapped = self.unwrapped
        agent_pos = np.array(unwrapped.agent_pos)
        
        # 扫描网格寻找物体
        key_pos, door_pos, goal_pos = None, None, None
        for i in range(unwrapped.grid.width):
            for j in range(unwrapped.grid.height):
                obj = unwrapped.grid.get(i, j)
                if obj is not None:
                    if isinstance(obj, Key): key_pos = np.array((i, j))
                    elif isinstance(obj, Door): door_pos = np.array((i, j))
                    elif isinstance(obj, Goal): goal_pos = np.array((i, j))

        # 状态判断
        has_key = unwrapped.carrying is not None and isinstance(unwrapped.carrying, Key)
        door_open = False
        if door_pos is not None:
             cell = unwrapped.grid.get(*door_pos)
             if cell and cell.is_open:
                 door_open = True

        potential = 0.0
        max_dist = unwrapped.grid.width + unwrapped.grid.height

        if door_open:
            # Stage 2: 门开了，奔向终点 (势能 2.0 - 3.0)
            if goal_pos is not None:
                dist = np.abs(agent_pos - goal_pos).sum()
                potential = 2.0 + (1.0 - dist / max_dist)
            
        elif has_key:
            # Stage 1: 有钥匙，奔向门 (势能 1.0 - 2.0)
            if door_pos is not None:
                dist = np.abs(agent_pos - door_pos).sum()
                potential = 1.0 + (1.0 - dist / max_dist)
            else:
                potential = 1.0 # 容错
                
        else:
            # Stage 0: 没钥匙，奔向钥匙 (势能 0.0 - 1.0)
            if key_pos is not None:
                dist = np.abs(agent_pos - key_pos).sum()
                potential = 0.0 + (1.0 - dist / max_dist)
            else:
                potential = 0.0 # 容错

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
        
        # 在 info 中存储原始奖励
        info['raw_reward'] = reward 
        
        return obs, total_reward, terminated, truncated, info