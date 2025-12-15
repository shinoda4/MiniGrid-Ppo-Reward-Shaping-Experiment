import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX
import numpy as np
from minigrid.core.world_object import Key, Door, Goal

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
                    if isinstance(obj, Key): key_pos = np.array((i, j))
                    elif isinstance(obj, Door): door_pos = np.array((i, j))
                    elif isinstance(obj, Goal): goal_pos = np.array((i, j))

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
        max_dist = unwrapped.grid.width + unwrapped.grid.height # 曼哈顿距离最大值

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