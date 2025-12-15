import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Key, Door, Goal

# ==========================================
# 1. 复制之前的 Wrapper 类定义 (必须包含)
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
        if door_open: # Stage 2
            if goal_pos is not None:
                dist = np.abs(agent_pos - goal_pos).sum()
                potential = 2.0 + (1.0 - dist / max_dist)
        elif has_key: # Stage 1
            if door_pos is not None:
                dist = np.abs(agent_pos - door_pos).sum()
                potential = 1.0 + (1.0 - dist / max_dist)
            else: potential = 1.0
        else: # Stage 0
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
# 2. 验证脚本：对比 8x8 环境下的势能逻辑
# ==========================================
def verify_8x8_logic():
    ENV_ID = "MiniGrid-DoorKey-8x8-v0"
    print(f"🔍 正在初始化 {ENV_ID} 进行逻辑验证...")
    
    try:
        base_env = gym.make(ENV_ID, render_mode="rgb_array")
        # 强制 Reset 一次以生成 grid
        base_env.reset(seed=123) 
    except Exception as e:
        print(f"❌ 错误: 无法加载环境 {ENV_ID}。请确认 gym-minigrid 版本。")
        print(f"   报错信息: {e}")
        return

    # 包装环境
    env_simple = SimplePotentialShaping(base_env)
    env_hierarchical = HierarchicalPotentialShaping(base_env)
    
    # 模拟关键场景
    # 场景 A: 初始状态 (没钥匙，离终点远)
    # 场景 B: 拿到钥匙 (Hierarchical应该高，Simple应该低因为钥匙可能离终点远)
    
    unwrapped = base_env.unwrapped
    
    # 寻找关键位置
    key_pos = None
    goal_pos = None
    for i in range(unwrapped.grid.width):
        for j in range(unwrapped.grid.height):
            obj = unwrapped.grid.get(i, j)
            if isinstance(obj, Key): key_pos = np.array((i, j))
            if isinstance(obj, Goal): goal_pos = np.array((i, j))
            
    print(f"   地图尺寸: {unwrapped.grid.width}x{unwrapped.grid.height}")
    print(f"   钥匙位置: {key_pos}")
    print(f"   终点位置: {goal_pos}")
    
    # --- 测试 1: 瞬移到钥匙旁边 ---
    # 强制修改 Agent 位置 (Cheat)
    unwrapped.agent_pos = key_pos 
    # 此时还没拿钥匙
    pot_simple_1 = env_simple.get_potential()
    pot_ours_1 = env_hierarchical.get_potential()
    
    print(f"\n--- 测试场景: 站在钥匙上 (未捡起) ---")
    print(f"   Simple Potential: {pot_simple_1:.4f} (只看终点距离)")
    print(f"   Ours Potential:   {pot_ours_1:.4f} (Stage 0: 满分接近 1.0)")
    
    # --- 测试 2: 捡起钥匙 ---
    # 强制让 Agent 拿着钥匙
    key_obj = unwrapped.grid.get(*key_pos)
    unwrapped.grid.set(*key_pos, None) # 地图上移除钥匙
    unwrapped.carrying = key_obj       # 放到手上
    
    pot_simple_2 = env_simple.get_potential()
    pot_ours_2 = env_hierarchical.get_potential()
    
    print(f"\n--- 测试场景: 捡起钥匙瞬间 ---")
    print(f"   Simple Potential: {pot_simple_2:.4f} (应该没变化，因为位置没变)")
    print(f"   Ours Potential:   {pot_ours_2:.4f} (应该暴涨! Jump into Stage 1, > 1.0)")

    if pot_ours_2 > 1.0 and abs(pot_simple_2 - pot_simple_1) < 0.01:
        print("\n✅ 验证通过！8x8 环境下，Hierarchical Shaping 逻辑正常，Simple Shaping 逻辑正常。")
        print("   Ours 成功捕捉到了‘捡起钥匙’的价值，而 Simple 对此无动于衷。")
    else:
        print("\n❌ 验证失败！势能计算不符合预期，请检查代码。")

if __name__ == "__main__":
    verify_8x8_logic()