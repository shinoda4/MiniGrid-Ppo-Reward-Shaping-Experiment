import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import ImgObsWrapper

from minigrid_cnn import MiniGridFeaturesExtractor
from my_wrapper import PotentialBasedRewardShaping, HierarchicalPotentialShaping


def run_empty_8x8_v0_training():
    ENV_ID = "MiniGrid-Empty-8x8-v0"
    TOTAL_TIMESTEPS = 100_000   # 10万步足够收敛，想看更稳可以设为 200_000
    SEEDS = [101, 102, 103]     # 跑3个种子取平均，这是论文标准
    LOG_DIR = "./logs_paper_minigrid_empty_8x8_v0"    # 日志保存位置
    
    def make_env(seed, use_shaping=False):
        """环境工厂：组装环境、Wrapper和监控器"""
        def _init():
            env = gym.make(ENV_ID, render_mode=None)
            env = ImgObsWrapper(env) # 必须先转为图像
            
            if use_shaping:
                # 启用你的核心算法
                env = PotentialBasedRewardShaping(env, shaping_weight=0.1)
                
            # Monitor 记录数据用于绘图 (allow_early_resets=True防止报错)
            group_name = "Ours" if use_shaping else "Baseline"
            log_path = os.path.join(LOG_DIR, f"{group_name}_seed{seed}")
            os.makedirs(log_path, exist_ok=True)
            env = Monitor(env, log_path)
            return env
        return _init
        
    print(f"--- 开始实验: {ENV_ID} ---")
    print(f"数据将保存至: {LOG_DIR}")
    
    # 定义 CNN 参数 (关键修复)
    policy_kwargs = dict(
        features_extractor_class=MiniGridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # 1. 训练 Baseline
    for seed in SEEDS:
        print(f"\n[Baseline] Training Seed {seed}...")
        env = DummyVecEnv([make_env(seed, use_shaping=False)])
        
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1, 
            seed=seed,
            policy_kwargs=policy_kwargs # 使用定制CNN
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(f"{LOG_DIR}/Baseline_seed{seed}/final_model")
        env.close()

    # 2. 训练 Ours
    for seed in SEEDS:
        print(f"\n[Ours] Training Seed {seed}...")
        env = DummyVecEnv([make_env(seed, use_shaping=True)])
        
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1, 
            seed=seed,
            policy_kwargs=policy_kwargs # 使用定制CNN
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(f"{LOG_DIR}/Ours_seed{seed}/final_model")
        env.close()
        
    print("\n所有训练结束！请运行绘图脚本。")




def run_doorkey_5x5_v0_training():
    ENV_ID = "MiniGrid-DoorKey-5x5-v0"
    TOTAL_TIMESTEPS = 200_000   
    SEEDS = [101, 102, 103]     
    LOG_DIR = "./logs_paper_minigrid_doorkey_5x5_v0" # 改个名字，方便和之前的实验区分
    
    def make_env_fn(algo_name, seed, use_shaping):
            def _init():
                env = gym.make(ENV_ID, render_mode="rgb_array")
                env = ImgObsWrapper(env)
                
                log_path = os.path.join(LOG_DIR, f"{algo_name}_seed{seed}")
                os.makedirs(log_path, exist_ok=True)
                env = Monitor(env, filename=os.path.join(log_path, "0"))
                
                if use_shaping:
                    # 使用新的分层势能 Wrapper
                    # weight 可以给大一点 (1.0) 甚至更大，因为这个势能非常准
                    env = HierarchicalPotentialShaping(env, shaping_weight=1.0) 
                    
                env.reset(seed=seed)
                return env
            return _init

    print(f"--- 开始实验: {ENV_ID} ---")
    print(f"数据将保存至: {LOG_DIR}")
    
    # 你的 CNN 参数 (保持不变)
    policy_kwargs = dict(
        features_extractor_class=MiniGridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # --- 循环训练 ---
    # 我们把 Baseline 和 Ours 的配置写在列表里遍历，代码更简洁
    experiments = [
        ("Baseline", False),
        ("Ours", True)
    ]

    for algo_name, use_shaping in experiments:
        for seed in SEEDS:
            print(f"\n[{algo_name}] Training Seed {seed} (Shaping={use_shaping})...")
            
            # --- 修复 2: 正确的 VecEnv 初始化 ---
            # DummyVecEnv 需要一个函数列表 [lambda: env, ...]
            env = DummyVecEnv([make_env_fn(algo_name, seed, use_shaping)])
            
            model = PPO(
                "CnnPolicy", 
                env, 
                verbose=1, 
                seed=seed,
                policy_kwargs=policy_kwargs,
                learning_rate=0.0003, # 推荐参数
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01 # 增加一点熵，防止 DoorKey 早期陷入局部最优
            )
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            
            # 保存模型
            save_path = os.path.join(LOG_DIR, f"{algo_name}_seed{seed}", "final_model")
            model.save(save_path)
            
            env.close()

    print(f"\n✅ 所有训练结束！请运行之前的 'plot_doorkey' 函数进行绘图。")









