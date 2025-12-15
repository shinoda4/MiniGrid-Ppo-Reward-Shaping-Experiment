import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 学术风格设置 ---
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "lines.linewidth": 2.5,
    "pdf.fonttype": 42
})

LOG_DIR = "./logs_paper_minigrid_doorkey_5x5_v0"  # 请确保数据在这个文件夹下
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"
MAX_STEPS = 250  # DoorKey-5x5 的默认最大步数

def compute_clean_reward(row):
    """重构 DoorKey 的标准奖励"""
    if row['l'] < MAX_STEPS:
        return 1 - 0.9 * (row['l'] / MAX_STEPS)
    return 0.0

def load_data():
    all_data = []
    print(f"正在读取 {LOG_DIR} 下的数据...")
    
    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                folder_name = os.path.basename(root)
                
                # 自动识别算法
                if "Baseline" in folder_name:
                    algo = "Baseline"
                elif "Ours" in folder_name:
                    algo = "Ours (Potential Shaping)"
                else:
                    continue
                
                try:
                    df = pd.read_csv(os.path.join(root, file), skiprows=1)
                    if 'l' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    # 核心：重构奖励，去除 Shaping 噪音
                    df['Clean_Reward'] = df.apply(compute_clean_reward, axis=1)
                    # 平滑处理
                    df['Smoothed_Reward'] = df['Clean_Reward'].rolling(window=200, min_periods=10).mean()
                    all_data.append(df)
                except:
                    pass
    
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_doorkey():
    df = load_data()
    if df.empty:
        print("❌ 没有数据！请先运行 DoorKey 的训练实验。")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.lineplot(
        data=df, x="Timesteps", y="Smoothed_Reward",
        hue="Algorithm", style="Algorithm",
        dashes={"Ours (Potential Shaping)": (1, 0), "Baseline": (2, 2)},
        palette=["#e74c3c", "#34495e"], # 红色(Ours) vs 深蓝(Baseline)
        errorbar='sd', ax=ax
    )
    # (Hierarchical Task)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Robustness: {ENV_NAME}", pad=15, weight='bold')
    ax.set_xlabel("Environment Interactions")
    ax.set_ylabel("Average Episodic Reward")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("logs_paper_minigrid_doorkey_5x5_v0.png", dpi=300)
    print("✅ DoorKey 实验图已生成！")

if __name__ == "__main__":
    plot_doorkey()

