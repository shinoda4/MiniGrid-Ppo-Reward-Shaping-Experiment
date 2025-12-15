import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 学术风格配置
# ==========================================
# 保持 Times New Roman/Serif 字体，高分辨率配置
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,  # 稍减线条宽度
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ==========================================
# 2. 配置参数
# ==========================================
LOG_DIR = "./logs_ablation_doorkey"  # 确保与训练脚本一致
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"
MAX_STEPS = 250  # DoorKey-5x5 的标准最大步数
# 针对 200k Timesteps 的数据，使用 2500 窗口进行平滑
SMOOTHING_WINDOW = 2500 

def compute_clean_reward(row):
    """
    重构标准奖励 (0.0 - 1.0)，去除 Shaping 的数值干扰。
    公式: R = 1 - 0.9 * (step_count / max_steps)
    """
    if row['l'] < MAX_STEPS:
        return 1 - 0.9 * (row['l'] / MAX_STEPS)
    return 0.0

def load_data():
    all_data = []
    print(f"📂 正在从 {LOG_DIR} 读取数据...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！请先运行训练脚本。")
        return pd.DataFrame()

    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                folder_name = os.path.basename(root)
                
                # --- 自动识别组别 ---
                if "Baseline" in folder_name:
                    algo = "Baseline"
                elif "SimpleShaping" in folder_name:
                    algo = "Simple Shaping (Goal Dist)"
                elif "Ours" in folder_name:
                    algo = "Ours (Hierarchical Potential)"
                else:
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    # 跳过第一行 header
                    df = pd.read_csv(file_path, skiprows=1)
                    
                    if 'l' not in df.columns:
                        continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # --- 核心: 使用 Clean Reward 进行公平对比 ---
                    df['Reward'] = df.apply(compute_clean_reward, axis=1)
                    
                    # *** 关键优化: 增大平滑窗口 ***
                    df['Smoothed_Reward'] = df['Reward'].rolling(
                        window=SMOOTHING_WINDOW, 
                        min_periods=100  # 确保有足够数据才开始平滑
                    ).mean()
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 读取 {file_path} 失败: {e}")
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_ablation():
    df = load_data()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成消融实验对比图 (高平滑度)...")
    
    # 定义绘图顺序和颜色
    # 顺序：Ours (最重要) -> Simple (对比) -> Baseline (基准)
    HUE_ORDER = ["Ours (Hierarchical Potential)", "Simple Shaping (Goal Dist)", "Baseline"]
    
    # 颜色方案
    PALETTE = {
        "Ours (Hierarchical Potential)": "#c0392b",       # 突出：深红色 
        "Simple Shaping (Goal Dist)": "#f39c12",          # 对比：橙黄色 
        "Baseline": "#34495e"                             # 基准：深蓝色/灰色
    }
    
    # 线型方案 (Ours实线，其他虚线)
    DASHES = {
        "Ours (Hierarchical Potential)": (1, 0),          # 实线
        "Simple Shaping (Goal Dist)": (4, 2),             # 虚线
        "Baseline": (1, 1)                                # 点线
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(
        data=df,
        x="Timesteps",
        y="Smoothed_Reward",
        hue="Algorithm",
        style="Algorithm",
        hue_order=HUE_ORDER,
        style_order=HUE_ORDER,
        palette=PALETTE,
        dashes=DASHES,
        errorbar=None, # *** 关键修改：取消阴影，让曲线更干净 ***
        linewidth=2.0,
        alpha=0.9,
        ax=ax
    )
    
    # 坐标轴设置
    ax.set_ylim(-0.05, 1.05) 
    ax.set_xlabel("Environment Interactions (Timesteps)")
    ax.set_ylabel("Average Episodic Reward (Clean)")
    ax.set_title(f"Ablation Study: {ENV_NAME}", pad=15, weight='bold')
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例设置 (放在右下角)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    
    save_path = "ablation_result_optimized.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！图片已保存为: {save_path}")

if __name__ == "__main__":
    plot_ablation()