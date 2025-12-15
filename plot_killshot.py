import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 学术风格配置 (High Quality)
# ==========================================
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,              # 稍微加大字体
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
    "lines.linewidth": 2.5,       # 加粗线条
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ==========================================
# 2. 关键参数配置 (针对 8x8 环境)
# ==========================================
LOG_DIR = "./logs_killshot_8x8"
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"

# [注意] 8x8 环境的 Max Steps 通常是 10 * 8 * 8 = 640
# 如果你的环境版本不一样，这里可能需要微调，但 640 是标准值
MAX_STEPS = 640 

# [注意] 针对 500k 总步数，我们需要更大的窗口来获得平滑曲线
# 5000 步的移动平均能过滤掉绝大部分震荡，留下干净的趋势
SMOOTHING_WINDOW = 5000

def compute_clean_reward(row):
    """
    重构标准奖励 (0.0 - 1.0)
    公式: R = 1 - 0.9 * (step_count / max_steps)
    """
    if row['l'] < MAX_STEPS:
        return 1 - 0.9 * (row['l'] / MAX_STEPS)
    return 0.0

def load_data():
    all_data = []
    print(f"📂 正在从 {LOG_DIR} 读取 Kill Shot 数据...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！")
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
                    df = pd.read_csv(file_path, skiprows=1)
                    
                    if 'l' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # 计算干净的奖励
                    df['Reward'] = df.apply(compute_clean_reward, axis=1)
                    
                    # 平滑处理
                    df['Smoothed_Reward'] = df['Reward'].rolling(
                        window=SMOOTHING_WINDOW, 
                        min_periods=100
                    ).mean()
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 读取 {file_path} 失败: {e}")
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_killshot():
    df = load_data()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成最终 Kill Shot 对比图...")
    
    # 定义绘图顺序
    HUE_ORDER = ["Ours (Hierarchical Potential)", "Simple Shaping (Goal Dist)", "Baseline"]
    
    # 颜色方案 (高对比度)
    PALETTE = {
        "Ours (Hierarchical Potential)": "#c0392b",       # 鲜艳的深红色 (主角)
        "Simple Shaping (Goal Dist)": "#f1c40f",          # 黄色 (对比组 - 应该很低)
        "Baseline": "#2c3e50"                             # 深黑色 (基线 - 应该贴地)
    }
    
    # 线型方案
    DASHES = {
        "Ours (Hierarchical Potential)": (1, 0),          # 实线
        "Simple Shaping (Goal Dist)": (3, 1),             # 虚线
        "Baseline": (1, 1)                                # 点线
    }

    fig, ax = plt.subplots(figsize=(10, 6)) # 宽一点的图，显得大气
    
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
        errorbar=None,       # 保持干净，不画阴影 (如果曲线重叠严重，关掉阴影更好看)
        linewidth=3.0,       # 线条更粗
        alpha=0.95,
        ax=ax
    )
    
    # 坐标轴与标签
    ax.set_ylim(-0.02, 1.02) # 稍微留一点余地
    ax.set_xlim(0, 500000)   # 强制显示到 500k
    
    ax.set_xlabel("Environment Interactions (Timesteps)")
    ax.set_ylabel("Average Episodic Reward (Clean)")
    ax.set_title(f"Scalability Test: {ENV_NAME}", pad=20, weight='bold')
    
    # 增加标注 (可选，如果 Ours 效果特别好，可以加个箭头)
    # ax.annotate('Ours converges', xy=(200000, 0.8), xytext=(250000, 0.9),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 图例放在左上角或最佳位置
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()
    
    save_path = "killshot_result_8x8.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绝杀图已生成！图片保存为: {save_path}")
    print("   预期效果: 红线稳步上升，黄线和黑线在底部变成一条直线 (0.0)")

if __name__ == "__main__":
    plot_killshot()