import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (对应 RND 对比实验)
# ==========================================
LOG_DIR = "./logs_comparison_rnd_8x8"  # 对应 train_comparison_with_rnd.py 的日志目录
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
MAX_STEPS = 500000                     # 总步数
SMOOTHING_WINDOW = 5000                # 平滑窗口 (针对500k步，5000比较平滑)

# ==========================================
# 2. 学术风格配置
# ==========================================
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
    "lines.linewidth": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def load_data():
    all_data = []
    print(f"📂 正在从 {LOG_DIR} 读取 RND 对比数据...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！请先运行 RND 训练脚本。")
        return pd.DataFrame()

    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                folder_name = os.path.basename(root)
                
                # --- 自动识别组别 (增加 RND) ---
                if "Baseline" in folder_name:
                    algo = "Baseline"
                elif "RND" in folder_name:
                    algo = "RND (Curiosity)"
                elif "Ours" in folder_name:
                    algo = "Ours (Hierarchical Potential)"
                else:
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, skiprows=1)
                    
                    if 'r' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # --- 计算成功率 (MiniGrid: Reward > 0 即成功) ---
                    df['Success'] = (df['r'] > 0).astype(float)
                    
                    # 平滑处理
                    df['Success_Rate'] = df['Success'].rolling(
                        window=SMOOTHING_WINDOW, 
                        min_periods=100
                    ).mean()
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 读取 {file_path} 失败: {e}")
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_success_rnd():
    df = load_data()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成 RND 对比成功率图...")
    
    # 绘图顺序: Ours 最强，RND 次之(或差不多)，Baseline 最弱
    HUE_ORDER = ["Ours (Hierarchical Potential)", "RND (Curiosity)", "Baseline"]
    
    # 颜色方案
    PALETTE = {
        "Ours (Hierarchical Potential)": "#c0392b",  # 深红 (Ours)
        "RND (Curiosity)": "#27ae60",                # 绿色 (RND - 代表探索/新奇)
        "Baseline": "#2c3e50"                        # 深灰 (Baseline)
    }
    
    # 线型方案
    DASHES = {
        "Ours (Hierarchical Potential)": (1, 0),     # 实线
        "RND (Curiosity)": (2, 1),                   # 虚线 (长短)
        "Baseline": (1, 1)                           # 点线
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x="Timesteps",
        y="Success_Rate",
        hue="Algorithm",
        style="Algorithm",
        hue_order=HUE_ORDER,
        style_order=HUE_ORDER,
        palette=PALETTE,
        dashes=DASHES,
        errorbar=None,  # 保持画面干净
        linewidth=3.0,
        alpha=0.95,
        ax=ax
    )
    
    # 坐标轴设置
    ax.set_ylim(-0.02, 1.02) # 0% - 100%
    ax.set_xlim(0, MAX_STEPS)
    
    # 格式化 Y 轴为百分比
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    
    ax.set_xlabel("Environment Interactions (Timesteps)")
    ax.set_ylabel("Task Success Rate")
    ax.set_title(f"Comparison with Curiosity (RND): {ENV_NAME}", pad=20, weight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()
    
    save_path = "comparison_rnd_success_8x8.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ RND 对比图已生成！图片保存为: {save_path}")

if __name__ == "__main__":
    plot_success_rnd()