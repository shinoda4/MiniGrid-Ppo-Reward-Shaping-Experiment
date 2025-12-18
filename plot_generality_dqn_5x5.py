import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (对应 DQN 通用性实验)
# ==========================================
LOG_DIR = "./logs_generality_dqn_5x5"  # 对应新的日志目录
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"   # 改名
MAX_STEPS = 200000                     # 改为 200k
SMOOTHING_WINDOW = 2000                # 改小一点


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
    print(f"📂 正在从 {LOG_DIR} 读取 DQN 通用性数据...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！请先运行 DQN 训练脚本。")
        return pd.DataFrame()

    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                folder_name = os.path.basename(root)
                
                # --- [关键修改] 自动识别 DQN 组别 ---
                # 对应 train_dqn_generality.py 中的命名
                if "DQN_Baseline" in folder_name:
                    algo = "DQN (Baseline)"
                elif "DQN_Simple" in folder_name:
                    algo = "DQN + Simple Shaping"
                elif "DQN_Ours" in folder_name:
                    algo = "DQN + Ours (Hierarchical)"
                else:
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    # 跳过第一行 header
                    df = pd.read_csv(file_path, skiprows=1)
                    
                    if 'r' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # --- 计算成功率 (Reward > 0 即成功) ---
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

def plot_success_dqn():
    df = load_data()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成 DQN 通用性验证图...")
    
    # 绘图顺序
    HUE_ORDER = ["DQN + Ours (Hierarchical)", "DQN + Simple Shaping", "DQN (Baseline)"]
    
    # 颜色方案
    PALETTE = {
        "DQN + Ours (Hierarchical)": "#c0392b",    # 深红
        "DQN + Simple Shaping": "#f39c12",         # 橙黄
        "DQN (Baseline)": "#34495e"                # 深灰
    }
    
    # 线型方案
    DASHES = {
        "DQN + Ours (Hierarchical)": (1, 0),       # 实线
        "DQN + Simple Shaping": (3, 1),            # 虚线
        "DQN (Baseline)": (1, 1)                   # 点线
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
        errorbar=None,
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
    ax.set_title(f"Generality Check (DQN): {ENV_NAME}", pad=20, weight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()
    
    save_path = "generality_dqn_success_5x5.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ DQN 验证图已生成！图片保存为: {save_path}")

if __name__ == "__main__":
    plot_success_dqn()