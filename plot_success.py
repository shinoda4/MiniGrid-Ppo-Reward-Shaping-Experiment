import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (根据你想画的实验修改这里)
# ==========================================
# 选项 A: 画 5x5 消融实验
LOG_DIR = "./logs_ablation_doorkey"
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"
MAX_STEPS = 200000 # 或者 200000，取决于你跑了多少
SMOOTHING_WINDOW = 2500

# 选项 B: 画 8x8 Kill Shot 实验 (默认开启)
# LOG_DIR = "./logs_killshot_8x8"
# ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
# MAX_STEPS = 500000
# SMOOTHING_WINDOW = 5000

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
    print(f"📂 正在从 {LOG_DIR} 读取数据计算成功率...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！请检查路径。")
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
                    
                    if 'r' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # ==========================================
                    # [核心逻辑] 计算成功率
                    # 在 MiniGrid 中，只要 r > 0 即代表成功到达终点
                    # ==========================================
                    df['Success'] = (df['r'] > 0).astype(float)
                    
                    # 平滑处理 (计算滑动平均成功率)
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
def load_data_8x8():
    # 选项 B: 画 8x8 Kill Shot 实验 (默认开启)
    LOG_DIR = "./logs_killshot_8x8"
    ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
    MAX_STEPS = 500000
    SMOOTHING_WINDOW = 5000

    all_data = []
    print(f"📂 正在从 {LOG_DIR} 读取数据计算成功率...")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 文件夹 {LOG_DIR} 不存在！请检查路径。")
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
                    
                    if 'r' not in df.columns: continue
                    
                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # ==========================================
                    # [核心逻辑] 计算成功率
                    # 在 MiniGrid 中，只要 r > 0 即代表成功到达终点
                    # ==========================================
                    df['Success'] = (df['r'] > 0).astype(float)
                    
                    # 平滑处理 (计算滑动平均成功率)
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

def plot_success():
    df = load_data()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成成功率 (Success Rate) 对比图...")
    
    # 绘图顺序
    HUE_ORDER = ["Ours (Hierarchical Potential)", "Simple Shaping (Goal Dist)", "Baseline"]
    
    # 颜色方案
    PALETTE = {
        "Ours (Hierarchical Potential)": "#c0392b",       # 红
        "Simple Shaping (Goal Dist)": "#f1c40f",          # 黄
        "Baseline": "#2c3e50"                             # 黑
    }
    
    # 线型方案
    DASHES = {
        "Ours (Hierarchical Potential)": (1, 0),          # 实线
        "Simple Shaping (Goal Dist)": (3, 1),             # 虚线
        "Baseline": (1, 1)                                # 点线
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
        errorbar=None,       # 成功率通常不需要阴影，保持干净
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
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Task Success Rate: {ENV_NAME}", pad=20, weight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()
    
    # 根据文件夹自动命名保存
    output_name = "success_rate_8x8.png" if "8x8" in LOG_DIR else "success_rate_5x5.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ 成功率图表已生成！图片保存为: {output_name}")
def plot_success_8x8():

    # 选项 B: 画 8x8 Kill Shot 实验 (默认开启)
    LOG_DIR = "./logs_killshot_8x8"
    ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
    MAX_STEPS = 500000
    SMOOTHING_WINDOW = 5000

    df = load_data_8x8()
    
    if df.empty:
        print("❌ 未找到有效数据，无法绘图。")
        return

    print("📊 正在生成成功率 (Success Rate) 对比图...")
    
    # 绘图顺序
    HUE_ORDER = ["Ours (Hierarchical Potential)", "Simple Shaping (Goal Dist)", "Baseline"]
    
    # 颜色方案
    PALETTE = {
        "Ours (Hierarchical Potential)": "#c0392b",       # 红
        "Simple Shaping (Goal Dist)": "#f1c40f",          # 黄
        "Baseline": "#2c3e50"                             # 黑
    }
    
    # 线型方案
    DASHES = {
        "Ours (Hierarchical Potential)": (1, 0),          # 实线
        "Simple Shaping (Goal Dist)": (3, 1),             # 虚线
        "Baseline": (1, 1)                                # 点线
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
        errorbar=None,       # 成功率通常不需要阴影，保持干净
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
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Task Success Rate: {ENV_NAME}", pad=20, weight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()
    
    # 根据文件夹自动命名保存
    output_name = "success_rate_8x8.png" if "8x8" in LOG_DIR else "success_rate_5x5.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ 成功率图表已生成！图片保存为: {output_name}")

if __name__ == "__main__":
    plot_success()