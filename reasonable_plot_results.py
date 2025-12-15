import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 学术论文绘图风格设置 ---
# 使用 Seaborn 的 'paper' 上下文，这会自动调整字体大小适合论文
sns.set_theme(context="paper", style="whitegrid") 

plt.rcParams.update({
    "font.family": "serif",          # 衬线体 (如 Times New Roman)
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,              # 保证导出 PDF 时字体可编辑
    "ps.fonttype": 42
})

LOG_DIR = "./logs_paper_minigrid_empty_8x8_v0"
MAX_STEPS = 256  # [重要] 请确认你环境的 max_steps (Empty-8x8通常是 4*8*8=256 或 100)

def compute_clean_reward(row):
    """
    根据步数重构标准的 MiniGrid 奖励，去除 Shaping 的影响。
    逻辑：在 Empty 环境中，除非超时(steps >= max_steps)，否则视为成功。
    """
    steps = row['l']
    if steps < MAX_STEPS:
        # 成功到达目标
        return 1 - 0.9 * (steps / MAX_STEPS)
    else:
        # 超时 (失败)
        return 0.0

def load_data():
    all_data = []
    
    print(f"正在读取数据，假设环境 MAX_STEPS = {MAX_STEPS} ...")
    
    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                folder_name = os.path.basename(root)
                
                # 区分算法
                if "Baseline" in folder_name:
                    algo = "Baseline (Sparse Reward)"
                elif "Ours" in folder_name:
                    algo = "Ours (Potential Shaping)"
                else:
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    # monitor.csv 第一行通常是元数据，skiprows=1 读取 header
                    df = pd.read_csv(file_path, skiprows=1)
                    
                    # 检查是否有必要的数据列
                    if 'l' not in df.columns:
                        print(f"警告: {file_path} 缺少 'l' (length) 列，跳过。")
                        continue

                    df['Algorithm'] = algo
                    df['Timesteps'] = df['l'].cumsum()
                    
                    # --- 核心修改：重构奖励 ---
                    # 不直接使用 df['r']，因为那里面可能包含 Shaping Reward
                    # 我们用步数 'l' 倒推标准奖励
                    df['Clean_Reward'] = df.apply(compute_clean_reward, axis=1)
                    
                    # 滑动平均 (Smoothing)
                    # window=100 表示对最近100个 episode 取平均，使曲线更平滑
                    df['Smoothed_Reward'] = df['Clean_Reward'].rolling(window=100, min_periods=10).mean()
                    
                    # 也可以计算成功率 (作为备选指标)
                    df['Success'] = (df['l'] < MAX_STEPS).astype(float)
                    df['Success_Rate'] = df['Success'].rolling(window=100, min_periods=10).mean()

                    all_data.append(df)
                    
                except Exception as e:
                    print(f"读取文件 {file_path} 出错: {e}")
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_paper_figure():
    df = load_data()
    
    if df.empty:
        print("❌ 错误：未找到有效数据，请检查路径或文件名。")
        return

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 5)) # 典型的单栏论文插图比例
    
    # 绘图：使用 'Smoothed_Reward' (重构后的标准奖励)
    # sns.lineplot(
    #     data=df,
    #     x="Timesteps",
    #     y="Smoothed_Reward",
    #     hue="Algorithm",
    #     style="Algorithm",     # 线型也会随算法变化 (实线/虚线)
    #     palette=["#535c68", "#c0392b"], # 灰色(Baseline) vs 深红色(Ours) - 这种配色打印黑白也能看清
    #     errorbar='sd',         # 绘制标准差阴影
    #     linewidth=2,
    #     ax=ax
    # )

    sns.lineplot(
        data=df,
        x="Timesteps",
        y="Smoothed_Reward",
        hue="Algorithm",
        style="Algorithm",
        # 这是一个字典：Key是算法名，Value是线型
        # (True, False) 无效，通常用 (2, 2) 表示虚线， "" 或 tuple() 表示实线
        dashes={"Ours (Potential Shaping)": (1, 0), "Baseline (Sparse Reward)": (2, 2)}, 
        palette=["#7f8c8d", "#c0392b"], 
        errorbar='sd',
        ax=ax
    )
    
    # 设置坐标轴范围 (强制 Y 轴在 0-1 之间，符合 MiniGrid 定义)
    ax.set_ylim(-0.05, 1.05)
    
    # 标题与标签
    ax.set_title("Learning Efficiency: MiniGrid-Empty-8x8", pad=15, weight='bold')
    ax.set_xlabel("Environment Interactions (Timesteps)")
    ax.set_ylabel("Average Episodic Reward (Normalized)")
    
    # 网格线优化
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 图例优化
    # 去掉图例标题，放在右下角
    ax.legend(title=None, loc="lower right", frameon=True, framealpha=0.9, fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    
    # 保存结果
    save_path = "logs_paper_minigrid_empty_8x8_v0_fixed.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！\n图片已保存为: {save_path}")
    print("注意：该图使用了基于步数重构的奖励，剔除了 Potential Shaping 的数值影响，范围应在 [0, 1] 之间。")

if __name__ == "__main__":
    plot_paper_figure()