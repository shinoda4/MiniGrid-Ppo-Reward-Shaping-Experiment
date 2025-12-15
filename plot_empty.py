import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 绘图风格设置 (学术论文风) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",     # 衬线字体
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "lines.linewidth": 2.5
})

LOG_DIR = "./logs_paper_minigrid_empty_8x8_v0"

def load_data():
    all_data = []
    
    # 遍历文件夹寻找 monitor.csv
    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith("monitor.csv"):
                # 解析路径获取算法名称和种子
                # 假设路径结构: ./logs_paper/Baseline_seed101/0.monitor.csv
                folder_name = os.path.basename(root)
                if "Baseline" in folder_name:
                    algo = "Baseline (Sparse Reward)"
                elif "Ours" in folder_name:
                    algo = "Ours (Potential Shaping)"
                else:
                    continue
                
                # 读取数据
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, skiprows=1) # 跳过第一行元数据
                
                # 处理数据
                df['Algorithm'] = algo
                df['Timesteps'] = df['l'].cumsum() # 累积步数
                # 滑动平均平滑曲线 (让图更好看)
                df['Reward'] = df['r'].rolling(window=100, min_periods=10).mean()
                
                all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_empty():
    print("正在加载数据并绘图...")
    df = load_data()
    
    if df.empty:
        print("错误：未找到数据，请先运行训练脚本！")
        return

    plt.figure(figsize=(10, 6))
    
    # Seaborn 自动处理均值和标准差阴影 (Mean ± Std)
    sns.lineplot(
        data=df,
        x="Timesteps",
        y="Reward",
        hue="Algorithm",
        style="Algorithm",
        palette=["#7f8c8d", "#c0392b"], # 灰色对比红色(突出你的方法)
        errorbar='sd' # 显示标准差
    )
    
    plt.title("Learning Efficiency: MiniGrid-Empty-8x8", pad=20)
    plt.xlabel("Environment Interactions (Timesteps)")
    plt.ylabel("Average Episodic Reward (Raw)")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    
    # 保存结果
    plt.savefig("logs_paper_minigrid_empty_8x8_v0.png", dpi=300)
    print("✅ 绘图完成！图片已保存为 logs_paper_minigrid_empty_8x8_v0.png")
    # plt.show() # 如果在服务器上跑，注释掉这行

if __name__ == "__main__":
    plot_empty()