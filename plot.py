import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# CSV読み込み
save = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4]

# 色リスト（6色）
colors = plt.cm.viridis(np.linspace(0, 1, len(save)))

plt.figure(figsize=(8,5))  # 一度だけ作成

# 移動平均の窓サイズ（調整してください）
window = 50  

for idx, i in enumerate(save):
    df = pd.read_csv(f"models/walker2d/{i}/v1/2021/eval_metrics.csv")
    seeds = ["seed_2021", "seed_2022", "seed_2023", "seed_2024", "seed_2025"]
    df["mean"] = df[seeds].mean(axis=1)
    df["std"] = df[seeds].std(axis=1)

    # --- 移動平均 ---
    df["mean_smooth"] = df["mean"].rolling(window, min_periods=1).mean()
    df["std_smooth"] = df["std"].rolling(window, min_periods=1).mean()

    # プロット
    plt.plot(df["step"], df["mean_smooth"], label=f"alpha={i}", color=colors[idx])
    plt.fill_between(df["step"], 
                     df["mean_smooth"] - df["std_smooth"], 
                     df["mean_smooth"] + df["std_smooth"], 
                     alpha=0.2, color=colors[idx])
    
    last_mean = df["mean"].tail(5).mean()
    print(last_mean)    

    # v2 のデータもあれば同様に処理
    if i == 0.2:
        df_v2 = pd.read_csv(f"models/walker2d/{i}/v2/2021/eval_metrics.csv")
        df_v2["mean"] = df_v2[seeds].mean(axis=1)
        df_v2["std"] = df_v2[seeds].std(axis=1)
        df_v2["mean_smooth"] = df_v2["mean"].rolling(window, min_periods=1).mean()
        df_v2["std_smooth"] = df_v2["std"].rolling(window, min_periods=1).mean()

        plt.plot(df_v2["step"], df_v2["mean_smooth"], label=f"sac-v2", color="red")
        plt.fill_between(df_v2["step"], 
                         df_v2["mean_smooth"] - df_v2["std_smooth"], 
                         df_v2["mean_smooth"] + df_v2["std_smooth"], 
                         alpha=0.2, color="red")
        last_mean = df_v2["mean"].tail(5).mean()
        print(last_mean) 

plt.xlabel("Time Steps")
plt.ylabel("Average Test Returns")
plt.ylim(0,6000)
plt.legend()
plt.grid(True)
plt.savefig("plot.png")
plt.show()
