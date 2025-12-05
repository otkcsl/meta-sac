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
save = [0.0001, 0.01, 0.05, 0.07965915650129318, 0.1, 0.2]

# 色リスト（6色）
colors = plt.cm.viridis(np.linspace(0, 1, len(save)))

plt.figure(figsize=(8,5))  # 一度だけ作成

# 移動平均の窓サイズ（調整してください）
window = 100  

for idx, i in enumerate(save):
    df = pd.read_csv(f"models/humanoid/{i}/v1/2021/eval_metrics.csv")
    seeds = ["seed_2021", "seed_2022", "seed_2023", "seed_2024", "seed_2025"]
    df["mean"] = df[seeds].mean(axis=1)
    df["std"] = df[seeds].std(axis=1)

    # --- 移動平均 ---
    df["mean_smooth"] = df["mean"].rolling(window, min_periods=1).mean()
    df["std_smooth"] = df["std"].rolling(window, min_periods=1).mean()

    # プロット
    plt.plot(df["step"], df["mean_smooth"], label=f"alpha={i}", color=colors[idx], linewidth=3)
    plt.fill_between(df["step"], 
                    df["mean_smooth"] - df["std_smooth"], 
                    df["mean_smooth"] + df["std_smooth"], 
                    alpha=0.2, color=colors[idx])
    
    last_mean = df["mean"].tail(5).mean()
    print(i, last_mean)    

    # v2 のデータもあれば同様に処理
    if i == 0.05:
        df_v2 = pd.read_csv(f"models/humanoid/{i}/v2/2021/eval_metrics.csv")
        df_v2["mean"] = df_v2[seeds].mean(axis=1)
        df_v2["std"] = df_v2[seeds].std(axis=1)
        df_v2["mean_smooth"] = df_v2["mean"].rolling(window, min_periods=1).mean()
        df_v2["std_smooth"] = df_v2["std"].rolling(window, min_periods=1).mean()

        plt.plot(df_v2["step"], df_v2["mean_smooth"], label=f"sac-v2", color="red", linewidth=3)
        plt.fill_between(df_v2["step"], 
                         df_v2["mean_smooth"] - df_v2["std_smooth"], 
                         df_v2["mean_smooth"] + df_v2["std_smooth"], 
                         alpha=0.2, color="red")
        last_mean = df_v2["mean"].tail(5).mean()
        print(i, last_mean) 

df0 = pd.read_csv(f"models/humanoid/[0.0001, 0.1]/v1/2021/eval_metrics0.csv")
df1 = pd.read_csv(f"models/humanoid/[0.0001, 0.1]/v1/2021/eval_metrics1.csv")
seeds = ["seed_2021", "seed_2022", "seed_2023", "seed_2024", "seed_2025"]
df0["mean"] = df0[seeds].mean(axis=1)
df0["std"] = df0[seeds].std(axis=1)
df1["mean"] = df1[seeds].mean(axis=1)
df1["std"] = df1[seeds].std(axis=1)

# --- 移動平均 ---
df0["mean_smooth"] = df0["mean"].rolling(window, min_periods=1).mean()
df0["std_smooth"] = df0["std"].rolling(window, min_periods=1).mean()
df1["mean_smooth"] = df1["mean"].rolling(window, min_periods=1).mean()
df1["std_smooth"] = df1["std"].rolling(window, min_periods=1).mean()

# プロット
plt.plot(df0["step"], df0["mean_smooth"], label=f"sac_v3_0.0001", color='black', linewidth=3)
plt.plot(df1["step"], df1["mean_smooth"], label=f"sac_v3_0.1", color='brown', linewidth=3)
plt.fill_between(df0["step"], 
                df0["mean_smooth"] - df0["std_smooth"], 
                df0["mean_smooth"] + df0["std_smooth"], 
                alpha=0.2, color='black')
plt.fill_between(df1["step"], 
                df1["mean_smooth"] - df1["std_smooth"], 
                df1["mean_smooth"] + df1["std_smooth"], 
                alpha=0.2, color='brown')

last_mean = df0["mean"].tail(5).mean()
print(0.0001, last_mean) 
last_mean = df1["mean"].tail(5).mean()
print(0.1, last_mean) 
    
plt.xlabel("Time Steps")
plt.ylabel("Average Test Returns")
#plt.ylim(-12, -2)
plt.legend()
plt.grid(True)
plt.savefig("humanoid_plot.png")
