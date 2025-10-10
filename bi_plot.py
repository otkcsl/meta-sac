import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパスを指定
save_path = "models/humanoid/[0.0001, 0.1]/v1/2021"
csv0_path = "eval_metrics0.csv"
csv1_path = "eval_metrics1.csv"

# DataFrame 作成 + 数値変換
df0 = pd.read_csv(f"{save_path}/{csv0_path}")
numeric_cols = ['step','avg_reward','policy_loss','critic1_loss','critic2_loss','ent_loss','alpha',
                'seed_2021','seed_2022','seed_2023','seed_2024','seed_2025']
df0[numeric_cols] = df0[numeric_cols].apply(pd.to_numeric, errors='coerce')

# seed 列の最小/最大を計算
seed_cols = ['seed_2021','seed_2022','seed_2023','seed_2024','seed_2025']
df0['seed_min'] = df0[seed_cols].min(axis=1)
df0['seed_max'] = df0[seed_cols].max(axis=1)

# DataFrame 作成 + 数値変換
df1 = pd.read_csv(f"{save_path}/{csv1_path}")
numeric_cols = ['step','avg_reward','policy_loss','critic1_loss','critic2_loss','ent_loss','alpha',
                'seed_2021','seed_2022','seed_2023','seed_2024','seed_2025']
df1[numeric_cols] = df1[numeric_cols].apply(pd.to_numeric, errors='coerce')

# seed 列の最小/最大を計算
seed_cols = ['seed_2021','seed_2022','seed_2023','seed_2024','seed_2025']
df1['seed_min'] = df1[seed_cols].min(axis=1)
df1['seed_max'] = df1[seed_cols].max(axis=1)

# avg_reward と min/max バンドの図
x = df0['step'].values.astype(float)
plt.figure(figsize=(8,5))
plt.plot(x, df0['avg_reward'].values.astype(float), marker='o', label='avg_reward0')
plt.fill_between(x, df0['seed_min'].values.astype(float), df0['seed_max'].values.astype(float),
                 alpha=0.25, label='seed range (min-max)')
plt.plot(x, df1['avg_reward'].values.astype(float), marker='x', label='avg_reward1')
plt.fill_between(x, df1['seed_min'].values.astype(float), df1['seed_max'].values.astype(float),
                 alpha=0.25, label='seed range (min-max)')
plt.xlabel('step')
plt.ylabel('avg_reward')
plt.title('avg_reward vs step with seed min-max band')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_path}/avg_reward_plot.png")

# 個別の図（policy_loss, critic1_loss, critic2_loss, ent_loss, alpha）
metrics = ['policy_loss', 'critic1_loss', 'critic2_loss', 'ent_loss', 'alpha']
for metric in metrics:
    plt.figure(figsize=(8,4))
    plt.plot(x, df0[metric].values.astype(float), marker='o', label=metric)
    plt.plot(x, df1[metric].values.astype(float), marker='x', label=metric)
    plt.xlabel('step')
    plt.ylabel(metric)
    plt.title(f'{metric} vs step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{metric}.png")
