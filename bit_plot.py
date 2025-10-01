import pandas as pd
import matplotlib.pyplot as plt
import json

# ==== JSONログをまとめてDataFrameに変換 ====
save_path = "models/walker2d/[0.0001, 0.1]/v1/2021"
csv0_path = "metrics0.jsonl"
csv1_path = "metrics1.jsonl"
json0_path = f"{save_path}/{csv0_path}"
json1_path = f"{save_path}/{csv1_path}"

# 改行ごとにJSONをパース
with open(json0_path, "r") as f:
    records0 = [json.loads(line) for line in f if line.strip()]  # 空行スキップ
df0 = pd.DataFrame(records0)

with open(json1_path, "r") as f:
    records1 = [json.loads(line) for line in f if line.strip()]
df1 = pd.DataFrame(records1)

# ==== 各項目をグラフ化 ====
x = df1["step"]

metrics = ["agent_policy_l2", "agent_policy_cos",
           "agent_critic_l2", "agent_critic_cos",
           "agent_critic_target_l2", "agent_critic_target_cos"]

for metric in metrics:
    plt.figure(figsize=(8,4))
    plt.plot(x, df0[metric], color="blue", linestyle="--", label=f"{metric} (metrics0)")
    plt.plot(x, df1[metric], color="orange", linestyle="-", alpha=0.7, label=f"{metric} (metrics1)")

    plt.xlabel("step")
    plt.ylabel(metric)
    if metric == "agent_critic_target_cos" or metric == "agent_critic_cos" or metric == "agent_policy_cos":
        plt.ylim(0.9998, 1.00001)  
    if metric == "agent_critic_l2":
        plt.ylim(0.7,2)  
    if metric == "agent_critic_target_l2":
        plt.ylim(0.5,1.5)  
    if metric == "agent_policy_l2":
        plt.ylim(1,2)  
    plt.title(f"{metric} vs step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{metric}.png")
