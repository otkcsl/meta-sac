import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
config = yaml.safe_load(open(sys.argv[1]))
version = 'v1' if not config['automatic_entropy_tuning'] else 'v2'
save_path = 'models/' + config['exp_id'] + '/' + version + '/' + str(config['seed']) + '/'

df = pd.read_csv(f'{save_path}/eval_metrics.csv') 

print(df)

plt.plot(df['step'], df['avg_reward'], label='Average Reward')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title('Average Reward over Training')
plt.legend()
plt.savefig(f"{save_path}/average_reward_plot.png")

plt.figure(figsize=(10,6))
plt.plot(df['step'], df['avg_reward'], label='Average Reward')
plt.plot(df['step'], df['policy_loss'], label='Policy Loss')
plt.xlabel('Step')
plt.ylabel('Metric Value')
plt.title('Training Metrics over Time')
plt.legend()
plt.savefig(f"{save_path}/mix_plot.png")
