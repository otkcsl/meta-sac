"""v2：共有リプレイバッファと α 条件付き共有ネットワーク（v1 README の E7、v2 README の (a)+(b)）。

v1 の proposed_main.py を元にした別ファイルで、v1 の本体（proposed_main.py・sac.py・model.py）には触れない。
v1 と違うのは次の3点だけである。

- 温度ごとの SAC を作らず、1組のネットワーク（twin critic・target・policy）を全温度で共有する。
  温度は、正規化した log α を状態の後ろにつなげて入力する。ネットワークの形は model.py の QNetwork・GaussianPolicy を
  そのまま使い、入力の次元を1つ増やすだけにした
- 全ワーカーの遷移を1つの ReplayMemory に入れ、そこから学習する。容量は replay_size × ワーカー数
- 同期の式 (9)(10) は無い。τ_g・GSI・ASI は使わない

更新の回数：ワーカー i が1ステップ進むたびに、α_i の損失で updates_per_step 回更新する（2026-11-21、ユーザの決定）。
総数 N の時点で更新は N 回になり、SAC と同じである。

それ以外のロールアウト・評価・乱数の種の与え方・eval_metrics{i}.csv の形式は v1 と同じにしてある。
評価は温度ごとに π(·|s, c_i) で行い、eval_metrics{i}.csv の i はワーカー（＝温度）の番号である。

    ../../.venvs/mujoco/bin/python proposed_main_v2.py configs_pro/walker2d1.yml --seed 2021 --alpha 0.01,0.5 \\
        --num_steps 1500000 --exp_id e7_walker2d
"""

import argparse
import json
import os
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--cuda", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alpha", type=str, default=None)      # カンマ区切り "0.01,0.5"
parser.add_argument("--num_steps", type=int, default=None)  # ワーカー1つあたり
parser.add_argument("--exp_id", type=str, default=None)
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
if args.cuda is not None:
    config["cuda"] = args.cuda
if args.seed is not None:
    config["seed"] = args.seed
if args.alpha is not None:
    config["alpha"] = [float(a) for a in args.alpha.split(",")]
if args.num_steps is not None:
    config["num_steps"] = args.num_steps
if args.exp_id is not None:
    config["exp_id"] = args.exp_id
# v1 の同期は使わない。集計（aggregate_results.method_label）はこのキーで v2 と見分ける
config["teian"] = False
config["v2"] = 'shared_buffer+alpha_conditioned_net'
for key in ('tau_g', 'qtog', 'gtoq'):
    config.pop(key, None)

cores = str(config['cores'])
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = cores

import random

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import GaussianPolicy, QNetwork
from replay_memory import ReplayMemory
from utils import hard_update, soft_update

alphas = config['alpha']
n_workers = len(alphas)

# 温度の入力。log α を全温度の平均と標準偏差で正規化する。2温度なら −1 と +1 になる
log_alphas = np.log(np.array(alphas, dtype=np.float64))
spread = log_alphas.std()
conds = (log_alphas - log_alphas.mean()) / spread if spread > 0 else np.zeros_like(log_alphas)

save_path = 'models/' + config['exp_id'] + '/' + str(alphas) + '/v2net/' + str(config['seed']) + '/'
print(save_path)
os.makedirs(save_path, exist_ok=True)
config['alpha_condition'] = [float(c) for c in conds]
with open(os.path.join(save_path, 'experiment_summary.json'), 'w') as f:
    json.dump({'config_contents': config}, f, indent=2)
print(config)


class SharedSAC:
    """全温度で共有する1組の SAC。温度は状態の後ろにつなげた条件 c で区別する。sac.py の SAC と同じ更新式を、α と c を引数にして行う。"""

    def __init__(self, obs_dim, action_space, config):
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.target_update_interval = config['target_update_interval']
        self.device = torch.device('cuda:' + str(config['cuda'])) if torch.cuda.is_available() and config['cuda'] >= 0 else torch.device('cpu')
        n_in, n_act, hidden = obs_dim + 1, action_space.shape[0], config['hidden_size']
        self.qf1 = QNetwork(n_in, n_act, hidden).to(self.device)
        self.qf2 = QNetwork(n_in, n_act, hidden).to(self.device)
        self.critic_optim = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=config['lr'])
        self.qf1_target = QNetwork(n_in, n_act, hidden).to(self.device)
        self.qf2_target = QNetwork(n_in, n_act, hidden).to(self.device)
        hard_update(self.qf1_target, self.qf1)
        hard_update(self.qf2_target, self.qf2)
        self.policy = GaussianPolicy(n_in, n_act, hidden, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config['lr'])

    def _with_cond(self, state, cond):
        c = torch.full((state.shape[0], 1), float(cond), dtype=state.dtype, device=self.device)
        return torch.cat([state, c], 1)

    def select_action(self, state, cond, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, mean = self.policy.sample(self._with_cond(state, cond))
        return (mean if eval else action).detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, alpha, cond, updates):
        s, a, r, s2, m = memory.sample(batch_size=batch_size)
        s = self._with_cond(torch.FloatTensor(s).to(self.device), cond)
        s2 = self._with_cond(torch.FloatTensor(s2).to(self.device), cond)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        m = torch.FloatTensor(m).to(self.device).unsqueeze(1)

        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(s2)
            q_next = torch.min(self.qf1_target(s2, a2), self.qf2_target(s2, a2)) - alpha * logp2
            y = r + m * self.gamma * q_next
        qf1_loss = F.mse_loss(self.qf1(s, a), y)
        qf2_loss = F.mse_loss(self.qf2(s, a), y)
        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        pi, logp, _ = self.policy.sample(s)
        policy_loss = ((alpha * logp) - torch.min(self.qf1(s, pi), self.qf2(s, pi))).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.qf1_target, self.qf1, self.tau)
            soft_update(self.qf2_target, self.qf2, self.tau)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    def save_model(self, save_path, env_name):
        torch.save(self.policy.state_dict(), f'{save_path}actor_{env_name}_shared')
        torch.save(self.qf1.state_dict(), f'{save_path}q1_{env_name}_shared')
        torch.save(self.qf2.state_dict(), f'{save_path}q2_{env_name}_shared')


# 学習用の環境の種は、ワーカーごとに seed + i にする。v1 はすべての体で seed なので、ここだけ v1 と違う。
# v1 と同じにすると、ランダム行動の期間（start_steps）に全ワーカーがまったく同じ遷移を集め、共有バッファに同じ遷移が
# ワーカー数ぶん重複する。2026-11-21 の手元の試しでは、このとき critic が発散した。温度を (0.01, 0.01) にしても発散し、
# 種を変えると収まったので、温度の違いではなく重複が原因と判断した（v1 README の E7）。
# 評価用の環境は v1 と同じ seed のままにする。評価はエピソードごとに seed + j で始めるので、v1 と同じ初期状態で比べられる
envs, eval_envs = {}, {}
for i in range(n_workers):
    envs[i] = gym.make(config['env_name'])
    envs[i].reset(seed=config['seed'] + i)
    envs[i].action_space.seed(config['seed'] + i)
    envs[i].observation_space.seed(config['seed'] + i)
    eval_envs[i] = gym.make(config['env_name'])
    eval_envs[i].reset(seed=config['seed'])
    eval_envs[i].action_space.seed(config['seed'])
    eval_envs[i].observation_space.seed(config['seed'])

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

net = SharedSAC(envs[0].observation_space.shape[0], envs[0].action_space, config)
memory = ReplayMemory(config['replay_size'] * n_workers, config['seed'])

EVAL_COLS = ['step', 'seed_2021', 'seed_2022', 'seed_2023', 'seed_2024', 'seed_2025',
             'seed_2021_episode', 'seed_2022_episode', 'seed_2023_episode', 'seed_2024_episode', 'seed_2025_episode',
             'avg_reward', 'policy_loss', 'critic1_loss', 'critic2_loss', 'ent_loss', 'alpha']
records = {i: [] for i in range(n_workers)}
pushed = {i: 0 for i in range(n_workers)}  # ワーカーごとに共有バッファへ入れた数。終わりに表示して確かめる

states = {i: envs[i].reset()[0] for i in range(n_workers)}
dones = {i: False for i in range(n_workers)}
steps = {i: 0 for i in range(n_workers)}
ep_steps = {i: 0 for i in range(n_workers)}
test_step = {i: 1000 for i in range(n_workers)}
losses = {i: (0, 0, 0) for i in range(n_workers)}
updates = 0


def evaluate(i):
    rewards, lengths = [], []
    for j in range(5):
        s, _ = eval_envs[i].reset(seed=config['seed'] + j)
        total, n, done = 0.0, 0, False
        while not done:
            s, r, term, trunc, _ = eval_envs[i].step(net.select_action(s, conds[i], eval=True))
            total += r
            n += 1
            done = term or trunc
        rewards.append(total)
        lengths.append(n)
    q1, q2, pl = losses[i]
    records[i].append([steps[i] - 1] + rewards + lengths + [float(np.mean(rewards)), pl, q1, q2, 0.0, alphas[i]])
    print(f"worker{i} alpha={alphas[i]} step {steps[i]}  Avg. Reward: {np.mean(rewards):.2f}", flush=True)


while not all(steps[j] > config['num_steps'] for j in range(n_workers)):
    for i in range(n_workers):
        if steps[i] > config['num_steps']:
            continue
        if dones[i]:
            states[i], _ = envs[i].reset()
            dones[i] = False
            ep_steps[i] = 0

        if config['start_steps'] > steps[i]:
            action = envs[i].action_space.sample()
        else:
            action = net.select_action(states[i], conds[i])

        if len(memory) > config['batch_size']:
            for _ in range(config['updates_per_step']):
                losses[i] = net.update_parameters(memory, config['batch_size'], alphas[i], conds[i], updates)
                updates += 1

        next_state, reward, terminated, truncated, _ = envs[i].step(action)
        dones[i] = terminated or truncated
        ep_steps[i] += 1
        steps[i] += 1
        max_ep = getattr(envs[i], '_max_episode_steps', getattr(envs[i].spec, 'max_episode_steps', 1000))
        mask = 1 if ep_steps[i] == max_ep else float(not dones[i])
        memory.push(states[i], action, reward, next_state, mask)
        pushed[i] += 1
        states[i] = next_state

        if steps[i] > test_step[i] and config['eval']:
            test_step[i] += 1000
            evaluate(i)

for i in range(n_workers):
    pd.DataFrame(records[i], columns=EVAL_COLS).to_csv(os.path.join(save_path, f'eval_metrics{i}.csv'), index=False)
    envs[i].close()
    eval_envs[i].close()
net.save_model(save_path, config['env_name'])
print(f'共有バッファへ入れた数: {pushed}、バッファの中身 {len(memory)}、更新の回数 {updates}')
