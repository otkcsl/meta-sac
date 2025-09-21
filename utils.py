import math
import torch
import json
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
import numpy as np
import os
import time


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def param_vector(net):
    # ネットワークの全パラメータを 1D ベクトルにして返す（CPU）
    return parameters_to_vector(list(net.parameters())).detach().cpu()

def param_l2_cosine(net_a, net_b):
    v_a = param_vector(net_a)
    v_b = param_vector(net_b)
    l2 = (v_a - v_b).norm().item()
    # cosine: avoid zero-division
    denom = (v_a.norm().item() * v_b.norm().item())
    cos = (float(F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item())
           if denom > 0 else 0.0)
    return l2, cos

def layerwise_l2(net_a, net_b):
    diffs = []
    for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
        diffs.append((p_a.detach().cpu() - p_b.detach().cpu()).norm().item())
    return diffs

def eval_critic(critic, states, actions):
    # states/actions: np.array -> torch.tensor
    with torch.no_grad():
        s = torch.as_tensor(states, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        try:
            q = critic(s, a)          # most implementations: critic(s,a) -> q
        except Exception:
            # fallback: critic may expect concatenated input
            inp = torch.cat([s, a], dim=-1)
            q = critic(inp)
        return q.squeeze().cpu().numpy()

def sample_states_from_memory(memory, n=1024):
    # try several attribute names
    for attr in ['storage', 'buffer', 'memory', 'data']:
        if hasattr(memory, attr):
            buf = getattr(memory, attr)
            break
    else:
        # fallback: if memory is list-like
        try:
            buf = memory
        except Exception:
            return np.zeros((0,))  # 取れない場合は空
    
    # buf expected as list of tuples (s,a,r,s2,mask) or dicts
    # collect last n states
    states = []
    for entry in buf[-n:]:
        if isinstance(entry, dict):
            states.append(entry['state'])
        else:
            # assume (s,a,r,s2,mask)
            states.append(entry[0])
    states = np.array(states)
    return states

def estimate_policy_entropy(agent, states, mc_samples=16):
    # states: np.array shape (N, obs_dim)
    s = torch.as_tensor(states, dtype=torch.float32)
    with torch.no_grad():
        # try analytic: policy returns (mu, log_std) or have attribute
        try:
            mu, log_std = agent.policy.get_mean_log_std(s)  # user impl
        except Exception:
            # try forward
            try:
                out = agent.policy(s)   # some impl return (mu, log_std, pre_tanh)
                mu, log_std = out[0], out[1]
            except Exception:
                mu, log_std = None, None

        if mu is not None and log_std is not None:
            # diagonal Gaussian entropy: 0.5 * (D*(1+ln(2pi)) + sum logvar)
            logvar = 2.0 * log_std
            ent = 0.5 * (mu.shape[1] * (1.0 + math.log(2.0 * math.pi)) + torch.sum(logvar, dim=1))
            return ent.mean().item()
        # fallback: empirical approx by sampling actions
        samples = []
        st_np = states
        for _ in range(mc_samples):
            acts = np.stack([agent.select_action(s_i, eval=False) for s_i in st_np])
            samples.append(acts)
        samples = np.stack(samples, axis=0)  # K x N x action_dim
        var = samples.var(axis=0) + 1e-8
        ent_approx = 0.5 * (samples.shape[-1] * (1 + math.log(2 * math.pi)) + np.log(var).sum(axis=1))
        return float(ent_approx.mean())

def agent_actions_for_states(agent, states, eval=True):
    # vector化されていない select_action をループで使う
    acts = []
    for s in states:
        a = agent.select_action(s, eval=eval)
        acts.append(a)
    return np.stack(acts)

def q_disagreement(agents_list, states):
    # returns pairwise mean abs differences and per-agent mean Q
    Qs = []
    for agent in agents_list:
        acts = agent_actions_for_states(agent, states, eval=True)
        q = eval_critic(agent.critic, states, acts)  # numpy array (N,)
        Qs.append(q)
    # pairwise mean abs diffs
    N = len(Qs)
    pairwise = []
    for i in range(N):
        for j in range(i+1, N):
            pairwise.append(float(np.mean(np.abs(Qs[i] - Qs[j]))))
    mean_qs = [float(np.mean(q)) for q in Qs]
    return pairwise, mean_qs

def log_metrics(i, step, agent, global_agent, memories, save_path):
    # agents_dict keys: 'agent0', 'agent1',...
    entry = {'step': step}
    # 1) パラメータ差分（policy と critic の L2 と cosine）
    l2_p, cos_p = param_l2_cosine(agent.policy, global_agent.policy)
    l2_c, cos_c = param_l2_cosine(agent.critic, global_agent.critic)
    entry['agent_policy_l2'] = l2_p
    entry['agent_policy_cos'] = cos_p
    entry['agent_critic_l2'] = l2_c
    entry['agent_critic_cos'] = cos_c

    # # 2) Qの不一致（軽め：固定states 256）
    # # choose a memory that is not empty
    # mem = list(memories.values())[0]
    # states = sample_states_from_memory(mem, n=256)
    # if states.size > 0:
    #     pairwise_qdiff, mean_qs = q_disagreement(agent, states)
    #     entry['q_pairwise_meanabs'] = pairwise_qdiff
    #     for i, mq in enumerate(mean_qs):
    #         entry[f'agent{i}_mean_q'] = mq

    # # 3) ポリシーエントロピー（近似）
    # if states.size > 0:
    #     for i, agent in enumerate(agents_list):
    #         ent = estimate_policy_entropy(agent, states[:128])
    #         entry[f'agent{i}_policy_entropy'] = ent

    # # 4) 報酬の分散（最新評価ログがあればavg_rewards等を参照）
    # # (あなたのコードでは avg_rewards があるため、それを参照してもよい)
    # # ここでは各エージェントの直近 eval を拾う例
    # for i in range(len(agents_list)):
    #     entry[f'agent{i}_last_avg_reward'] = (avg_rewards[i][-1] if len(avg_rewards[i])>0 else None)

    # 書き込み
    metrics_fp = os.path.join(save_path, f'metrics{i}.jsonl')
    with open(metrics_fp, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    # # 状態サンプルを別ファイルとして保存（オフライン解析用）
    # if states.size > 0:
    #     np.save(os.path.join(save_path, f'states_step_{step}.npy'), states)

def log_sum_metrics(step, agent_0, agent_1, memories, save_path):
    entry = {'step': step}
    l2_p, cos_p = param_l2_cosine(agent_0.policy, agent_1.policy)
    l2_c, cos_c = param_l2_cosine(agent_0.critic, agent_1.critic)
    entry['agent_policy_l2'] = l2_p
    entry['agent_policy_cos'] = cos_p
    entry['agent_critic_l2'] = l2_c
    entry['agent_critic_cos'] = cos_c
    
    metrics_fp = os.path.join(save_path, f'metrics_sum.jsonl')
    with open(metrics_fp, 'a') as f:
        f.write(json.dumps(entry) + '\n')