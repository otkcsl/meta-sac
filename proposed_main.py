import sys
import yaml
import pandas as pd
config = yaml.safe_load(open(sys.argv[1])) # custom hyperparams
print(config)

import os
cores = str(config['cores'])
os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

import random
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
import torch.nn as nn
from proposed_sac import SAC
from replay_memory import ReplayMemory

env_names = {
    "Ant-v4": 'ant',
    "Hopper-v4": 'hopper',
    "HalfCheetah-v4": 'halfcheetah', 
    "Humanoid-v4": 'humanoid',
    "Walker2d-v4": 'walker2d',
    "Swimmer-v4": 'swimmer',
    "Reacher-v4": 'reacher'
}

import time


if config['exp_id'] != 'debug':
    log_dir = '../common/vanilla_SAC_log/{}/'.format(config['seed'])
    os.makedirs(log_dir, exist_ok=True)
    version = 'v1' if not config['automatic_entropy_tuning'] else 'v2'
    log_file = log_dir + env_names[config['env_name']] + '_' + version + '.txt'
    print(log_file)
    sys.stdout = open(log_file, 'w')

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
save_path = 'models/' + config['exp_id'] + '/' + str(config['alpha']) + '/' + version + '/' + str(config['seed']) + '/'
print(save_path)
os.makedirs(save_path, exist_ok=True)
    
print(config)
print(os.getpid())

def sync_gtoq_params(target, source):
    with torch.no_grad():
        # l_params_before = [p.clone().cpu() for p in target.parameters()]
        # g_params_before = [p.clone().cpu() for p in source.parameters()]
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(param)
        # l_params_after = [p.clone().cpu() for p in target.parameters()]
        # g_params_after = [p.clone().cpu() for p in source.parameters()]

        # for idx, (before, after) in enumerate(zip(g_params_before, g_params_after)):
        #     diff = (after - before).abs().sum().item()
        #     print(f"gtoq GlobalAgent param {idx}: diff = {diff:.6f}")
        # for idx, (before, after) in enumerate(zip(l_params_before, l_params_after)):
        #     diff = (after - before).abs().sum().item()
        #     print(f"gtoq LocalAgent param {idx}: diff = {diff:.6f}")

def sync_qtog_params(target, source):
    with torch.no_grad():
        # l_params_before = [p.clone().cpu() for p in target.parameters()]
        # g_params_before = [p.clone().cpu() for p in source.parameters()]
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(config['tau_g'] * param + (1 - config['tau_g']) * target_param)
        # l_params_after = [p.clone().cpu() for p in target.parameters()]
        # g_params_after = [p.clone().cpu() for p in source.parameters()]

        # for idx, (before, after) in enumerate(zip(g_params_before, g_params_after)):
        #     diff = (after - before).abs().sum().item()
        #     print(f"qtog LocalAgent param {idx}: diff = {diff:.6f}")
        # for idx, (before, after) in enumerate(zip(l_params_before, l_params_after)):
        #     diff = (after - before).abs().sum().item()
        #     print(f"qtog GlobalAgent param {idx}: diff = {diff:.6f}")

def run(agent, memory, env, config, total_step, state, done, episode_reward, test, test_rewards, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, updates, agent_acc_log_alpha, glo, index):
    if config['teian'] == True:
        if (total_step + 1) % config['qtog'] == 0:
            sync_qtog_params(glo.critic, agent.critic)
            sync_qtog_params(glo.critic_target, agent.critic_target)
            sync_qtog_params(glo.policy, agent.policy)

        if (total_step + 1) % config['gtoq'] == 0:
            sync_gtoq_params(agent.critic, glo.critic)
            sync_gtoq_params(agent.critic_target, glo.critic_target)
            sync_gtoq_params(agent.policy, glo.policy)

    if len(memory) > config['batch_size']:
        for _ in range(config['updates_per_step']):
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, config['batch_size'], updates)
            updates += 1
            agent_acc_log_alpha += np.log(alpha)

    if config['start_steps'] > total_step:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)
    
    step_result = env.step(action)
    if len(step_result) == 5:  
        next_state, reward, terminated, truncated, _ = step_result
        done = terminated or truncated
    else: 
        next_state, reward, done, _ = step_result
        
    total_step += 1
    episode_reward += reward

    max_episode_steps = getattr(env, '_max_episode_steps', getattr(env.spec, 'max_episode_steps', 1000))
    mask = 1 if total_step == max_episode_steps else float(not done)

    memory.push(state, action, reward, next_state, mask)
    state = next_state

    if total_step % config['test_step'] == 0 and config['eval'] == True:
        avg_reward = 0.
        episodes = 5
        test_part_reward = []
        for j in range(episodes):
            test_reset_result = env.reset(seed=config['seed']+ j)
            if isinstance(test_reset_result, tuple):
                test_state, _ = test_reset_result
            else:
                test_state = test_reset_result
                
            test_episode_reward = 0
            test_done = False
            while not test_done:
                test_action = agent.select_action(test_state, eval=True)
                
                test_step_result = env.step(test_action)
                if len(test_step_result) == 5:
                    test_next_state, test_reward, test_terminated, test_truncated, _ = test_step_result
                    test_done = test_terminated or test_truncated
                else:
                    test_next_state, test_reward, test_done, _ = test_step_result
                    
                test_episode_reward += test_reward
                test_state = test_next_state
            test_part_reward.append(test_episode_reward)
        print(test_part_reward)

        avg_reward = np.mean(test_part_reward)

        temp_step[index].append(total_step)
        seed_2021[index].append(test_part_reward[0])
        seed_2022[index].append(test_part_reward[1])
        seed_2023[index].append(test_part_reward[2])
        seed_2024[index].append(test_part_reward[3])
        seed_2025[index].append(test_part_reward[4])
        avg_rewards[index].append(avg_reward)
        policy_losses[index].append(policy_loss if 'policy_loss' in locals() else None)
        critic_1_losses[index].append(critic_1_loss if 'critic_1_loss' in locals() else None)
        critic_2_losses[index].append(critic_2_loss if 'critic_2_loss' in locals() else None)
        ent_losses[index].append(ent_loss if 'ent_loss' in locals() else None)
        sum_alphas[index].append(alpha if 'alpha' in locals() else None)

        print("----------------------------------------")
        print("Total_step: {}, Avg. Reward: {}".format(total_step, round(avg_reward, 2)))
        if config['automatic_entropy_tuning']:
            print("Test Log Alpha: {}".format(agent.log_alpha.item()))
        print("----------------------------------------")
    return total_step, state, done, episode_reward, updates, agent_acc_log_alpha, alpha

envs = {}
for i in range(len(config['alpha'])):
    envs[i] = gym.make(config['env_name'])
    # envs[i].reset(seed=config['seed'])
    envs[i].action_space.seed(config['seed'])
    envs[i].observation_space.seed(config['seed'])

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

global_agent = {
    f'global': SAC(envs[0].observation_space.shape[0], envs[0].action_space, config, config['alpha'][0])
}
agents = {
    f'agent{i}': SAC(envs[i].observation_space.shape[0], envs[i].action_space, config, config['alpha'][i])
    for i in range(len(config['alpha']))
}
memories = {
    f'memory{i}': ReplayMemory(config['replay_size'])
    for i in range(len(config['alpha']))
}

if config['teian'] == True:
    for i in range(len(agents)):
        print(f'sync agent{i} parameters')
        sync_gtoq_params(agents[f'agent{i}'].critic, global_agent['global'].critic)
        sync_gtoq_params(agents[f'agent{i}'].critic_target, global_agent['global'].critic_target)
        sync_gtoq_params(agents[f'agent{i}'].policy, global_agent['global'].policy)

temp_step = [[] for _ in range(len(agents))]
seed_2021 = [[] for _ in range(len(agents))]
seed_2022 = [[] for _ in range(len(agents))]
seed_2023 = [[] for _ in range(len(agents))]
seed_2024 = [[] for _ in range(len(agents))]
seed_2025 = [[] for _ in range(len(agents))]
avg_rewards = [[] for _ in range(len(agents))]
policy_losses = [[] for _ in range(len(agents))]
critic_1_losses = [[] for _ in range(len(agents))]
critic_2_losses = [[] for _ in range(len(agents))]
ent_losses = [[] for _ in range(len(agents))]
sum_alphas = [[] for _ in range(len(agents))]

agent_states = {}
agent_dones = {}
agent_episode_reward = {}
agent_episode_steps = {}
agent_acc_log_alpha = {}
policy_loss = {}
critic_1_loss = {}
critic_2_loss = {}
ent_loss = {}
alpha = {}
updates = {}

total_steps = {i: 0 for i in range(len(agents))}
test_rewards = {i: [] for i in range(len(agents))}
test = {i: [] for i in range(len(agents))}

for i in range(len(agents)):
    agent_states[i], _ = envs[i].reset(seed = 42 + total_steps[i])
    agent_dones[i] = False
    agent_episode_reward[i] = 0
    agent_episode_steps[i] = 0
    agent_acc_log_alpha[i] = 0.
    updates[i] = 0
    critic_1_loss[i] = 0
    critic_2_loss[i] = 0
    policy_loss[i] = 0
    ent_loss[i] = 0
    alpha[i] = 0

while not all(total_steps[j] >= config['num_steps'] for j in range(len(agents))):
    for i in range(len(agents)):
        print(f"agent{i}: total_steps {total_steps[i]}")
        if total_steps[i] >= config['num_steps']:
            continue

        if agent_dones[i]:
            agent_states[i], _ = envs[i].reset(seed=42 + total_steps[i])
            agent_dones[i] = False
            agent_episode_reward[i] = 0
        total_steps[i], agent_states[i], agent_dones[i], agent_episode_reward[i] , updates[i], agent_acc_log_alpha[i], alpha[i]= \
            run(agents[f"agent{i}"], memories[f"memory{i}"], envs[i], config, total_steps[i], agent_states[i], \
                agent_dones[i], agent_episode_reward[i], test[i], test_rewards[i], \
                critic_1_loss[i], critic_2_loss[i], policy_loss[i], ent_loss[i], alpha[i], updates[i], agent_acc_log_alpha[i], global_agent['global'], i)

for i in range(len(agents)):
    df_eval = pd.DataFrame({
        'step': temp_step[i],
        'seed_2021': seed_2021[i],
        'seed_2022': seed_2022[i],
        'seed_2023': seed_2023[i],
        'seed_2024': seed_2024[i],
        'seed_2025': seed_2025[i],
        'avg_reward': avg_rewards[i],
        'policy_loss': policy_losses[i],
        'critic1_loss': critic_1_losses[i],
        'critic2_loss': critic_2_losses[i],
        'ent_loss': ent_losses[i],
        'alpha': sum_alphas[i]
    })
    df_eval.to_csv(os.path.join(save_path, f'eval_metrics{i}.csv'), index=False)

for i in range(len(agents)):
    envs[i].close()


# env.seed(config['seed'])
# env.action_space.np_random.seed(config['seed'])

# l_params_before = [p.clone().cpu() for p in target.parameters()]
# g_params_before = [p.clone().cpu() for p in source.parameters()]
# l_params_after = [p.clone().cpu() for p in target.parameters()]
# g_params_after = [p.clone().cpu() for p in source.parameters()]

# for idx, (before, after) in enumerate(zip(g_params_before, g_params_after)):
#     diff = (after - before).abs().sum().item()
#     print(f"GlobalAgent param {idx}: diff = {diff:.6f}")
# for idx, (before, after) in enumerate(zip(l_params_before, l_params_after)):
#     diff = (after - before).abs().sum().item()
#     print(f"LocalAgent param {idx}: diff = {diff:.6f}")
