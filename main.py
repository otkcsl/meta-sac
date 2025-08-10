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
from sac import SAC
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
    dir = '../common/vanilla_SAC_log/{}/'.format(config['seed'])
    os.makedirs(dir, exist_ok=True)
    version = 'v1' if not config['automatic_entropy_tuning'] else 'v2'
    log_file = dir + env_names[config['env_name']] + '_' + version + '.txt'
    print(log_file)
    sys.stdout = open(log_file, 'w')

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
save_path = 'models/' + config['exp_id'] + '/' + version + '/' + str(config['seed']) + '/'
print(save_path)
os.makedirs(save_path)
    
print(config)
print(os.getpid())

env = gym.make(config['env_name'])
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
env.reset(seed=config['seed'])
#env.seed(config['seed'])
#env.action_space.np_random.seed(config['seed'])
env.action_space.seed(config['seed'])
env.observation_space.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

agent = SAC(env.observation_space.shape[0], env.action_space, config)

memory = ReplayMemory(config['replay_size'])

temp_step = []
ave_re = []
avg_rewards = []
policy_losses = []
critic_1_losses = []
critic_2_losses = []
ent_losses = []
sum_alphas = []
total_numsteps = 0
updates = 0
test_step = 1000

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    
    reset_result = env.reset(seed = 42 + total_numsteps)
    if isinstance(reset_result, tuple):
        state, _ = reset_result  
    else:
        state = reset_result 

    acc_log_alpha = 0.
    while not done:
        if config['start_steps'] > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if len(memory) > config['batch_size']:
            for i in range(config['updates_per_step']):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, config['batch_size'], updates)
                updates += 1
                acc_log_alpha += np.log(alpha)

        step_result = env.step(action)
        if len(step_result) == 5:  
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else: 
            next_state, reward, done, _ = step_result
            
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        max_episode_steps = getattr(env, '_max_episode_steps', getattr(env.spec, 'max_episode_steps', 1000))
        mask = 1 if episode_steps == max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask)
        state = next_state

        if total_numsteps > test_step and config['eval'] == True:
            test_step += 1000
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                reset_result = env.reset(seed=config['seed'])
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
                    
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, eval=True)
                    
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = step_result
                        
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            ave_re.append(avg_reward)

            temp_step.append(total_numsteps-1)
            avg_rewards.append(avg_reward)
            policy_losses.append(policy_loss if 'policy_loss' in locals() else None)
            critic_1_losses.append(critic_1_loss if 'critic_1_loss' in locals() else None)
            critic_2_losses.append(critic_2_loss if 'critic_2_loss' in locals() else None)
            ent_losses.append(ent_loss if 'ent_loss' in locals() else None)
            sum_alphas.append(alpha if 'alpha' in locals() else None)

            print("----------------------------------------")
            print("Total_numsteps: {}, Avg. Reward: {}".format(total_numsteps, round(avg_reward, 2)))
            if config['automatic_entropy_tuning']:
                print("Test Log Alpha: {}".format(agent.log_alpha.item()))
            print("----------------------------------------")

    if total_numsteps > config['num_steps']:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {} mean log alpha {}".format(
        i_episode, total_numsteps, episode_steps, round(episode_reward, 2), acc_log_alpha / episode_steps
        ))

df_eval = pd.DataFrame({
    'step': temp_step,
    'avg_reward': avg_rewards,
    'policy_loss': policy_losses,
    'critic1_loss': critic_1_losses,
    'critic2_loss': critic_2_losses,
    'ent_loss': ent_losses,
    'alpha': sum_alphas
})
df_eval.to_csv(os.path.join(save_path, 'eval_metrics.csv'), index=False)

df = pd.DataFrame({'average_reward': ave_re})
file_path = os.path.join(save_path, 'average_rewards.csv')
df.to_csv(file_path, index=False)
agent.save_model(save_path, config['env_name'], suffix = None)
env.close()

