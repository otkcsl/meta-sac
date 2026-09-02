import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class SAC(object):
    def __init__(self, num_inputs, action_space, config, alpha):
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = alpha

        self.policy_type = config['policy']
        self.target_update_interval = config['target_update_interval']
        self.automatic_entropy_tuning = config['automatic_entropy_tuning']

        self.device = torch.device('cuda:' + str(config['cuda'])) if torch.cuda.is_available() and config['cuda'] >= 0 else torch.device('cpu')

        self.qf1 = QNetwork(num_inputs, action_space.shape[0], config['hidden_size']).to(device=self.device)
        self.qf2 = QNetwork(num_inputs, action_space.shape[0], config['hidden_size']).to(device=self.device)
        self.critic_optim = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=config['lr'])

        self.qf1_target = QNetwork(num_inputs, action_space.shape[0], config['hidden_size']).to(self.device)
        self.qf2_target = QNetwork(num_inputs, action_space.shape[0], config['hidden_size']).to(self.device)
        hard_update(self.qf1_target, self.qf1)
        hard_update(self.qf2_target, self.qf2)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp().item()
                self.alpha_optim = Adam([self.log_alpha], lr=config['lr'])

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], config['hidden_size'], action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config['lr'])

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # if updates == 50:
            #     print(self.alpha, reward_batch.mean(), next_state_action.mean())
            #     policy_weights = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
            #     print(policy_weights.mean().item())
            #     print(policy_weights.std().item())
            qf1_next_target = self.qf1_target(next_state_batch, next_state_action) 
            qf2_next_target = self.qf2_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1 = self.qf1(state_batch, action_batch)  
        qf2 = self.qf2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi = self.qf1(state_batch, pi)
        qf2_pi = self.qf2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            with torch.no_grad():
                _, log_pi, _ = self.policy.sample(state_batch)
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_tlogs = self.alpha
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.qf1_target, self.qf1, self.tau)
            soft_update(self.qf2_target, self.qf2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs

    # Save model parameters    
    def save_model(self, save_path = None, env_name = None, suffix = None):
        if save_path is None:
            save_path = './models/'

        actor_path = '{}actor_{}_{}'.format(save_path, env_name, suffix)
        q1_path = "{}q1_{}_{}".format(save_path, env_name, suffix)
        q2_path = "{}q2_{}_{}".format(save_path, env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, q1_path, q2_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.qf1.state_dict(), q1_path)
        torch.save(self.qf2.state_dict(), q2_path)

