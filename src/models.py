import torch as tr
import torch.nn as nn
from torch.distributions.normal import Normal
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torchvision.ops import MLP
from utils import he_initialization_leaky_relu
from utils import time_it

class Critic(nn.Module):
    def __init__(self, config: dict):
        super(Critic, self).__init__()
        hidden_dims = [config['hidden_dims_critic']] * config['num_hidden_layers_critic'] + [1]
        self.net = MLP(
            config['obs_dim'] + config['action_dim'],
            hidden_dims,
            activation_layer=nn.LeakyReLU,
            #dropout=config['dropout_critic']
        )
        self.net.apply(he_initialization_leaky_relu)

    def forward(self, state, action):
        x = tr.cat([state, action], dim=1)
        return self.net(x)


class TwinCritic(nn.Module):
    def __init__(self, config: dict):
        super(TwinCritic, self).__init__()
        self.critic1 = Critic(config)
        self.critic2 = Critic(config)

    def forward(self, state, action):
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2


class SquashedGaussianActor(nn.Module):
    def __init__(self, config: dict):
        super(SquashedGaussianActor, self).__init__()
        self.act_scaling = tr.tensor(config['act_scaling'], dtype=tr.float64)
        hidden_dims = [config['hidden_dims_actor']] * config['num_hidden_layers_actor']
        self.net = MLP(
            config['obs_dim'],
            hidden_dims,
            #norm_layer=nn.LayerNorm,
            activation_layer=nn.LeakyReLU,
            #dropout=config['dropout_actor']
        )
        self.net.apply(he_initialization_leaky_relu)

        self.mu_layer = nn.Linear(hidden_dims[-1], config['action_dim'])
        self.log_std_layer = nn.Linear(hidden_dims[-1], config['action_dim'])
        self.log_std_min = -2
        self.log_std_max = 20

    def forward(self, state, deterministic=False, with_logprob=False):
        assert not tr.isnan(state).any(), f"State contains NaNs"
        x = self.net(state)
        assert not tr.isnan(x).any(), f"x contains NaNs with dtypes {x.dtype}, {state.dtype}"
        mu = self.mu_layer(x)
        assert not tr.isnan(mu).any(), f"Mu contains NaNs"

        if deterministic:
            action = mu
            pi_action = self.act_scaling * tr.tanh(action)
        else:
            log_std = self.log_std_layer(x)
            log_std = tr.clamp(log_std, self.log_std_min, self.log_std_max)
            std = tr.exp(log_std)
            action = mu + std * tr.randn_like(std)
            pi_action = self.act_scaling * tr.tanh(action)
            assert not tr.isnan(pi_action).any(), f"Action contains NaNs: {action}, {pi_action}"

            if with_logprob:
                log_prob = Normal(mu, std).log_prob(action)
                log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action)))
                log_prob -= tr.log(self.act_scaling)
                
                return pi_action, log_prob

            else: return pi_action
            
        return pi_action





class SoftActorCritic(nn.Module):
    def __init__(self, config: dict, device):
        super().__init__()
        self.entropy_coeff = tr.tensor(
            config['entropy_coeff'],
            requires_grad=False
        ).to(device)

        self.log_alpha = tr.log(self.entropy_coeff).to(device)
        self.target_entropy = -config['action_dim']
        self.entropy_lr = config['learning_rate_entropy']
        self.gamma = config['future_discount_factor']
        self.polyak_tau = config['polyak_tau']
        self.device = device
        self.loss = nn.MSELoss(reduction='none')
        self.grad_clip_critic = config["grad_clip_critic"]
        self.grad_clip_actor = config["grad_clip_actor"]

        self.actor = SquashedGaussianActor(config)
        self.critic = TwinCritic(config)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = tr.optim.Adam(
            self.actor.parameters(),
            lr=config['learning_rate_actor']
            )
        self.critic_optimizer = tr.optim.Adam(
            self.critic.parameters(),
            lr=config['learning_rate_critic']
            )

    def act(self, state: tr.Tensor, deterministic=True):
        with tr.no_grad():
            state = tr.from_numpy(state).float().to(self.device)
            action = self.actor(state, deterministic=deterministic)
        return action.cpu().detach().numpy()

    def _update_target_network(self):
        with tr.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_target.data.mul_(self.polyak_tau)
                p_target.data.add_((1 - self.polyak_tau) * p.data)

    def update(self, sample):
        # Sample from the buffer
        state = sample['obs'].to(self.device)
        action = sample['action'].to(self.device)
        reward = sample['reward'].to(self.device)
        next_state = sample['next_obs'].to(self.device)
        done = sample['done'].to(self.device)

        # Update the critic
        q1, q2 = self.critic(state, action)
        with tr.no_grad():
            next_action, log_prob = self.actor(next_state, with_logprob=True)
            next_q1, next_q2 = self.target_critic(next_state, next_action)
            next_q = tr.min(next_q1, next_q2)

            assert next_q.shape == log_prob.shape, f"Next Q shape: {next_q.shape}, log_prob shape: {log_prob.shape}"
            assert next_q.shape == reward.shape, f"Next Q shape: {next_q.shape}, reward shape: {reward.shape}"

            target_q = reward + self.gamma*(1-done)*(next_q - self.entropy_coeff*log_prob)

        td_error_batch = self.loss(q1, target_q) + self.loss(q2, target_q)
        critic_loss = td_error_batch.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        tr.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_critic)
        self.critic_optimizer.step()

        # Update the actor
        proposed_action, log_prob = self.actor(state, with_logprob=True)

        q1, q2 = self.critic(state, proposed_action)
        q = tr.min(q1, q2)

        assert q.shape == log_prob.shape, f"Q shape: {q.shape}, log_prob shape: {log_prob.shape}"
        
        actor_loss = (self.entropy_coeff * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        tr.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_actor)
        self.actor_optimizer.step()

        # Update the target network
        self._update_target_network()

        # Update Entropy Coefficient by explicit gradient descent
        mean_log_prob = log_prob.detach().mean()
        entropy_update_step = self.entropy_lr*(mean_log_prob + self.target_entropy)
        self.log_alpha += entropy_update_step
        
        # Ensure entropy coefficient is positive
        self.entropy_coeff = self.log_alpha.exp()
        
        return actor_loss.item(), critic_loss.item(), self.entropy_coeff.item()*mean_log_prob.item(), self.entropy_coeff.item(), td_error_batch

