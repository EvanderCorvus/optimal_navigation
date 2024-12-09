import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit, jit
from .flow_fields import *
from utils import AB_step

class StaticEnvironment(gym.Env):
    def __init__(self, config):
        super(StaticEnvironment, self).__init__()
        self.action_space = spaces.Box(
            low=-np.pi, 
            high=np.pi, 
            shape=(1, 1)
        )
        self.observation_space = spaces.Box(
            low=np.array([[-config['size'], -config['size']]]),
            high=np.array([[config['size'], config['size']]]),
        )
        self.goal_space = spaces.Box(
            low=np.array([[0.5-0.025, -0.025]]),
            high=np.array([[0.5+0.025, 0.025]]),
            dtype=float
            )

        for key, value in config.items():
            setattr(self, key, value)

        self.obs = None
        self.step_count = 0
        self.train_step = 0
        self.noise_std = np.sqrt(config['dt'])*config['characteristic_size']

    def reset(self, **kwargs):
        mode = kwargs.get('mode', 'deterministic')
        self.step_count = 0
        if mode == 'random':
            self.obs = self.observation_space.sample()
        else:
            self.obs = np.array([[-0.5, 0.]])

        return self.obs
    
    def step(self, action):
        next_obs = AB_step(
            self.obs,
            action,
            self.dt,
            self.noise_std,
            self.flow_field,
            self.boundary_condition_func
        )
        
        reward = self._get_reward(next_obs)
        self.obs = next_obs
        terminated = self.goal_space.contains(next_obs)
        truncated = self.step_count >= self.n_steps
        self.step_count += 1

        return next_obs, reward, terminated, truncated, {}, {}

    
    def _get_reward(self, obs):
        if self.goal_space.contains(obs):
            return np.array([1])
        else:
            return np.array([-self.dt])

    def flow_field(self, obs, **kwargs):
        if self.field_type == 'mexican_hat':
            return mexican_hat_field(obs, self.v0, **kwargs)

        
    def boundary_condition_func(self, pos):
        return np.clip(pos, -self.size, self.size)


class TimeDependentEnv(gym.Env):
    def __init__(self, config):
        super(TimeDependentEnv, self).__init__()
        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1, 1)
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [[-config['size'],
                  -config['size'],
                  0]]),
            high=np.array(
                [[config['size'],
                  config['size'],
                  config['dt'] * config['n_steps']]]),
        )
        self.goal_space = spaces.Box(
            low=np.array([[0.5 - 0.025, -0.025]]),
            high=np.array([[0.5 + 0.025, 0.025]]),
            dtype=float
        )

        for key, value in config.items():
            setattr(self, key, value)

        self.obs = None
        self.step_count = 0
        self.train_step = 0
        self.noise_std = np.sqrt(config['dt']) * config['characteristic_size']

    def reset(self, **kwargs):
        mode = kwargs.get('mode', 'deterministic')
        self.step_count = 0
        if mode == 'random':
            self.obs = self.observation_space.sample()
            self.obs[0][-1] = 0
        else:
            self.obs = np.array([[-0.5, 0., 0.]])

        return self.obs

    def step(self, action):
        next_pos = AB_step(
            self.obs,
            action,
            self.dt,
            self.noise_std,
            self.flow_field,
            self.boundary_condition_func
        )

        reward = self._get_reward(next_pos)
        next_obs = np.concatenate([next_pos, np.array([[self.step_count*self.dt]])], axis=1)
        self.obs = next_obs
        terminated = self.goal_space.contains(next_obs)
        truncated = self.step_count >= self.n_steps
        self.step_count += 1

        return next_obs, reward, terminated, truncated, {}, {}

    def _get_reward(self, pos):
        if self.goal_space.contains(pos):
            return np.array([1])
        else:
            return np.array([-self.dt])

    def flow_field(self, obs, **kwargs):
        if self.field_type == 'sinusoidal':
            return sinusoidal_field(obs, self.v0, **kwargs)
        else:
            raise Exception('Invalid field type')

    def boundary_condition_func(self, pos):
        return np.clip(pos, -self.size, self.size)
'''
        if config['time_dependent']:
            
'''