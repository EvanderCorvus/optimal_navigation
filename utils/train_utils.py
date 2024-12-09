import numpy as np
from tqdm import tqdm
from utils.utils import time_it
from numba import njit, jit
import torch as tr
from tqdm import tqdm

def train_episode(
        env, 
        agent, 
        replay_buffer, 
        current_episode, 
        config, 
        writer=None
    ):
    observation = env.reset(mode='random')
    log_states = [env.obs.tolist()[0]]
    
    for step in range(int(config['n_steps'])):
        # Perform Actions
        if step <= config['batch_size'] and current_episode == 0:
            action = env.action_space.sample()
        else:
            action = agent.act(observation, deterministic=False)
        # Environment Steps
        next_obs, rewards, terminated, truncated, _, _ = env.step(action)

        replay_buffer.add_transition(
            tr.tensor(observation, dtype=tr.float32),#.unsqueeze(0),
            tr.tensor(action, dtype=tr.float32),#.unsqueeze(0),
            tr.tensor(rewards, dtype=tr.float32).unsqueeze(0),
            tr.tensor(next_obs, dtype=tr.float32),#.unsqueeze(0),
            tr.tensor([terminated], dtype=tr.float32).unsqueeze(0)
            )
        

        observation = next_obs
        log_states.append(env.obs.tolist()[0])
        if writer is not None:
            writer.add_scalar(
                'Reward (Group Average)', 
                rewards.mean(), 
                config['n_steps']*current_episode + step 
            )
        if len(replay_buffer) > config['batch_size']:
            # Update Agents
            #loss_actors, loss_critics, entropies, entropy_coeff = [], [], [], []
            for _ in range(config['n_optim']):
                sample, info = replay_buffer.sample(
                    config['batch_size'], 
                    return_info=True
                    )
                loss_actors, loss_critics, entropies, alpha, td_error = agent.update(sample)
                
                replay_buffer.update_priority(
                    info['index'],
                    td_error
                )
                
                if writer is not None:
                    writer.add_scalar(
                        'Loss/Actors',
                        loss_actors,
                        env.train_step
                    )
                    writer.add_scalar(
                        'Loss/Critics',
                        loss_critics,
                        env.train_step
                    )
                    writer.add_scalar(
                        'Entropy/Regularization (Group Average)',
                        entropies,
                        env.train_step
                    )
                    writer.add_scalar(
                        'Entropy/Coefficient (Group Average)',
                        alpha,
                        env.train_step
                    )
                env.train_step += 1
        if terminated or truncated: break
    if len(replay_buffer) <= config['batch_size']:
        return log_states, np.mean(rewards), 0, 0
    return log_states, np.mean(rewards), np.mean(loss_actors), np.mean(loss_critics)


def test_episode(
        env, 
        agents, 
        config,
        writer=None
    ):
    observations = env.reset()
    list_obs = []
    list_actions = []
    for step in range(int(config['n_steps'])):
        action = agents.act(observations)  #np.array([agents[f'agent_{i}'].act(observations[i]) for i in range(env.n_agents)])
        #print(prof.key_averages().table(sort_by="cuda_time_total")) #, row_limit=10 
        next_obs, test_rewards, terminated, truncated, _, _ = env.step(action)

        list_obs.append(observations.tolist()[0])
        list_actions.append(action.tolist())

        observations = next_obs
        if writer is not None:

            writer.add_scalar(
                'Test/Reward (Group Average)',
                test_rewards.mean(),
                step,
            )
        if terminated or truncated: 
            break

    return list_obs, list_actions
