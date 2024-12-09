from utils import train_episode, test_episode, hyperparams_dict, SingleAgentPrioritizedReplayBuffer
from src import StaticEnvironment, TimeDependentEnv, SoftActorCritic
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import os
import dotenv
import shutil
import torch as tr
import numpy as np
import json

def main():
    dotenv.load_dotenv()
    hparam_path = os.getenv('hparam_path')
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    config = hyperparams_dict("hparams", hparam_path)
    config['obs_dim'] = 3
    experiments_dir = os.getenv('experiment_dir')
    experiment_number = len(os.listdir(experiments_dir))+1

    env = TimeDependentEnv(config)
    agent = SoftActorCritic(config, device).to(device)
    path = f'{experiments_dir}/{experiment_number}'
    #os.makedirs(path)
    replay_buffer = SingleAgentPrioritizedReplayBuffer(
        size=config['buffer_size'],
        alpha=config['alpha'],
        beta=config['beta']
    )
    writer = SummaryWriter(path)

    
    train_states = []
    for episode in tqdm(range(int(config['n_episodes']))):
        log_states, rewards, loss_actors, loss_critics = train_episode(
            env,
            agent,
            replay_buffer,
            episode,
            config,
            writer
        )

        train_states.append(log_states)

    tr.save(agent, f'{path}/agent.pth')
    shutil.copy(hparam_path, f'{experiments_dir}/{experiment_number}/hparams.ini')
    #np.save(f'{path}/train_states.npy', train_states)

    with open(f'{path}/train_states.json', 'w') as f:
        json.dump(train_states, f)

    test_states = []
    for episode in tqdm(range(int(config['n_test_episodes']))):
        test_log_states, _ = test_episode(
            env,
            agent,
            config,
        )
        test_states.append(test_log_states)


    #plot_states(test_states, path, 'test', config['dt'])
    #np.save(f'{path}/test_states.npy', test_states)
    with open(f'{path}/test_states.json', 'w') as f:
        json.dump(test_states, f)

if __name__ == '__main__':
    main()
