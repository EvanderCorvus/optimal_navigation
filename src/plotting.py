import numpy as np
import matplotlib.pyplot as plt
import os
import dotenv
import torch as tr
import json
from matplotlib.patches import Rectangle

plt.rcParams['mathtext.fontset'] = 'cm'
#plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = 14
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

def plot_states(states, ax, mode, dt):
    ax.set_aspect('equal')
    #plot_flow_field(ax, 0.4, 0.75)
    T = 0
    for state in states:
        T+=len(state)*dt
        array = np.array(state)
        ax.plot(array[:, 0], array[:, 1], c='b', alpha=0.7)

    if mode == 'test': ax.plot(-0.5, 0, 'ro', label='Start')
    goal_box = Rectangle(
        (0.5 - 0.025, -0.025),
        0.05,
        0.05,
        edgecolor='black',
        facecolor='none',
        label='Goal',
        linestyle='-'
    )
    ax.add_patch(goal_box)
    ax.set(xlabel='x', ylabel='y', aspect='equal')
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)

    ax.legend()
    ax.set_aspect('equal')
    if mode == 'test': ax.set_title('T = {}s'.format(round(T/len(states), 2)))


def plot_flow_field(ax, v0, size, field_type, **kwargs):

    ax.set_aspect('equal')
    x = np.linspace(-size, size, 100)
    y = np.linspace(-size, size, 100)
    X, Y = np.meshgrid(x, y)

    if field_type=='sinusoidal':
        omega = kwargs.get('omega', 2*np.pi)
        k = kwargs.get('k', 2*np.pi)
        n_steps = kwargs.get('nsteps', 350)
        dt = kwargs.get('dt', 0.0375)
        t = np.linspace(0, n_steps*dt, n_steps)
        U = v0*np.sin(k*X + omega*t)

    if field_type == 'mexican_hat':
        # Calculate radial distance
        rho = np.sqrt(X ** 2 + Y ** 2)

        # Initialize potential array
        U = np.zeros_like(X)

        # Calculate potential for ρ ≤ 1/2
        mask = rho <= 0.5
        U[mask] = 16 * v0*(rho[mask] ** 2 - 0.25) ** 2


    im = ax.pcolormesh(X, Y, U, cmap='Reds', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$U(x,y)$', shrink=0.7)


def plot_action_Q_field(ax, agent, size):
    ax.set_aspect('equal')
    x = np.linspace(-size, size, 100)
    y = np.linspace(-size, size, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    actions = agent.act(positions)
    Q_1, Q_2, = agent.critic(
        tr.from_numpy(positions).float().to(device),
        tr.from_numpy(actions).float().to(device)
    )
    Q_values = tr.min(Q_1, Q_2)
    Q_values = Q_values.detach().cpu().numpy().reshape(X.shape)
    im = ax.pcolormesh(X, Y, Q_values.T, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$Q(x,y,\theta)$', shrink=0.7)

    x = np.linspace(-size, size, 20)
    y = np.linspace(-size, size, 20)
    X, Y = np.meshgrid(x, y)
    positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    actions = agent.act(positions)

    U = 0.2*np.array([np.cos(theta) for theta in actions])
    V = 0.2*np.array([np.sin(theta) for theta in actions])
    ax.quiver(X, Y, U.T, V.T, color='black', alpha=0.75)

    #plt.savefig(path + f'/Q_field.png')
    #plt.close()


if __name__ == '__main__':
    dotenv.load_dotenv()
    experiments_dir = os.getenv('experiment_dir')
    experiment_number = len(os.listdir(experiments_dir))
    path = f'{experiments_dir}/{experiment_number}'
    with open(f'{path}/train_states.json', 'r') as f:
        train_states = json.load(f)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(24, 8))

    plot_states(train_states, ax1, 'train', 0.0375)

    with open(f'{path}/test_states.json', 'r') as f:
        test_states = json.load(f)
    plot_states(test_states, ax2, 'test', 0.0375)
    agent = tr.load(f'{path}/agent.pth').to(device)

    plot_action_Q_field(ax3, agent, 0.75)
    fig.tight_layout()

    fig.savefig(path + f'/plots.png')
    plt.close(fig)
