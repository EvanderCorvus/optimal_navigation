import numpy as np


def mexican_hat_field(obs, v0, **kwargs):
    r = np.linalg.norm(obs)
    vorticity = 0
    if r > 0.5:
        return np.array([[0, 0]]), vorticity
    else:
        v_d = -64 * v0 * (r ** 2 - 0.25) * obs
        return v_d, vorticity

def sinusoidal_field(obs, v0, **kwargs):
    omega = kwargs.get('omega', 2*np.pi)
    k = kwargs.get('k', 2*np.pi)
    phi = kwargs.get('phi', 0)

    x, y, t = obs[0, 0], obs[0, 1], obs[0, 2]
    v_x = 0
    v_y = v0 * np.sin(k*x + omega*t + phi)
    v_D = np.array([[v_x, v_y]])
    vorticity = k*v0*np.cos(k*x + omega*t + phi)

    return v_D, vorticity
