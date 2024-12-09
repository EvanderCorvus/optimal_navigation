import numpy as np


def AB_step(
        obs,
        action,
        dt,
        noise_std,
        force_func,
        boundary_condition_func
    ):
    forces, vorticity = force_func(obs)

    rnd = np.random.randn(1, 1)
    thetas = action + vorticity*dt + noise_std*rnd
    e = np.empty((1, 2))
    e[:, 0] = np.cos(thetas)
    e[:, 1] = np.sin(thetas)

    v = e + forces
    next_pos = obs[:, :-1] + v*dt
    next_pos = boundary_condition_func(next_pos)

    return next_pos
