import numpy as np


# TODO - make this work for generic dts and not just a set one
def ode_filter(func: callable, init_mean: np.ndarray, init_cov: np.ndarray, dt: float, n_steps: int):
    dim = init_mean.shape
    assert (dim, dim) == init_cov.shape, 'Error - init mean and covariance have a shape mismatch!'
    A = calc_A_mat(dt, dim)
    Q = calc_Q_mat(dt, dim)

    t = 0
    means = np.zeros((dim, n_steps), dtype=init_mean.dtype)
    covs = np.zeros((dim, n_steps, n_steps), dtype=init_mean.dtype)
    for _ in range(1, n_steps):
        raise NotImplementedError

        t += dt

    return means, covs
