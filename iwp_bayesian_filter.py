import numpy as np


# TODO - make this work for generic dts and not just a set one
def ode_filter(func: callable, init_mean: np.ndarray, init_cov: np.ndarray, dt: float, n_steps: int):
    dim = init_mean.shape
    assert (dim, dim) == init_cov.shape, 'Error - init mean and covariance have a shape mismatch!'
    A = calc_A_mat(dt, dim)
    Q = calc_Q_mat(dt, dim)
    H = calc_H_mat(dim)  # don't really need a function for this one but ¯\_(ツ)_/¯

    t = 0

    means = np.zeros((n_steps, dim), dtype=init_mean.dtype)
    covs = np.zeros((n_steps, dim, dim), dtype=init_mean.dtype)
    means[0] = init_mean
    covs[0] = init_cov

    prev_mean, prev_cov = init_mean, init_cov
    for step in range(1, n_steps):
        # might want to consider changing integration schemes, eg. to leapfrog. Not sure how rigorous that would be
        t += dt

        approx_mean = A @ prev_mean
        approx_cov = A @ prev_cov @ A.T + Q

        obs = func(t, approx_mean)  # TODO - is this literally just the derivative, AKA x'=f(t, x)?
        mean_error = obs - H @ prev_mean

        S_mat = H @ approx_cov @ H.T  # TODO - understand meaning
        # TODO is this numerically stable? Might be able to improve stability by changing the inv to a left-solve
        rel_unc = approx_cov @ H.T @ np.linalg.inv(S_mat)

        mean = approx_mean + rel_unc @ mean_error
        cov = approx_cov - rel_unc @ S_mat @ rel_unc.T

        prev_mean, prev_cov = mean, cov
        means[step], covs[step] = mean, cov

    return means, covs
