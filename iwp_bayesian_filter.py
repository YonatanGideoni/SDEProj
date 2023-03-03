import numpy as np
import math

# TODO: Is this correct?
def calc_H_mat(dim: int):
    H = np.zeros((1, dim))
    H[0] = 1
    return H

def calc_A_mat(dt: float, dim: int):
    A = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        for j in range(dim):
            if j >= i:
                A[i, j] = (dt ** (j - i)) / math.factorial(j - i)
    return A

# Sigma associated with IWP prior
def calc_Q_mat(dt: float, dim: int, sigma = 1.0):
    Q = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        for j in range(dim):
            num = dt ** (2 * dim + 1 - i - j)
            denom = (2 * dim + 1 - i - j) * math.factorial(dim - i) * math.factorial(dim - j)
            Q[i, j] = (sigma ** 2) * (num / denom)
    return Q

# TODO - make this work for generic dts and not just a set one
def ode_filter(func: callable, init_mean: np.ndarray, init_cov: np.ndarray, dt: float, n_steps: int):
    dim = init_mean.shape[0]
    # print(dim)

    # if dim == ():
    #     dim = 1
    # else:
    #     assert (dim, dim) == init_cov.shape, 'Error - init mean and covariance have a shape mismatch!'

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

if __name__ == "__main__":

    def wiener_1d(t: float, mean: float):
        return np.random.normal(loc=mean, scale=1.0)
    
    def simple_ode(t: float, mean: float):
        return mean + t
    
    means, covs = ode_filter(
        func=wiener_1d,
        init_mean=np.array(0, dtype=np.float32).reshape(1),
        init_cov=np.array(1, dtype=np.float32).reshape(1, 1),
        dt=0.01,
        n_steps=100)

    print(means)
    print(covs)

    import scipy

    init = np.array(0, dtype=np.float32).reshape(1)
    res = scipy.integrate.solve_ivp(wiener_1d, (0.0, 1.0), init, rtol=1e-5, atol=1e-5, method='RK45')
    print(res)