import numpy as np
import math

# TODO: Is this correct?
# If everything is 1-dimensional, Magnani et al. use of H in algorithm is correct
# We don't know if we set 1st or 2nd element should be 1 for extracting measurement.
# Most likely second thing - as this is what we directly observe - the "measurement"
def calc_H_mat(q: int):
    H = np.zeros((1, q + 1)) # Assume paper is wrong about (1, q) - must be (1, q + 1) to work!
    H[0, 1] = 1
    return H

def calc_A_mat(dt: float, q: int):
    A = np.zeros((q + 1, q + 1), dtype=np.float32)
    for i in range(q + 1):
        for j in range(q + 1):
            if j >= i:
                A[i, j] = (dt ** (j - i)) / math.factorial(j - i)
    return A

# Sigma associated with IWP prior
def calc_Q_mat(dt: float, q: int, sigma = 1.0):
    Q = np.zeros((q + 1, q + 1), dtype=np.float32)
    for i in range(q + 1):
        for j in range(q + 1):
            num = dt ** (2 * q + 1 - i - j)
            denom = (2 * q + 1 - i - j) * math.factorial(q - i) * math.factorial(q - j)
            Q[i, j] = (sigma ** 2) * (num / denom)
    return Q

# TODO - make this work for generic dts and not just a set one
def ode_filter(func: callable, init_mean: np.ndarray, init_cov: np.ndarray, dt: float, n_steps: int):
    dim = init_mean.shape[0] # dim = q + 1
    # print(dim)

    # if dim == ():
    #     dim = 1
    # else:
    #     assert (dim, dim) == init_cov.shape, 'Error - init mean and covariance have a shape mismatch!'

    A = calc_A_mat(dt, q=dim - 1) # (q + 1, q + 1)
    Q = calc_Q_mat(dt, q=dim - 1) # (q + 1, q + 1)
    H = calc_H_mat(q=dim - 1)  # (1, q)

    t = 0

    means = np.zeros((n_steps, dim, 1), dtype=init_mean.dtype)
    covs = np.zeros((n_steps, dim, dim), dtype=init_mean.dtype)
    means[0] = init_mean # (q + 1, 1)
    covs[0] = init_cov # (q + 1, q + 1)

    prev_mean, prev_cov = init_mean, init_cov
    for step in range(1, n_steps):
        # might want to consider changing integration schemes, eg. to leapfrog. Not sure how rigorous that would be
        t += dt

        print(A.shape, prev_mean.shape)
        approx_mean = A @ prev_mean # (q + 1, 1) = (q + 1, q + 1) x (q + 1, 1)
        approx_cov = A @ prev_cov @ A.T + Q # (q + 1, q + 1) = (q + 1, q + 1) x (q + 1, q + 1) x (q + 1, q + 1)

        obs = func(t, approx_mean)  # (q + 1, 1) TODO - is this literally just the derivative, AKA x'=f(t, x)?
        mean_error = obs - H @ prev_mean # (q + 1, 1) = (q + 1, 1) - (1, 1) (FIXME: ???)
        # What's the issue? Error should be different for EACH! so instead of (1, q + 1) x (q + 1, 1) = (1, 1)
        # we want a result that's (q + 1, 1) = (x, y) x (q + 1, 1) hence H should be (q + 1, q + 1)

        # FIXME: 1x1 for 1-dim case?
        S_mat = H @ approx_cov @ H.T  # (1, 1) = (1, q + 1) x (q + 1, q + 1) x (q + 1, 1) TODO - understand meaning
        # TODO is this numerically stable? Might be able to improve stability by changing the inv to a left-solve
        rel_unc = approx_cov @ H.T @ np.linalg.inv(S_mat) # (q + 1, 1) = (q + 1, q + 1) x (q + 1, 1) x (1, 1)

        print(approx_mean.shape, rel_unc.shape, mean_error.shape)
        mean = approx_mean + rel_unc @ mean_error # (q + 1, 1) + (q + 1, 1) x (q + 1, 1) (FIXME: can't multiply this!)
        cov = approx_cov - rel_unc @ S_mat @ rel_unc.T

        prev_mean, prev_cov = mean, cov
        means[step], covs[step] = mean, cov

    return means, covs

# How to choose a q? i.e. size of init_mean and init_cov.
# Should be at least 2 - even for 1-dimensional case as we need original + derivative
if __name__ == "__main__":

    def wiener_1d(t: float, mean: float):
        return np.random.normal(loc=mean, scale=1.0)
    
    def simple_ode(t: float, mean: float):
        return mean + t

    # Time step size should be 0.5 for this.
    def negative_exponential_ode(t: float, mean: float):
        return -mean
    
    means, covs = ode_filter(
        func=negative_exponential_ode,
        init_mean=np.zeros((2, 1), dtype=np.float32), # Should be defined to be the order we want to work with. Check p7 notes on intialisation.
        init_cov=np.eye(2, dtype=np.float32),
        dt=0.5,
        n_steps=24)

    print(means)
    print(covs)

    import scipy

    # init = np.array(0, dtype=np.float32).reshape(1)
    init = np.zeros((2, 1), dtype=np.float32)
    res = scipy.integrate.solve_ivp(negative_exponential_ode, (0.0, 12.0), init, rtol=1e-5, atol=1e-5, method='RK45')
    print(res)

# Use L1 or L2 over function space to measure closeness
# Also plot and examine functions

# Also, assuming they converge,
# we can plot error w.r.t. GT vs # iterations and plot RK4 vs our method
# and hope our error drops faster.

# Questions for Francisco:
# - How to choose q?
# - How to do multi-dimensional?
# - How to choose prior?