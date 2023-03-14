import numpy as np
import torch
from scipy import integrate

import odesolver
from consts import SIGMA, IMG_TENS_SHAPE, SMALL_NUMBER, float_dtype
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper, marginal_prob_std, \
    marginal_prob_std_der


def ode_func(t, y):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
    divergent_term = float(marginal_prob_std(torch.tensor(t), SIGMA).cpu().numpy())
    divergent_der = float(marginal_prob_std_der(torch.tensor(t), SIGMA).cpu().numpy())
    x = divergent_term * y

    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample = x[:-IMG_TENS_SHAPE[0]]
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)

    trimmed_y = y[:-1].numpy() if isinstance(y, torch.Tensor) else y[:-1]
    rescaled_grad = (sample_grad - divergent_der * trimmed_y) / divergent_term

    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([rescaled_grad, logp_grad], axis=0)


def solve_scipy(init_x, min_timestep, rtol=1e-5, atol=1e-5, method='RK45'):
    res = integrate.solve_ivp(lambda t, x: reverse_time(ode_func, t, x), (min_timestep, 1.0), init_x, rtol=rtol,
                              atol=atol, method=method)
    return res["y"], res["t"]


def solve_magnani(init_x, min_timestep, steps, q=2, solver_params: dict = {}, prior='OU', print_t=False):
    # Define special version of ODE func that deals with JAX
    def ode_func(t, y):
        """The ODE function for the black-box solver."""
        t = np.asarray(t)
        y = np.asarray(y)

        time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
        divergent_term = float(marginal_prob_std(torch.tensor(t), SIGMA).cpu().numpy())
        divergent_der = float(marginal_prob_std_der(torch.tensor(t), SIGMA).cpu().numpy())
        x = divergent_term * y

        g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
        sample = x[:, :-IMG_TENS_SHAPE[0]]
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)

        trimmed_y = y[0, :-1].numpy() if isinstance(y, torch.Tensor) else y[0, :-1]
        rescaled_grad = (sample_grad - divergent_der * trimmed_y) / divergent_term

        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([rescaled_grad, logp_grad], axis=0)

    # Initial x sampled from distribution at t=1.0
    x_0 = init_x

    # Compute derivative at x_0 when t=1.0
    f_x0 = reverse_time(ode_func, 0.0, x_0.reshape(1, 785))

    # Initialise initial means and covariances
    m0 = np.zeros((785, q + 1, 1))
    P0 = np.zeros((785, q + 1, q + 1))
    for i in range(1, q + 1):
        P0[:, i, i] = 1

    # Set means and covs as defined in Magnani et al. p7
    m0[:, 0, 0] = x_0
    m0[:, 1, 0] = f_x0
    m0 = torch.Tensor(m0)
    P0 = torch.Tensor(P0)

    # Solve the ODE!
    ms, ts = odesolver.solve_kf(m0, P0, lambda t, x: reverse_time(ode_func, t, x), t0=min_timestep, t1=1.0, steps=steps,
                                q=q, method=prior, **solver_params)

    return ms, ts


def second_order_heun_int(x, ts: np.array):
    res = [x]
    dt = abs(ts[0] - ts[1])
    for t in ts[:-1]:
        approx_grad = reverse_time(ode_func, t, x)
        approx_x = x + approx_grad * dt

        x += (approx_grad + reverse_time(ode_func, t + dt, approx_x)) * dt / 2

        res.append(x)

    return torch.tensor(np.array([np.array(r) for r in res])).permute(1, 0), ts


def euler_int(x, ts: np.array):
    res = [x]

    dt = abs(ts[0] - ts[1])

    for t in ts[:-1]:
        x += reverse_time(ode_func, t, x) * dt
        res.append(x)

    return torch.tensor(np.array([np.array(r) for r in res])).permute(1, 0), ts


def reverse_time(func: callable, t, x, T=1.0):
    return -func(max(T - t, SMALL_NUMBER), x)
