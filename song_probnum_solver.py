import numpy as np
import torch
from scipy import integrate

import odesolver
from consts import SIGMA, IMG_TENS_SHAPE, float_dtype
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper


def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
    sample = x[:-IMG_TENS_SHAPE[0]]
    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)


def solve_scipy(init_x, min_timestep, rtol=1e-5, atol=1e-5, method='RK45'):
    res = integrate.solve_ivp(lambda t, x: reverse_time(ode_func, t, x), (min_timestep, 1.0), init_x, rtol=rtol,
                              atol=atol, method=method)
    return res["y"], res["t"]


def solve_magnani(init_x, min_timestep, steps, q=2, solver_params: dict = {}, prior='OU', print_t=False):
    # Define special version of ODE func that deals with JAX
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        t = np.asarray(t)
        x = np.asarray(x)
        time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
        sample = x[:, :-1]
        g = diffusion_coeff(torch.tensor(t, dtype=float_dtype), SIGMA).cpu().numpy()
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)

        if print_t is True:
            print(t)

        return np.concatenate([sample_grad, logp_grad], axis=0)

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
    for t in ts:
        approx_grad = reverse_time(ode_func, t, x)
        approx_x = x + approx_grad * dt

        x += (approx_grad + reverse_time(ode_func, t + dt, approx_x)) * dt / 2

        res.append(x)

    return torch.tensor(res).permute(1, 0), ts


def euler_int(x, ts: np.array):
    res = [x]

    dt = abs(ts[0] - ts[1])

    for t in ts:
        x += reverse_time(ode_func, t, x) * dt
        res.append(x)

    return torch.tensor(res).permute(1, 0), ts


def reverse_time(func: callable, t, x, T=1.0):
    return -func(max(T - t, 1e-7), x)
