import numpy as np
import torch
from scipy import integrate

import odesolver
from consts import SIGMA, IMG_TENS_SHAPE
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
    res = integrate.solve_ivp(ode_func, (1.0, min_timestep), init_x, rtol=rtol, atol=atol, method=method)
    return res["y"], res["t"]


def solve_magnani(init_x, min_timestep, h=1e-3, q=2, prior='OU', print_t=False):
    steps = int(1.0 / h)

    # Define special version of ODE func that deals with JAX
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        t = np.asarray(t)
        x = np.asarray(x)
        time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
        sample = x[:, :-1]
        g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)

        if print_t is True:
            print(t)

        return np.concatenate([sample_grad, logp_grad], axis=0)

    # Initial x sampled from distribution at t=1.0
    x_0 = init_x

    # Compute derivative at x_0 when t=1.0
    f_x0 = ode_func(1.0, x_0.reshape(1, 785))

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
    ms, ts = odesolver.solve_kf(m0, P0, ode_func, t0=1.0, t1=min_timestep, steps=steps, q=q, method=prior)

    return ms, ts


def second_order_heun_int(x, t0: float, dt: float, tmax: float = 1.0):
    t = tmax
    res = [x]
    ts = [t]
    while t > t0:
        approx_grad = ode_func(t, x)
        approx_x = x - approx_grad * dt

        # TODO: t - dt?
        x -= (approx_grad + ode_func(max(t0, t - dt), approx_x)) * dt / 2

        res.append(x)
        ts.append(t)

        t -= dt

    return torch.tensor(res).permute(1, 0), ts


def euler_int(x, t0: float, dt: float, tmax: float = 1.0):
    t = tmax
    res = [x]
    ts = [t]
    while t > t0:
        x -= ode_func(t, x) * dt

        res.append(x)
        ts.append(t)

        t -= dt

    return torch.tensor(res).permute(1, 0), ts
