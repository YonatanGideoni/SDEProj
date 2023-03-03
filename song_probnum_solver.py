# Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
import functools

import numpy as np
import torch
from scipy import integrate
from torch import nn

from consts import DEVICE
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper, img_tens_shape, marginal_prob_std, \
    BS, SIGMA


def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((img_tens_shape[0],)) * t
    sample = x[:-img_tens_shape[0]]
    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)


if __name__ == '__main__':
    min_timestep = 1e-3
    t_final = 1.

    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]

    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)
    # Black-box ODE solver TODO - look at its outputs and error over time, how good/efficient is this? Compare to euler?
    res = integrate.solve_ivp(ode_func, (t_final, min_timestep), init_x, rtol=1e-5, atol=1e-5, method='RK45')
