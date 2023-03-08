import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import integrate

import odesolver
from consts import DEVICE, BS, SIGMA, IMG_TENS_SHAPE
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper, marginal_prob_std


def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((IMG_TENS_SHAPE[0],)) * t
    sample = x[:-IMG_TENS_SHAPE[0]]
    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)


def solve_scipy(min_timestep, rtol=1e-5, atol=1e-5, method='RK45'):
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)
    res = integrate.solve_ivp(ode_func, (1.0, min_timestep), init_x, rtol=rtol, atol=atol, method=method)
    return res["y"], res["t"]


def solve_magnani(min_timestep, h=1e-3, q=2, prior='OU', print_t=False):
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
    t = torch.ones(BS, device=DEVICE)
    x_0 = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    x_0 = np.concatenate([x_0.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)

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


# Expects a list of results for all steps
# ms -> [785, steps]
def plot_trajectory(ms, ts, solver_name: str = ''):
    # m -> [steps]
    for m in ms:
        plt.plot(ts, m)

    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.title("Pixel's values over time for " + solver_name)

    plt.show()


# TODO - plot also time
def plot_results(ms):
    fig = plt.figure()  # make figure

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(ms[:-1, 0].reshape(28, 28), cmap=plt.get_cmap('jet'), vmin=0, vmax=1.0)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(ms[:-1, j].reshape(28, 28))
        # return the artists set
        return [im]

    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(ms[0])),
                                  interval=75, blit=True, repeat_delay=2000)
    plt.show()


if __name__ == "__main__":
    # Plot the result and trajectory
    # and plot the animation over time

    # FID / some accuracy vs step size and/or vs runtime
    # Plot this graph for various solvers
    # We want to know where it starts diverging to identify phase transition

    ms, ts = solve_magnani(min_timestep=1e-2, h=1e-2, print_t=True)
    ms = torch.stack(ms).permute(1, 0, 2, 3)[:, :, 0, 0].detach().cpu().numpy()
    plot_results(ms)
    plot_trajectory(ms[:-1], ts, solver_name='IOU')

    ms, ts = solve_scipy(1e-3, method='RK45')
    plot_results(ms)
    plot_trajectory(ms[:-1], ts, solver_name='Scipy RK45')
