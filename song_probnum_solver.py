# Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
import functools

import numpy as np
import torch
from scipy import integrate
from torch import nn

from consts import DEVICE
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper, img_tens_shape, marginal_prob_std, \
    BS, SIGMA

from probnum import diffeq, filtsmooth, randvars, randprocs, problems
from probnum.diffeq.odefilter import ODEFilter
from probnum.diffeq.stepsize import ConstantSteps, AdaptiveSteps
from probnum.diffeq import probsolve_ivp

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((img_tens_shape[0],)) * t
    sample = x[:-img_tens_shape[0]]
    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)

def solve_scipy(min_timestep, rtol=1e-5, atol=1e-5, method='RK45'):
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)
    res = integrate.solve_ivp(ode_func, (1.0, min_timestep), init_x, rtol=rtol, atol=atol, method=method)
    return res["y"], res["t"]

def solve_magnani(min_timestep, h=1e-3,  q=2, prior='OU', print_t=False):

    steps = int(1.0 / h)

    # Define special version of ODE func that deals with JAX
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        t = np.asarray(t)
        x = np.asarray(x)
        time_steps = np.ones((img_tens_shape[0],)) * t
        sample = x[:, :-1]
        g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)

        if print_t is True:
            print(t)

        return np.concatenate([sample_grad, logp_grad], axis=0)

    # Initial x sampled from distribution at t=1.0
    t = torch.ones(BS, device=DEVICE)
    x_0 = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    x_0 = np.concatenate([x_0.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)

    # Compute derivative at x_0 when t=1.0
    f_x0 = ode_func(1.0, x_0.reshape(1, 785))

    # Initialise initial means and covariances
    m0 = np.zeros((785, q + 1, 1))
    P0 = np.zeros((785, q + 1, q + 1))
    for i in range(1, q+1):
        P0[:, i, i] = 1

    # Set means and covs as defined in Magnani et al. p7
    m0[:, 0, 0] = x_0
    m0[:, 1, 0] = f_x0
    m0 = jnp.array(m0)
    P0 = jnp.array(P0)

    # Solve the ODE!
    ms, ts = odesolver.solve_kf(m0, P0, lambda t, x : ode_func(1.0 - t, x), t0=min_timestep, t1=1.0, steps=steps, q=q, method=prior)

    return ms, ts

# Expects a list of results for all steps
def plot_trajectory(ms, ts):

    for m in ms:
        plt.plot(ts, m)

    plt.xlabel("Time / s")
    plt.ylabel("x")
    
    plt.show()

def plot_results(ms):

    fig = plt.figure() # make figure

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(ms[:-1, 0].reshape(28, 28), cmap='gray', vmin=0, vmax=1.0)

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
    
    import odesolver
    import jax.numpy as jnp

    ms, ts = solve_magnani(min_timestep=1e-2, h=1e-2, print_t=True)
    ms = np.array(ms)[:, 0, 0].reshape(784, 0)
    plot_results(ms)
    plot_trajectory(ms, ts)

    ms, ts = solve_scipy(1e-3, method='RK45')
    plot_results(ms)
    plot_trajectory(ms, ts)

# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     t0 = 1e-3
#     tmax = 1.0
#     t = torch.ones(BS, device=DEVICE)
#     init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
#     init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)

#     print("Solving basic")
#     sol = probsolve_ivp(
#         f=lambda t, x: ode_func(1.0 - t, x),
#         t0=t0,
#         tmax=tmax,
#         y0=init_x,
#         adaptive=False,
#         step=0.1,
#         dense_output=False,
#         method='EK1',
#         algo_order=1,
#     )
#     print("Solved basic")
#     print(sol)

#     # ivp = problems.InitialValueProblem(t0=t0, tmax=tmax, f=lambda t, x: ode_func(1.0 - t, x), y0=init_x)
#     # print("Initialised")

#     iwp = randprocs.markov.integrator.IntegratedWienerProcess(
#         initarg=ivp.t0,
#         num_derivatives=1,
#         wiener_process_dimension=ivp.dimension
#     )
#     # iwp = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckProcess(
#     #     initarg=ivp.t0,
#     #     driftspeed=1.,
#     #     num_derivatives=2,
#     #     wiener_process_dimension=ivp.dimension,
#     # )

#     dt = 0.5
#     # steprule = ConstantSteps(dt)
#     steprule = AdaptiveSteps(0.1, 1e5, 1e5)

#     solver = ODEFilter(
#         steprule=steprule,
#         prior_process=iwp,
#     )

#     print("Starting solve")
#     odesol = solver.solve(ivp)
#     print("Ended solve")

#     evalgrid = np.arange(ivp.t0, ivp.tmax, step=0.1)
#     sol = odesol(evalgrid)

#     plt.plot(evalgrid, sol.mean, "o-", linewidth=1)
#     plt.xlabel("Time")
#     plt.show()

# if __name__ == '__main__':

#     # Solve for n initialisations
#     results = []
#     n = 4
#     for i in range(n):
#         res = solve_rk45(min_timestep=1e-3, t_final=1.0)
#         results.append(res)

#     import matplotlib.pyplot as plt
#     from mpl_toolkits.axes_grid1 import ImageGrid

#     visualise_results(results)

#     # Plot ODE solution for first result
#     res = results[0]
#     print(res)
#     for i in range(res["y"].shape[0] - 1):
#         plt.plot(res["t"], res["y"][i], label="Scipy RK45")
#     plt.ylabel("ODE Solution")
#     plt.xlabel("t")
#     plt.show()

#     # How many pixels get clipped?
#     for res in results:
#         im = res["y"][:-1, -1].reshape(28, 28)
#         clipped = np.extract(im.flatten() > 0.0, im.flatten())
#         clipped = np.extract(clipped < 1.0, clipped)
#         print(f"Of {len(im.flatten())} pixels, {len(clipped)} were not clipped")