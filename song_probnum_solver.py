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

def ode_func(t, x):
    """The ODE function for the black-box solver."""
    # Jax bullshit
    t = np.asarray(t)
    x = np.asarray(x)

    time_steps = np.ones((img_tens_shape[0],)) * t
    # sample = x[:-img_tens_shape[0]]
    sample = x[:, :-1]
    # breakpoint()
    print("EE", torch.tensor(t))
    g = diffusion_coeff(torch.tensor(t), SIGMA).cpu().numpy()
    sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
    print(t)
    return np.concatenate([sample_grad, logp_grad], axis=0)

def solve_rk45(min_timestep, t_final):
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)
    res = integrate.solve_ivp(ode_func, (t_final, min_timestep), init_x, rtol=1e-5, atol=1e-5, method='RK45')
    return res

def visualise_results(results):

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, res in zip(grid, results):
        im = res["y"][:-1, -1].reshape(28, 28)
        ax.imshow(im.clip(0.0, 1.0))

    plt.show()

if __name__ == "__main__":
    
    import odesolver
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    t0 = 0.01
    t1 = 1.0
    h = 1e-3
    steps = int(t1 / h)

    q = 2

    m0 = np.zeros((785, q + 1, 1))
    P0 = np.zeros((785, q + 1, q + 1))
    for i in range(1, q+1):
        P0[:, i, i] = 1

    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)

    # res = integrate.solve_ivp(ode_func, (t1, h), init_x, rtol=1e-5, atol=1e-5, method='RK45')

    # t = h
    # x = init_x
    # for _ in range(steps):
    #     # breakpoint()
    #     print(abs(x[:-1]).mean())
    #     x -= ode_func(max(1-t,h), x) * h
    #     t += h
    # # breakpoint()
    # im = res["y"][:-1, -1].reshape(28, 28)
    # im = x[:-1].reshape(28, 28)
    # plt.imshow(im)
    # plt.show()

    # Fix initialisation of m0 s.t. (x_0, f(0, x_0), 0,0,...) or ??? (x_0, f(1.0, x_0), 0,0,...)
    # in the rows
    f_x = ode_func(1.0, init_x.reshape(1, 785))

    m0[:, 0, 0] = init_x
    m0[:, 1, 0] = f_x
    m0 = jnp.array(m0)
    P0 = jnp.array(P0)
    # print(m0[:, 0, 0])
    # print(m0[:, 1, 0])
    # exit(0)

    print(m0.shape, P0.shape)
    ms, ts = odesolver.solve_kf(m0, P0, lambda t, x : ode_func(1.0 - t, x), t0=t0, t1=t1, steps=steps, q=q)

    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)

    results = ms[-1]

    import matplotlib.pyplot as plt
    
    # for i in range(784):
    #     plt.plot(np.asarray(ms)[:,i,0,0])
    plt.imshow(results[:-1, 0, 0].reshape(28, 28))
    plt.show()

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