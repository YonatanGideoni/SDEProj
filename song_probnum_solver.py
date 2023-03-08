import numpy as np
import torch
from scipy import integrate

from consts import DEVICE
from song_utils import diffusion_coeff, score_eval_wrapper, divergence_eval_wrapper, img_tens_shape, marginal_prob_std, \
    BS, SIGMA

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import odesolver
import jax.numpy as jnp

def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((img_tens_shape[0],)) * t
    sample = x[:-img_tens_shape[0]]
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
        time_steps = np.ones((img_tens_shape[0],)) * t
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
    ms, ts = odesolver.solve_kf(m0, P0, lambda t, x: ode_func(1.0 - t, x), t0=min_timestep, t1=1.0, steps=steps, q=q,
                                method=prior)

    return ms, ts


# Expects a list of results for all steps
# ms -> [785, steps]
def plot_trajectory(ms, ts):
    # m -> [steps]
    for m in ms:
        plt.plot(ts, m)

    plt.xlabel("Time / s")
    plt.ylabel("x")

    plt.show()


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

def euler_int(x, t0: float, dt: float):
    t = t0
    res=[x]
    ts=[t]
    while t < t0:
        t+=dt
        x-=ode_func(1-t,x)*dt

        res.append(x)
        ts.append(t)
    
    return x, ts

if __name__ == "__main__":
    # Plot the result and trajectory
    # and plot the animation over time

    # FID / some accuracy vs step size and/or vs runtime
    # Plot this graph for various solvers
    # We want to know where it starts diverging to identify phase transition

    from timeit import default_timer as timer
    import copy

    # Define some starting point
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((img_tens_shape[0],))], axis=0)

    # Ground truth
    print("Computing ground truth")
    ms, ts = solve_scipy(init_x, 1e-3, rtol=1e-8, atol=1e-8, method='RK45')
    gt = ms[:-1, -1]

    # euler integration
    hs = [0.25, 0.2, 1e-1, 1e-2, 1e-3] #[ 1e-4, 1e-5]
    euler_mses = []
    euler_times = []
    print("Euler integration:")
    for h in hs:
        print(f"{h}")

        loss = 0
        time = 0
        for i in range(3):
            start = timer()
            ms, ts = euler_int(init_x, t0=1e-3, dt=h)
            end = timer()

            res = ms[:-1]
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(res.flatten()), torch.tensor(gt.flatten()))
            # print(mse_loss)
            loss += mse_loss.item()
            time += end - start

        euler_mses.append(loss / 3.0)
        euler_times.append(time / 3.0)

    # Magnani integration
    hs = [0.25, 0.2, 1e-1, 1e-2, 1e-3] #[ 1e-4, 1e-5]
    mses = []
    times = []
    print("Magnani integration:")
    for h in hs:
        print(f"{h}")

        loss = 0
        time = 0
        for i in range(3):
            start = timer()
            ms, ts = solve_magnani(init_x, min_timestep=1e-3, h=h)
            end = timer()
            ms = torch.stack(ms).permute(1, 0, 2, 3)[:, :, 0, 0].detach().cpu().numpy()

            res = ms[:-1, -1]
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(res.flatten()), torch.tensor(gt.flatten()))
            loss += mse_loss.item()
            time += end - start

        mses.append(loss / 3.0)
        times.append(time / 3.0)

    tols = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    sci_mses = []
    sci_times = []
    for tol in tols:
        
        loss = 0
        time = 0
        for i in range(3):
            start = timer()
            ms, ts = solve_scipy(init_x, 1e-3, atol=tol, rtol=tol, method='RK45')
            end = timer()
            res = ms[:-1, -1]
            # mse_loss = np.mean((res - gt) ** 2)
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(res.flatten()), torch.tensor(gt.flatten()))
            # print(mse_loss)
            # breakpoint()
            loss += mse_loss.item()
            time += end - start

        sci_mses.append(loss / 3.0)
        sci_times.append(time / 3.0)

    plt.plot(times, mses, label='Magnani et al.')
    plt.plot(euler_times, euler_mses, label='Euler')
    plt.plot(sci_times, sci_mses, label='SciPy RK45')
    plt.xlabel("Runtime")
    plt.ylabel("MSE")

    plt.semilogy()
    plt.show()

    # plot_results(ms)
    # plot_trajectory(ms[:-1], ts)