import numpy as np
import torch
from matplotlib import pyplot as plt

from consts import BS, DEVICE, IMG_TENS_SHAPE, SIGMA
from song_probnum_solver import solve_scipy, euler_int, solve_magnani
from song_utils import marginal_prob_std


def time_solver(init_x: torch.Tensor, gt: torch.Tensor, solver: callable, sol_params: list, torch_stack: bool = False,
                n_trials: int = 3):
    mses = []
    runtimes = []
    for params in sol_params:
        loss = 0
        runtime = 0
        for _ in range(n_trials):
            start = timer()
            ms, ts = solver(copy.deepcopy(init_x), **params)
            end = timer()

            if torch_stack:
                ms = torch.stack(ms).permute(1, 0, 2, 3)[:, :, 0, 0].detach().cpu().numpy()

            res = ms[:-1, -1]
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(res.flatten()), torch.tensor(gt.flatten()))

            loss += mse_loss.item()
            runtime += end - start

        mses.append(loss / n_trials)
        runtimes.append(runtime / n_trials)

    return mses, runtimes


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
    init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)

    # Ground truth
    print("Computing ground truth")
    ms, ts = solve_scipy(copy.deepcopy(init_x), 1e-3, rtol=1e-8, atol=1e-8, method='RK45')
    gt = ms[:-1, -1]

    # euler integration
    hs = [0.25, 0.2, 1e-1, 1e-2, 1e-3]
    print("Euler integration:")
    euler_mses, euler_times = time_solver(init_x, gt, lambda x, **kwargs: euler_int(x, t0=1e-3, **kwargs),
                                          sol_params=[{'dt': h} for h in hs])

    hs = [0.25, 0.2, 1e-1, 1e-2, 8e-3, 5e-3, 3e-3, 2e-3, 1e-3]
    print("Magnani IOU integration:")
    ou_mses, ou_times = time_solver(init_x, gt, lambda x, **kwargs: solve_magnani(x, min_timestep=1e-3, **kwargs),
                                    sol_params=[{'h': h} for h in hs], torch_stack=True)

    hs = [0.25, 0.2, 1e-1, 1e-2, 8e-3, 5e-3, 3e-3, 2e-3, 1e-3]
    print("Magnani IWP integration:")
    iwp_mses, iwp_times = time_solver(init_x, gt,
                                      lambda x, **kwargs: solve_magnani(x, min_timestep=1e-3, prior='IWP', **kwargs),
                                      sol_params=[{'h': h} for h in hs], torch_stack=True)

    tols = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    print("Scipy RK45 integration:")
    sci_mses, sci_times = time_solver(init_x, gt, lambda x, **kwargs: solve_scipy(x, 1e-3, method='RK45', **kwargs),
                                      sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])

    plt.plot(ou_times, ou_mses, label='Magnani et al. (IOUP)')
    plt.plot(euler_times, euler_mses, label='Euler')
    plt.plot(sci_times, sci_mses, label='SciPy RK45')
    plt.plot(iwp_times, iwp_mses, label='Magnani et al. (IWP)')
    plt.xlabel("Runtime")
    plt.ylabel("MSE")
    plt.legend()

    plt.semilogy()
    plt.show()
