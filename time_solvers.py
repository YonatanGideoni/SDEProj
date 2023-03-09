import copy
from timeit import default_timer as timer

import numpy as np
import torch
from matplotlib import pyplot as plt

from consts import BS, DEVICE, IMG_TENS_SHAPE, SIGMA
from plot_utils import plot_final_results
from song_probnum_solver import solve_scipy, euler_int, solve_magnani, second_order_heun_int
from song_utils import marginal_prob_std


def time_solver(init_x: torch.Tensor, gt: torch.Tensor, solver: callable, sol_params: list, torch_stack: bool = False,
                n_trials: int = 3):
    mses = []
    results_over_time = []
    runtimes = []
    for params in sol_params:
        loss = 0
        runtime = 0
        for _ in range(n_trials):

            init_x_copy = copy.deepcopy(init_x)
            start = timer()
            res_over_time, ts = solver(init_x_copy, **params)
            end = timer()

            if torch_stack:
                res_over_time = torch.stack(res_over_time).permute(1, 0, 2, 3)[:, :, 0, 0].detach().cpu().numpy()

            res = res_over_time[:-1, -1]
            results_over_time.append(res_over_time[:-1])
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(res.flatten()), torch.tensor(gt.flatten()))

            loss += mse_loss.item()
            runtime += end - start

        mses.append(loss / n_trials)
        runtimes.append(runtime / n_trials)

    return mses, runtimes, results_over_time


if __name__ == "__main__":
    # Plot the result and trajectory
    # and plot the animation over time

    # FID / some accuracy vs step size and/or vs runtime
    # Plot this graph for various solvers
    # We want to know where it starts diverging to identify phase transition

    # TODO: Use consistent torch seed
    # torch.manual_seed(42)

    # Define some starting point
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)

    min_timestep = 1e-7

    # Ground truth
    print("Computing ground truth")
    ms, ts = solve_scipy(copy.deepcopy(init_x), 1e-3, rtol=1e-8, atol=1e-8, method='RK45')
    gt = ms[:-1, -1]

    # euler integration
    hs = [0.25, 0.2, 1e-1, 1e-2, 0.005]
    print("Euler integration:")
    euler_mses, euler_times, euler_diffusions = time_solver(init_x, gt,
                                                            lambda x, **kwargs: euler_int(x, t0=1e-7, **kwargs),
                                                            sol_params=[{'dt': h} for h in hs])

    hs = [0.25, 0.2, 1e-1, 1e-2, 0.005]
    print("2nd order Heun integration:")
    heun_mses, heun_times, heun_diffusions = time_solver(init_x, gt,
                                                         lambda x, **kwargs: second_order_heun_int(x, t0=1e-7,
                                                                                                   **kwargs),
                                                         sol_params=[{'dt': h} for h in hs])

    hs = [0.25, 0.2, 1e-1, 1e-2, 8e-3, 5e-3, 3e-3, 2e-3]
    print("Magnani IOU integration, q=2:")
    ou2_mses, ou2_times, ou2_diffusions = time_solver(init_x, gt,
                                                      lambda x, **kwargs: solve_magnani(x, min_timestep=1e-7, **kwargs),
                                                      sol_params=[{'h': h} for h in hs], torch_stack=True)

    print("Magnani IWP integration, q=2:")
    iwp2_mses, iwp2_times, iwp2_diffusions = time_solver(init_x, gt,
                                                         lambda x, **kwargs: solve_magnani(x, min_timestep=1e-7,
                                                                                           prior='IWP', **kwargs),
                                                         sol_params=[{'h': h} for h in hs], torch_stack=True)

    print("Magnani IOU integration, q=1:")
    ou1_mses, ou1_times, ou1_diffusions = time_solver(init_x, gt,
                                                      lambda x, **kwargs: solve_magnani(x, min_timestep=1e-7, q=1,
                                                                                        **kwargs),
                                                      sol_params=[{'h': h} for h in hs], torch_stack=True)

    print("Magnani IWP integration, q=1:")
    iwp1_mses, iwp1_times, iwp1_diffusions = time_solver(init_x, gt,
                                                         lambda x, **kwargs: solve_magnani(x, min_timestep=1e-7,
                                                                                           prior='IWP', q=1,
                                                                                           **kwargs),
                                                         sol_params=[{'h': h} for h in hs], torch_stack=True)

    tols = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    print("Scipy RK45 integration:")
    sci45_mses, sci45_times, sci45_diffusions = time_solver(init_x, gt,
                                                            lambda x, **kwargs: solve_scipy(x, 1e-7, method='RK45',
                                                                                            **kwargs),
                                                            sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])

    tols = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    print("Scipy RK23 integration:")
    sci23_mses, sci23_times, sci23_diffusions = time_solver(init_x, gt,
                                                            lambda x, **kwargs: solve_scipy(x, 1e-7, method='RK23',
                                                                                            **kwargs),
                                                            sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])

    plt.plot(ou1_times, ou1_mses, label='Magnani, q=1 (IOUP)')
    plt.plot(ou2_times, ou2_mses, label='Magnani, q=2 (IOUP)')
    plt.plot(iwp1_times, iwp1_mses, label='Magnani, q=1 (IWP)')
    plt.plot(iwp2_times, iwp2_mses, label='Magnani, q=2 (IWP)')
    plt.plot(euler_times, euler_mses, label='Euler')
    plt.plot(heun_times, heun_mses, label='Heun')
    plt.plot(sci45_times, sci45_mses, label='SciPy RK45')
    plt.plot(sci23_times, sci23_mses, label='SciPy RK23')

    plt.xlabel("Runtime [s]")
    plt.ylabel("MSE")
    plt.legend()

    plt.semilogy()

    plot_final_results([(ou1_diffusions[-1], 'OU,q=1'), (ou2_diffusions[-1], 'OU,q=2'),
                        (iwp1_diffusions[-1], 'IWP,q=1'), (iwp2_diffusions[-1], 'IWP,q=2'),
                        (euler_diffusions[-1], 'Euler'), (heun_diffusions[-1], 'Heun'),
                        (sci23_diffusions[-1], 'Scipy RK23'), (sci45_diffusions[-1], 'Scipy RK45')])

    plt.show()
