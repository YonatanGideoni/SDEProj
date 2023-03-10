import copy
import itertools
import os.path
import pickle
from timeit import default_timer as timer

import numpy as np
import torch
from matplotlib import pyplot as plt

from consts import BS, DEVICE, IMG_TENS_SHAPE, SIGMA
from plot_utils import plot_final_results
from song_probnum_solver import solve_scipy, euler_int, solve_magnani, second_order_heun_int
from song_utils import marginal_prob_std

N_TRIALS = 3


def time_solver(init_x: torch.Tensor, gt: torch.Tensor, solver: callable, sol_params: list, torch_stack: bool = False,
                n_trials: int = N_TRIALS):
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


def run_method(func: callable, steps_list, final_time, method_name):
    fname = f'{method_name}_{final_time}_{steps_list}_{N_TRIALS}.pkl'
    if os.path.isfile(fname):
        print(f"Found cached {method_name}. Loading...")
        with open(fname, 'rb') as f:
            loaded_dict = pickle.load(f)
            times, diffusions = loaded_dict["times"], loaded_dict["diffusions"]
        mses = [torch.nn.functional.mse_loss(torch.tensor(diffusion[:, -1].flatten()), torch.tensor(gt.flatten())) for
                diffusion in diffusions]
        mses = np.mean(np.array(mses).reshape(-1, N_TRIALS), axis=1)
    else:
        print(f"Computing {method_name} integration...")
        mses, times, diffusions = func()
        with open(fname, 'wb') as f:
            pickle.dump({
                "times": times,
                "diffusions": diffusions,
            }, f)

    return mses, times, diffusions


if __name__ == "__main__":
    # Plot the result and trajectory
    # and plot the animation over time

    # FID / some accuracy vs step size and/or vs runtime
    # Plot this graph for various solvers
    # We want to know where it starts diverging to identify phase transition

    # TODO: Use consistent torch seed
    seed = 42
    torch.manual_seed(seed)

    # Define some starting point
    t = torch.ones(BS, device=DEVICE)
    init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
    init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)

    steps_list = [10, 50, 100, 1000]
    final_time = 1e-7
    tss = [np.linspace(1.0, final_time, steps + 1) for steps in steps_list]
    tols = [1, 1e-1, 1e-2, 1e-3]

    # Ground truth
    print("Computing ground truth")
    rtol, atol = 1e-8, 1e-8
    fname = f'gt_{final_time}_{rtol}_{atol}_{seed}.pkl'
    if os.path.isfile(fname):
        print("Found ground truth data. Loading...")
        with open(fname, 'rb') as f:
            loaded_dict = pickle.load(f)
            ms, ts = loaded_dict["ms"], loaded_dict["ts"]
    else:
        print("No ground truth data found. Computing...")
        ms, ts = solve_scipy(copy.deepcopy(init_x), final_time, rtol=1e-8, atol=1e-8, method='RK45')
        with open(fname, 'wb') as f:
            pickle.dump({
                "ms": ms,
                "ts": ts,
            }, f)
    gt = ms[:-1, -1]

    # euler integration
    euler_mses, euler_times, euler_diffusions = run_method(
        method_name='euler',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: euler_int(x, **kwargs),
                                 sol_params=[{'ts': ts} for ts in tss])
    )

    heun_mses, heun_times, heun_diffusions = run_method(
        method_name='heun',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: second_order_heun_int(x, **kwargs),
                                 sol_params=[{'ts': ts} for ts in tss])
    )

    ou2_mses, ou2_times, ou2_diffusions = run_method(
        method_name='ou2',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    iwp2_mses, iwp2_times, iwp2_diffusions = run_method(
        method_name='iwp2',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, prior='IWP', **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    ou1_mses, ou1_times, ou1_diffusions = run_method(
        method_name='ou1',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, q=1, **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    iwp1_mses, iwp1_times, iwp1_diffusions = run_method(
        method_name='iwp1',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, prior='IWP', q=1,
                                                                   **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    sci45_mses, sci45_times, sci45_diffusions = run_method(
        method_name='sci45',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: solve_scipy(x, final_time,
                                                                             method='RK45',
                                                                             **kwargs),
                                 sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])
    )

    sci23_mses, sci23_times, sci23_diffusions = run_method(
        method_name='sci23',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: solve_scipy(x, final_time,
                                                                             method='RK23',
                                                                             **kwargs),
                                 sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])
    )

    marker = itertools.cycle((',', '+', '.', 'o', '*', 'd'))
    plt.plot(ou1_times, ou1_mses, label='Magnani, q=1 (IOUP)', marker=next(marker))
    plt.plot(ou2_times, ou2_mses, label='Magnani, q=2 (IOUP)', marker=next(marker))
    plt.plot(iwp1_times, iwp1_mses, label='Magnani, q=1 (IWP)', marker=next(marker))
    plt.plot(iwp2_times, iwp2_mses, label='Magnani, q=2 (IWP)', marker=next(marker))
    plt.plot(euler_times, euler_mses, label='Euler', marker=next(marker))
    plt.plot(heun_times, heun_mses, label='Heun', marker=next(marker))
    plt.plot(sci45_times, sci45_mses, label='SciPy RK45', marker=next(marker))
    plt.plot(sci23_times, sci23_mses, label='SciPy RK23', marker=next(marker))

    plt.xlabel("Runtime [s]")
    plt.ylabel("MSE")
    plt.legend()

    plt.semilogy()

    plot_final_results([(ou1_diffusions[-1], 'OU,q=1'), (ou2_diffusions[-1], 'OU,q=2'),
                        (iwp1_diffusions[-1], 'IWP,q=1'), (iwp2_diffusions[-1], 'IWP,q=2'),
                        (euler_diffusions[-1], 'Euler'), (heun_diffusions[-1], 'Heun'),
                        (sci23_diffusions[-1], 'Scipy RK23'), (sci45_diffusions[-1], 'Scipy RK45')])

    plt.show()
