import copy
import itertools
import os.path
import pickle
from timeit import default_timer as timer

import numpy as np
import torch
from matplotlib import pyplot as plt

from consts import BS, DEVICE, IMG_TENS_SHAPE, SIGMA, SMALL_NUMBER, float_dtype
from plot_utils import plot_final_results, plot_trajectory, plot_reverse_process
from song_probnum_solver import solve_scipy, euler_int, solve_magnani, second_order_heun_int
from song_utils import marginal_prob_std

N_TRIALS = 3


def time_solver(init_x: torch.Tensor, gt: torch.Tensor, solver: callable, sol_params: list, torch_stack: bool = False,
                n_trials: int = N_TRIALS):
    mses = []
    mses_stds = []
    results_over_time = []
    runtimes = []
    for params in sol_params:
        losses = []
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

            losses.append(mse_loss.item())
            runtime += end - start

        mses.append(np.mean(losses))
        mses_stds.append(np.std(losses))
        runtimes.append(runtime / n_trials)

    return mses, mses_stds, runtimes, results_over_time


def run_method(func: callable, steps_list, final_time, method_name):
    fname = f'{method_name}_{final_time}_{steps_list}_{N_TRIALS}_{SMALL_NUMBER}_{float_dtype}.pkl'
    print(fname)
    if os.path.isfile(fname):
        print(f"Found cached {method_name}. Loading...")
        with open(fname, 'rb') as f:
            loaded_dict = pickle.load(f)
            times, diffusions = loaded_dict["times"], loaded_dict["diffusions"]
        mses = [torch.nn.functional.mse_loss(torch.tensor(diffusion[:, -1].flatten()), torch.tensor(gt.flatten())) for
                diffusion in diffusions]
        stds = np.std(np.array(mses).reshape(-1, N_TRIALS), axis=1)
        mses = np.mean(np.array(mses).reshape(-1, N_TRIALS), axis=1)
    else:
        print(f"Computing {method_name} integration...")
        mses, stds, times, diffusions = func()
        with open(fname, 'wb') as f:
            pickle.dump({
                "times": times,
                "diffusions": diffusions,
            }, f)

    return mses, stds, times, diffusions


if __name__ == "__main__":
    # Plot the result and trajectory
    # and plot the animation over time

    # FID / some accuracy vs step size and/or vs runtime
    # Plot this graph for various solvers
    # We want to know where it starts diverging to identify phase transition

    # TODO: Use consistent torch seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Define some starting point
    seed_file = 'seed.pkl'
    if os.path.isfile(seed_file):
        with open(seed_file, 'rb') as f:
            init_x = pickle.load(f)
    else:
        t = torch.ones(BS, device=DEVICE)
        init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE, dtype=float_dtype) * \
                 marginal_prob_std(t, SIGMA)[:, None, None, None]
        init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)
        with open(seed_file, 'wb') as f:
            pickle.dump(init_x, f)

    steps_list = [10, 30, 50, 100, 300, 500, 750, 1000]
    final_time = 1e-7
    tss = [np.linspace(final_time, 1.0, steps + 1) for steps in steps_list]
    tols = [1, 5e-1, 1e-1, 5e-2, 1e-2, 1e-3, 5e-4, 1e-4]

    # Ground truth
    print("Computing ground truth")
    rtol, atol = 1e-8, 1e-8
    fname = f'gt_{final_time}_{rtol}_{atol}_{seed}_{SMALL_NUMBER}.pkl'
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
    euler_mses, euler_stds, euler_times, euler_diffusions = run_method(
        method_name='euler',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: euler_int(x, **kwargs),
                                 sol_params=[{'ts': ts} for ts in tss])
    )

    ou2_mses, ou2_stds, ou2_times, ou2_diffusions = run_method(
        method_name='ou2',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    iwp2_mses, iwp2_stds, iwp2_times, iwp2_diffusions = run_method(
        method_name='iwp2',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, prior='IWP', **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    ou1_mses, ou1_stds, ou1_times, ou1_diffusions = run_method(
        method_name='ou1',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, q=1, **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    iwp1_mses, iwp1_stds, iwp1_times, iwp1_diffusions = run_method(
        method_name='iwp1',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt,
                                 lambda x, **kwargs: solve_magnani(x, min_timestep=final_time, prior='IWP', q=1,
                                                                   **kwargs),
                                 sol_params=[{'steps': steps} for steps in steps_list],
                                 torch_stack=True)
    )

    sci45_mses, sci45_stds, sci45_times, sci45_diffusions = run_method(
        method_name='sci45',
        steps_list=steps_list,
        final_time=final_time,
        func=lambda: time_solver(init_x, gt, lambda x, **kwargs: solve_scipy(x, final_time,
                                                                             method='RK45',
                                                                             **kwargs),
                                 sol_params=[{'atol': tol, 'rtol': tol} for tol in tols])
    )

    marker = itertools.cycle(('s', 'x', '.', 'o', '*', 'd'))
    plt.plot(iwp1_times, iwp1_mses, label='IWP, q=1', marker=next(marker))
    plt.plot(iwp2_times, iwp2_mses, label='IWP, q=2', marker=next(marker))
    plt.plot(ou1_times, ou1_mses, label='IOUP, q=1', marker=next(marker))
    plt.plot(ou2_times, ou2_mses, label='IOUP, q=2', marker=next(marker))
    plt.plot(euler_times, euler_mses, label='Euler', marker=next(marker))
    plt.plot(sci45_times, sci45_mses, label='scipy RK45', marker=next(marker))

    fs = 18
    small_fs = 16
    plt.xlabel("Runtime [s]", fontsize=fs)
    plt.ylabel("MSE", fontsize=fs)
    plt.legend(fontsize=small_fs)
    plt.xticks(fontsize=small_fs)
    plt.yticks(fontsize=small_fs)

    plt.semilogy()
    plt.xlim(0)

    plot_final_results([(ou1_diffusions[-1], 'IOUP, q=1'), (ou2_diffusions[-1], 'IOUP, q=2'),
                        (iwp1_diffusions[-1], 'IWP, q=1'), (iwp2_diffusions[-1], 'IWP, q=2'),
                        (euler_diffusions[-1], 'Euler'), (sci45_diffusions[-1], 'scipy RK45')])

    plt.show()
